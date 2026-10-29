#!/usr/bin/env python
"""Is a codebook-free weight representation worth its bytes?

IQ2_XXS stores `w = sign * mag{8,25,43} * rs` (rs = per-32 scale): 2 bits of
magnitude + 1 bit of sign per value, packed into 2.0625 bits/value by a 256-entry
codebook of 8-value patterns plus a parity-folded sign. The codebook is where the
density comes from, and the gather + parity chain is where the decode time goes,
so the one remaining speed/size lever is the trade between them: a "3-bit plane"
rep (2-bit magnitudes and 1-bit signs in separate planes, no codebook, no
parity) decodes in two selects, at ~1.5x the bytes.

This measures the trade instead of guessing at it:

  1. exactness - rebuild the 3-bit planes from REAL checkpoint blocks and check
     the plane decode equals `ref_decode` bit for bit.
  2. speed     - same tiling and split-K as `_gemv_kernel`, plane decode vs
     in-tree `launch_linear`, interleaved so box drift cannot fake a win.

Layout: codes uint16 [N, K/8] (8 values, 2 bits each), signs uint8 [N, K/8]
(8 values, 1 bit each), scale fp16 [N, K/32] -> 3.0625 bits/value.

Run from the vllm repo root:
  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/plane3_ab.py
"""
import json
import os
import statistics
import sys

import torch
import triton
import triton.language as tl
from flashinfer.testing import bench_gpu_time_with_cupti
from safetensors import safe_open

sys.path.insert(0, os.getcwd())  # repo root, for the test module's own imports
sys.path.insert(0, "tests/kernels/quantization")
from test_modelopt_block_triton import ref_decode  # noqa: E402

from vllm.model_executor.layers.quantization import (  # noqa: E402
    modelopt_block_quant_triton as t,
)

CKPT = "/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
MAGS = torch.tensor([8.0, 25.0, 43.0], device="cuda")


@triton.jit
def _plane3_kernel(
    x_ptr, code_ptr, sign_ptr, sc_ptr, out_ptr, N, k_per_split,
    stride_cn, stride_sn, stride_sc, BN: tl.constexpr, BK: tl.constexpr,
):
    """`_gemv_kernel` for BM=1 against a codebook-free weight rep."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    row = offs_n.to(tl.int64)[:, None]
    R: tl.constexpr = BK // 32
    j8 = tl.arange(0, 8)[None, None, None, :]
    # [R, 4] word/group offsets: 4 words per 32-value scale group.
    woffs = (tl.arange(0, R)[:, None] * 4 + tl.arange(0, 4)[None, :])[None, :, :]
    xoffs = tl.arange(0, BK)[None, :]
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        rs = tl.load(sc_ptr + row * stride_sc + k0 // 32 + tl.arange(0, R)[None, :],
                     mask=n_mask[:, None], other=0.0).to(tl.float32)  # fmt: skip
        cd = tl.load(code_ptr + row * stride_cn + k0 // 8 + woffs,
                     mask=n_mask[:, None], other=0)  # fmt: skip
        sg = tl.load(sign_ptr + row * stride_sn + k0 // 8 + woffs,
                     mask=n_mask[:, None], other=0)  # fmt: skip
        x = tl.load(x_ptr + k0 + xoffs).to(tl.float32)
        c = (cd.to(tl.int32)[:, :, :, None] >> (2 * j8)) & 3
        s = (sg.to(tl.int32)[:, :, :, None] >> j8) & 1
        w = tl.reshape(tl.where(c == 0, 8.0, tl.where(c == 1, 25.0, 43.0))
                       * (1 - 2 * s) * rs[:, :, None, None], (BN, BK))
        acc += tl.sum(w * x, axis=1)
    tl.store(out_ptr + pid_k * N + offs_n, acc, mask=n_mask)


def us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


def to_planes(raw):
    """Real ggml IQ2_XXS blocks -> (codes u16, signs u8, scales fp16), exactly."""
    N = raw.shape[0]
    w = ref_decode(raw, "iq2_xxs")  # [N, K] fp32 ground truth
    K = w.shape[1]
    blocks = raw.reshape(-1, 66)
    d = blocks[:, :2].contiguous().view(torch.float16).float().reshape(-1, 1)
    aux = blocks[:, 2:].contiguous().view(torch.int32).reshape(-1, 8, 2)[..., 1]
    rs = (d * (0.5 + ((aux >> 28) & 0xF).float()) * 0.25).reshape(N, -1)
    rs_rep = rs.repeat_interleave(32, 1)
    live = rs_rep > 0
    r = w.abs() / torch.where(live, rs_rep, torch.ones_like(rs_rep))
    c = torch.argmin((r[:, :, None] - MAGS).abs(), dim=2)
    c = torch.where(live, c, torch.zeros_like(c)).to(torch.uint8)
    off = ((r - MAGS[c.long()]).abs() * live).max()
    assert off < 1e-3, f"magnitudes are not on the 8/25/43 grid: {off}"
    sgn = (w < 0).to(torch.uint8)
    dec = MAGS[c.long()] * (1.0 - 2.0 * sgn.to(torch.float32)) * rs_rep
    assert torch.equal(dec, w), "3-bit plane rep is not lossless"
    bits = torch.arange(8, device="cuda", dtype=torch.uint8)
    codes = ((c.reshape(N, -1, 8).to(torch.int32) << (2 * bits))
             .sum(2).to(torch.uint16))  # fmt: skip
    signs = (sgn.reshape(N, -1, 8) << bits).sum(2).to(torch.uint8)
    return codes, signs, rs.half(), N, K


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    sms, smem = props.multi_processor_count, props.shared_memory_per_block_optin

    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    name = "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight"
    with safe_open(f"{CKPT}/{idx[name]}", framework="pt") as f:
        codes, signs, rs32, N, K = to_planes(f.get_tensor(name).cuda())
    bpv = 8.0 * (2 * codes.numel() + signs.numel() + 2 * rs32.numel()) / (N * K)
    print(f"real shared_experts.up_proj {N}x{K}: the codebook-free 3-bit plane rep "
          f"is BIT-EXACT vs ref_decode, {bpv:.4f} bits/value vs 2.0625")

    lut = t.iq2_lut(torch.device("cuda"))
    for tag, n, k in [("shared_up", 5376, 4096), ("shared_down", 4096, 5376),
                   ("expert_up", 2048, 1024), ("expert_down", 1024, 2048)]:
        qweight, scales = (
            x.contiguous()
            for x in t.repack_blocks(
                torch.randint(0, 256, (n, k // 256, 66), dtype=torch.uint8), "iq2_xxs"
            )
        )
        cfg = t._linear_config(1, n, k, "iq2_xxs", sms, smem)
        bn, bk, split = cfg["BN"], cfg["BK"], cfg["SPLIT"]
        x = torch.randn(1, k, dtype=torch.bfloat16)
        pc = torch.randint(0, 65536, (n, k // 8), dtype=torch.uint16)
        ps = torch.randint(0, 256, (n, k // 8), dtype=torch.uint8)
        psc = (torch.rand(n, k // 32) * 1e-3).half()
        pout = torch.empty((split, 1, n), dtype=torch.float32)
        py = torch.empty((1, n), dtype=torch.bfloat16)

        def plane():
            _plane3_kernel[(triton.cdiv(n, bn), split)](
                x, pc, ps, psc, pout, n, k // split, k // 8, k // 8, k // 32,
                BN=bn, BK=bk, num_warps=1, num_stages=2,
            )
            # The same split-K reduce `launch_linear` runs, so the A/B is fair.
            t._splitk_reduce_kernel[(triton.cdiv(n, 1024),)](
                pout, py, n, split, BN=1024, num_warps=4
            )
        tree = lambda: t.launch_linear(x, qweight, scales, lut, "iq2_xxs", cfg)  # noqa: E731
        plane(), tree(), torch.cuda.synchronize()
        a = sorted(us(plane) for _ in range(3))[1]
        b = sorted(us(tree) for _ in range(3))[1]
        pmb, tmb = n * (2 * k / 8 + k / 8 + 2 * k / 32) / 1e6, n * k * 66 / 256 / 1e6
        print(f"\n{tag} N={n} K={k} M=1 BN={bn} BK={bk} SPLIT={split}")
        print(f"  3-bit plane : {a:6.1f} us  {pmb:5.1f} MB  {pmb / a * 1e3:5.0f} GB/s")
        print(f"  in-tree IQ2 : {b:6.1f} us  {tmb:5.1f} MB  {tmb / b * 1e3:5.0f} GB/s")
        print(f"  plane time / in-tree time = {a / b:.2f}  for {pmb / tmb:.2f}x bytes")
        print("  (both sides include the split-K reduce kernel)")


if __name__ == "__main__":
    main()
