#!/usr/bin/env python
"""TensorFold/llama.cpp-style IQ2 decode vs our fp32 gather decode, A/B interleaved.

What the three research lanes found, independently: nobody turns an IQ2 magnitude
into a float per value. llama.cpp keeps the codebook row as packed int8 and uses
8x __dp4a (vecdotq.cuh:1105-1141); Strata folds grid+sign into one __byte_perm
selector (moe_ws_gu_iq2.cu:128-168); TensorFold builds packed bf16 magnitudes,
signs by XOR, and applies the per-32 scale on the fp32 accumulator after a bf16
mma (ggml.cu:24-42,140-170).

Triton has no PRMT/dp4a, but the byte tricks are not the copyable part: our
gather is already free, so what transfers is that the DECODED TILE should already
carry its sign, and be narrow (bf16) rather than fp32 + a negate chain. Variants,
same tiling and split-K as `_gemv_kernel`:

  V1   in-tree `_gemv_kernel`                       -> baseline
  V2s  signed 512-entry fp32 table, no negate chain -> isolates "table has sign"
  V2b  signed 512-entry bf16 table                  -> isolates tile width/regs
  V3   signed bf16 tile + tl.dot, per-32 scale on the fp32 result of each 32-value
       group (TensorFold's structure: weights stay exact integers into the tensor
       core), BN=16/BK=32

Timings are interleaved variant by variant so box drift cannot fake a win, and
each variant is checked against V1's numbers on the same tensors.

Run from the vllm repo root:
  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/bf16_decode_ab.py
"""
import statistics

import torch
import triton
import triton.language as tl
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

SHAPES = [("shared_up", 5376, 4096), ("shared_down", 4096, 5376), ("expert_up", 2048, 1024)]


def signed_tables(device: torch.device):
    """512*8 table: rows 0-255 the grid magnitudes, 256-511 the same negated.
    bf16 bits (int16) and fp32, same index -> `index | sign_bit << 8`."""
    grid = torch.tensor(t._IQ2XXS_GRID, dtype=torch.int64, device=device)
    mags = (grid[:, None] >> (8 * torch.arange(8, device=device))) & 0xFF
    bf = torch.cat([mags, -mags], 0).to(torch.bfloat16).view(torch.int16)
    f32 = torch.cat([mags, -mags], 0).to(torch.float32)
    return bf.flatten().contiguous(), f32.flatten().contiguous()


@triton.jit
def _tabbed(
    qw_ptr, sc_ptr, tab_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
    SIGNED16: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """IQ2 tile with the sign inside the table: build one index per value (2
    shifts + an OR), one gather, and the value is done - no negate, no select.
    Returns the [BN, BK] tile and the [BN, BK] per-32 scale broadcast."""
    R: tl.constexpr = BK // 32
    rr = k0 // 32 + tl.arange(0, R)
    row = offs_n.to(tl.int64)[:, None]
    d = tl.load(sc_ptr + row * stride_sn + (rr // 8)[None, :], mask=n_mask[:, None],
                other=0.0).to(tl.float32)  # fmt: skip
    idx = tl.load(qw_ptr + row * stride_qn + 2 * rr[None, :], mask=n_mask[:, None], other=0)
    aux = tl.load(qw_ptr + row * stride_qn + (2 * rr + 1)[None, :], mask=n_mask[:, None], other=0)
    rs = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.25
    g = tl.arange(0, 4)[None, None, :]
    gidx = (idx[:, :, None] >> (8 * g)) & 255
    s = (aux[:, :, None] >> (7 * g)) & 127
    p = s ^ (s >> 4)
    p = p ^ (p >> 2)
    p = p ^ (p >> 1)
    smask = s | ((p & 1) << 7)
    j = tl.arange(0, 8)[None, None, None, :]
    sel = gidx[:, :, :, None] | (((smask[:, :, :, None] >> j) & 1) << 8)
    if SIGNED16:
        w = tl.reshape(tl.load(tab_ptr + sel * 8 + j).to(tl.bfloat16, bitcast=True), (BN, BK))
    else:
        w = tl.reshape(tl.load(tab_ptr + sel * 8 + j), (BN, BK))
    rs_full = tl.reshape(tl.broadcast_to(rs[:, :, None, None], (BN, R, 8, 4)), (BN, BK))
    return w, rs_full


@triton.jit(do_not_specialize=["M"])
def _v2_kernel(
    x_ptr, qw_ptr, sc_ptr, tab_ptr, out_ptr, M, N, k_per_split,
    stride_xm, stride_qn, stride_sn, SIGNED16: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    """In-tree fp32 accumulate, table-folded decode."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        w, rs = _tabbed(qw_ptr, sc_ptr, tab_ptr, offs_n, n_mask, k0, stride_qn,
                        stride_sn, SIGNED16, BN, BK)  # fmt: skip
        x = tl.load(x_ptr + k0 + tl.arange(0, BK)[None, :]).to(tl.float32)
        acc += tl.sum(w.to(tl.float32) * rs * x, axis=1)
    tl.store(out_ptr + pid_k * N + offs_n, acc, mask=n_mask)


@triton.jit(do_not_specialize=["M"])
def _v3_kernel(
    x_ptr, qw_ptr, sc_ptr, tab_ptr, out_ptr, M, N, k_per_split,
    stride_xm, stride_qn, stride_sn, BN: tl.constexpr, BK: tl.constexpr,
):
    """TensorFold structure with BK = one 32-value group: exact integer
    magnitudes as bf16 into the tensor core, scale applied once on the fp32
    result. Column 0 of the [BN, 16] result is the answer (x is broadcast)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    BQ: tl.constexpr = 16
    acc = tl.zeros((BN, BQ), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        w, rs = _tabbed(qw_ptr, sc_ptr, tab_ptr, offs_n, n_mask, k0, stride_qn,
                        stride_sn, True, BN, BK)  # fmt: skip
        x = tl.load(x_ptr + k0 + tl.arange(0, BK)[:, None])
        p = tl.dot(w, tl.broadcast_to(x, (BK, BQ)))
        # BK == 32 -> one scale per row; rs is that scale broadcast to 32.
        acc += p * tl.max(rs, axis=1)[:, None]
    q = tl.arange(0, BQ)[None, :]
    tl.store(out_ptr + pid_k * N + offs_n, tl.sum(tl.where(q == 0, acc, 0.0), axis=1),
             mask=n_mask)  # fmt: skip


def us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    sms, smem = props.multi_processor_count, props.shared_memory_per_block_optin
    lut = t.iq2_lut(dev)
    tab16, tab32 = signed_tables(dev)

    for tag, N, K in SHAPES:
        raw = torch.randint(0, 256, (N, K // 256, 66), dtype=torch.uint8)
        # Random bytes make the fp16 block scale NaN/Inf, which poisons every
        # numeric comparison; the tests' make_raw does the same thing.
        raw[:, :, 0:2] = (torch.randn(N, K // 256, 1) * 2e-3).half().view(torch.uint8)
        qweight, scales = (x.contiguous() for x in t.repack_blocks(raw, "iq2_xxs"))
        cfg = t._linear_config(1, N, K, "iq2_xxs", sms, smem)
        bn, bk, split = cfg["BN"], cfg["BK"], cfg["SPLIT"]
        x = torch.randn(1, K, dtype=torch.bfloat16)
        outs, fns = {}, {}

        def build(kind, bn=bn, bk=bk, split=split):
            buf = torch.empty((split, N), dtype=torch.float32)
            outs[kind] = buf
            b, k_, sp = (16, 32, max(1, K // 32 // 128)) if kind == "V3" else (bn, bk, split)
            grid = (triton.cdiv(N, b), sp)
            common = (x, qweight, scales, 1, N, K // sp, x.stride(0),
                      qweight.stride(0), scales.stride(0))
            if kind == "V1":
                return lambda: t._gemv_kernel[grid](*common[:3], lut, buf, *common[3:],
                                                    CODEC=1, BM=1, BN=b, BK=k_,
                                                    num_warps=1, num_stages=2)
            tab = tab16 if kind in ("V2b", "V3") else tab32
            if kind == "V3":
                return lambda: _v3_kernel[grid](*common[:3], tab, buf, *common[3:],
                                                BN=b, BK=k_, num_warps=1, num_stages=2)
            return lambda: _v2_kernel[grid](*common[:3], tab, buf, *common[3:],
                                            SIGNED16=kind == "V2b", BN=b, BK=k_,
                                            num_warps=1, num_stages=2)

        for kind in ("V1", "V2s", "V2b", "V3"):
            try:
                fns[kind] = build(kind)
                fns[kind]()
            except Exception as e:
                print(f"{kind} unavailable: {type(e).__name__}: {str(e)[-700:]}")
        torch.cuda.synchronize()

        print(f"\n=== {tag} N={N} K={K} BN={bn} BK={bk} SPLIT={split} ===")
        ref = outs["V1"].sum(0).double()
        for kind in fns:
            got = outs[kind].sum(0).double()
            rel = ((got - ref).abs().max() / ref.abs().max()).item()
            print(f"  {kind:4s} max rel diff vs V1: {rel:.2e}")
        res = {k: [] for k in fns}
        for _ in range(3):
            for kind, f in fns.items():
                res[kind].append(us(f))
        base = sorted(res["V1"])[1]
        for kind in fns:
            v = sorted(res[kind])[1]
            print(f"  {kind:4s} {v:7.1f} us   {base / v:5.2f}x vs V1")


if __name__ == "__main__":
    main()
