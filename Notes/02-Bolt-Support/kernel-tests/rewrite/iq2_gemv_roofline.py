#!/usr/bin/env python
"""What actually costs the time in the IQ2 GEMV? A mode-switched probe kernel.

  0 load-only   packed idx/aux/scale bytes only, no decode  -> memory floor
  1 bcast-lut   full decode, table read as one broadcast     -> gather ceiling
  2 no-dot      full decode, no activation load or multiply  -> decode + store
  3 real        the in-tree structure                        -> reference

If MODE 0 is far below MODE 3 the shape is decode-bound and there is room; if
they meet, ~250 GB/s of packed bytes is the wall and tile/dispatch tuning is
done. Results go in 04-progress-log.md.

Run from the vllm repo root:
  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/iq2_gemv_roofline.py
"""
import statistics

import torch
import triton
import triton.language as tl
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t


@triton.jit
def x_load(p, k0, BK: tl.constexpr):
    return tl.load(p + k0 + tl.arange(0, BK)).to(tl.float32)


@triton.jit
def _probe(
    x_ptr, qw_ptr, sc_ptr, lut_ptr, out_ptr, N, k_per_split, stride_xm, stride_qn,
    stride_sn, MODE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    """Same tiling as `_gemv_kernel` for BM=1; MODE only, results are garbage."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    row = offs_n.to(tl.int64)[:, None]
    R: tl.constexpr = BK // 32
    acc = tl.zeros((BN,), dtype=tl.float32)
    acc4 = tl.zeros((BN, R, 4, 8), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        rr = k0 // 32 + tl.arange(0, R)
        d = tl.load(
            sc_ptr + row * stride_sn + (rr // 8)[None, :], mask=n_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        idx = tl.load(qw_ptr + row * stride_qn + (2 * rr)[None, :], mask=n_mask[:, None], other=0)
        aux = tl.load(qw_ptr + row * stride_qn + (2 * rr + 1)[None, :], mask=n_mask[:, None], other=0)
        if MODE == 0:
            acc += tl.sum((idx + aux).to(tl.float32) * d, axis=1)
        else:
            rs = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.25
            g = tl.arange(0, 4)[None, None, :]
            gidx = (idx[:, :, None] >> (8 * g)) & 255
            s = (aux[:, :, None] >> (7 * g)) & 127
            p = s ^ (s >> 4)
            p = p ^ (p >> 2)
            p = p ^ (p >> 1)
            smask = s | ((p & 1) << 7)
            j = tl.arange(0, 8)[None, None, None, :]
            if MODE == 1:  # loop-varying but uniform address: not a gather
                mag = tl.load(lut_ptr + (k0 & 3) * 8 + j).to(tl.float32, bitcast=True)
            else:
                mag = tl.load(lut_ptr + gidx[:, :, :, None] * 8 + j).to(tl.float32, bitcast=True)
            if MODE == 8:  # parity bit skipped (wrong signs, timing only)
                smask = s
            if MODE == 9:  # accumulate the 4-D tile: no per-step reshape/reduce
                neg = (smask[:, :, :, None] >> j) & 1
                xb = tl.reshape(x_load(x_ptr, k0, BK), (1, R, 4, 8))
                acc4 += tl.where(neg != 0, -mag, mag) * rs[:, :, None, None] * xb
            elif MODE == 10:  # same, scale applied per 32-block after the dot
                neg = (smask[:, :, :, None] >> j) & 1
                xb = tl.reshape(x_load(x_ptr, k0, BK), (1, R, 4, 8))
                part = tl.sum(tl.reshape(tl.where(neg != 0, -mag, mag) * xb,
                                         (BN, R, 32)), axis=2)  # fmt: skip
                acc += tl.sum(part * rs, axis=1)
            elif MODE == 5:  # no sign at all (wrong values, timing only)
                w = tl.reshape(mag * rs[:, :, None, None], (BN, BK))
            elif MODE == 6:  # sign, no per-block scale
                neg = (smask[:, :, :, None] >> j) & 1
                w = tl.reshape(tl.where(neg != 0, -mag, mag), (BN, BK))
            elif MODE == 7:  # sign by XOR on the fp32 sign bit, no negate/select
                sb = smask[:, :, :, None] << (31 - j)
                bits = tl.cast(mag, tl.int32, bitcast=True) ^ sb
                w = tl.reshape(
                    tl.cast(bits, tl.float32, bitcast=True) * rs[:, :, None, None],
                    (BN, BK),
                )
            else:
                neg = (smask[:, :, :, None] >> j) & 1
                w = tl.reshape(
                    tl.where(neg != 0, -mag, mag) * rs[:, :, None, None], (BN, BK)
                )
            if MODE >= 9:  # these two accumulate in their own branch above
                pass
            elif MODE == 2:
                acc += tl.sum(w, axis=1)
            else:
                x = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
                acc += tl.sum(w * x[None, :], axis=1)
    if MODE == 9:
        acc = tl.sum(tl.sum(tl.reshape(acc4, (BN, R, 32)), axis=2), axis=1)
    tl.store(out_ptr + pid_k * tl.num_programs(0) * BN + offs_n, acc, mask=n_mask)


def us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


torch.set_default_device("cuda")
torch.manual_seed(0)
lut = t.iq2_lut(torch.device("cuda"))
print(f"{'shape':26s} {'MB':>5s} " + " ".join(f"{m:>11s}" for m in
      ["load-only", "bcast-lut", "no-dot", "real", "no-sign", "no-scale",
       "xor-sign", "no-parity", "acc4d", "acc4d-fold", "fresh-out",
       "launch_linear"]))
for name, N, K, bn, bk, split in [
    ("shared_up 5376x4096", 5376, 4096, 8, 256, 8),
    ("shared_down 4096x5376", 4096, 5376, 8, 256, 7),
    ("expert_up 2048x1024", 2048, 1024, 8, 256, 1),
]:
    qweight, scales = (x.contiguous() for x in t.repack_blocks(
        torch.randint(0, 256, (N, K // 256, 66), dtype=torch.uint8), "iq2_xxs"))
    x = torch.randn(1, K, dtype=torch.bfloat16)
    tiles = triton.cdiv(N, bn)
    out = torch.empty((split * tiles * bn,), dtype=torch.float32, device="cuda")
    mb = N * K * 66 / 256 / 1e6
    times = []
    for mode in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10):
        times.append(us(lambda m=mode: _probe[(tiles, split)](
            x, qweight, scales, lut, out, N, K // split, x.stride(0),
            qweight.stride(0), scales.stride(0), MODE=m, BN=bn, BK=bk,
            num_warps=1, num_stages=2,
        )))  # fmt: skip
    # Fresh output buffer per call, like `launch_linear` does: whether the
    # split-K partials land on a hot or a rotating L2 line.
    fresh = us(lambda: _probe[(tiles, split)](
        x, qweight, scales, lut,
        torch.empty((split * tiles * bn,), dtype=torch.float32, device="cuda"),
        N, K // split, x.stride(0), qweight.stride(0), scales.stride(0),
        MODE=3, BN=bn, BK=bk, num_warps=1, num_stages=2,
    ))  # fmt: skip
    real = us(lambda: t.launch_linear(
        x, qweight, scales, lut, "iq2_xxs",
        {"BM": 1, "BN": bn, "BK": bk, "SPLIT": split, "num_warps": 1,
         "num_stages": 2}))
    print(f"{name:26s} {mb:5.1f} " + " ".join(f"{v:6.1f} ({mb/v*1e3:3.0f})"
                                              for v in [*times, fresh, real]))