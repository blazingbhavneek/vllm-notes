"""Packed BF16x2 IQ2 decode with exact weights and BF16 activations.

Four weights per logical Triton element, using PRMT/sub.rn.bf16x2 rather than
per-value table lookups. Run from vllm, CUDA_VISIBLE_DEVICES=1.
"""

import argparse
import statistics
import sys

sys.path.insert(0, ".")
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import (
    make_raw,
    ref_decode,
    assert_close_to_cublas,
)
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t


@triton.jit
def packed_dot4(gw, sg, rs, x0, x1):
    return tl.inline_asm_elementwise(
        asm="""{
            .reg .b32 p0,p1,s0,s1,c128;
            mov.b32 c128,0x43004300;
            .reg .f32 w0,w1,w2,w3,x0,x1,x2,x3,v;
            prmt.b32 p0, $1, 0x43434343, 0x4140;
            prmt.b32 p1, $1, 0x43434343, 0x4342;
            sub.rn.bf16x2 p0, p0, c128;
            sub.rn.bf16x2 p1, p1, c128;
            and.b32 s0, $2, 3;
            shr.u32 s1, $2, 2;
            mul.lo.u32 s0, s0, 0x40008000;
            mul.lo.u32 s1, s1, 0x40008000;
            and.b32 s0, s0, 0x80008000;
            and.b32 s1, s1, 0x80008000;
            xor.b32 p0,p0,s0;
            xor.b32 p1,p1,s1;
            shl.b32 w0,p0,16;
            and.b32 w1,p0,0xffff0000;
            shl.b32 w2,p1,16;
            and.b32 w3,p1,0xffff0000;
            shl.b32 x0,$4,16;
            and.b32 x1,$4,0xffff0000;
            shl.b32 x2,$5,16;
            and.b32 x3,$5,0xffff0000;
            mul.f32 w0,w0,$3;
            mul.f32 w1,w1,$3;
            mul.f32 w2,w2,$3;
            mul.f32 w3,w3,$3;
            mul.f32 v,w0,x0;
            fma.rn.f32 v,w1,x1,v;
            fma.rn.f32 v,w2,x2,v;
            fma.rn.f32 $0,w3,x3,v;
            }""",
        constraints="=f,r,r,f,r,r",
        args=[gw, sg, rs, x0, x1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def packed_kernel(
    xp, qp, sp, gp, op, N, Kpart, SQ, SS, BN: tl.constexpr, BK: tl.constexpr
):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    group = tl.arange(0, BK // 4)
    acc = tl.zeros((BN, BK // 4), tl.float32)
    for k0 in range(part * Kpart, (part + 1) * Kpart, BK):
        k4 = k0 // 4 + group
        rr = k4 // 8
        idx = tl.load(
            qp + ns.to(tl.int64)[:, None] * SQ + 2 * rr[None, :],
            ns[:, None] < N,
            other=0,
        )
        aux = tl.load(
            qp + ns.to(tl.int64)[:, None] * SQ + 2 * rr[None, :] + 1,
            ns[:, None] < N,
            other=0,
        )
        d = tl.load(
            sp + ns.to(tl.int64)[:, None] * SS + (rr // 8)[None, :],
            ns[:, None] < N,
            other=0.0,
        ).to(tl.float32)
        rs = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.25
        g = (k4 % 8) // 2
        ix = (idx >> (8 * g[None, :])) & 255
        sg = (aux >> (7 * g[None, :])) & 127
        parity = sg ^ (sg >> 4)
        parity ^= parity >> 2
        parity ^= parity >> 1
        sg |= (parity & 1) << 7
        sg = (sg >> (4 * (k4 % 2)[None, :])) & 15
        gw = tl.load(gp + ix * 2 + (k4 % 2)[None, :])
        x0 = tl.load(xp.to(tl.pointer_type(tl.int32)) + 2 * k4)
        x1 = tl.load(xp.to(tl.pointer_type(tl.int32)) + 2 * k4 + 1)
        dot = packed_dot4(gw, sg, rs, x0[None, :], x1[None, :])
        acc += dot
    val = tl.sum(acc, 1)
    tl.store(op + part * N + ns, val, ns < N)


def us(fn):
    for _ in range(10):
        fn()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    grid = torch.tensor(t._IQ2XXS_GRID, dtype=torch.int64).view(torch.int32)
    lut = t.iq2_lut(torch.device("cuda"))
    props = torch.cuda.get_device_properties(0)
    for N, K in [(5376, 4096), (4096, 5376), (2048, 1024), (1024, 2048)]:
        raw = make_raw((N,), K, "iq2_xxs", 0)
        q, s = t.repack_blocks(raw, "iq2_xxs")
        x = torch.randn(1, K, dtype=torch.bfloat16)
        cfg = t._linear_config(
            1,
            N,
            K,
            "iq2_xxs",
            props.multi_processor_count,
            props.shared_memory_per_block_optin,
        )
        baseline = lambda: t.launch_linear(x, q, s, lut, "iq2_xxs", cfg)
        ref = baseline()
        a = us(baseline)
        candidates = []
        tiles = [(cfg["BN"], cfg["BK"], cfg["SPLIT"], 1)]
        if args.sweep:
            tiles = [
                (bn, bk, sk, warps)
                for bn in (1, 2, 4, 8, 16)
                for bk in (128, 256, 512)
                for sk in (1, 2, 3, 4, 7, 8, 16)
                for warps in (1,)
                if K % (bk * sk) == 0
            ]
        for bn, bk, sk, warps in tiles:
            buf = torch.empty((sk, 1, N), dtype=torch.float32)
            out = torch.empty((1, N), dtype=torch.bfloat16)

            def run():
                packed_kernel[(triton.cdiv(N, bn), sk)](
                    x,
                    q,
                    s,
                    grid,
                    buf,
                    N,
                    K // sk,
                    q.stride(0),
                    s.stride(0),
                    BN=bn,
                    BK=bk,
                    num_warps=warps,
                    num_stages=2,
                )
                t._splitk_reduce_kernel[(triton.cdiv(N, 1024),)](
                    buf, out, N, sk, BN=1024, num_warps=4
                )
                return out

            got = run()
            assert_close_to_cublas(got, x, ref_decode(raw, "iq2_xxs"))
            torch.testing.assert_close(
                got, ref, rtol=0, atol=ref.abs().max().item() / 256
            )
            b = us(run)
            candidates.append((b, (bn, bk, sk, warps)))
        for b, conf in sorted(candidates)[:5]:
            print(
                N,
                K,
                "base",
                round(a, 2),
                "packed",
                round(b, 2),
                "speedup",
                round(a / b, 3),
                conf,
                flush=True,
            )


if __name__ == "__main__":
    main()
