"""Lossless expanded IQ2: exact fp32 decode and interleaved cold-L2 A/B.

Run from vllm with CUDA_VISIBLE_DEVICES=1. --sweep tunes expanded GEMV tiles.
"""

import argparse
import statistics
import sys

sys.path.insert(0, ".")
import torch
import triton
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import make_raw, ref_decode
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", action="store_true")
args = ap.parse_args()
torch.set_default_device("cuda")
torch.manual_seed(0)
p = torch.cuda.get_device_properties(0)
lut = t.iq2_lut(torch.device("cuda"))


def us(fn):
    for _ in range(10):
        fn()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1000


def fp32_decode(q, s, k):
    out = torch.empty((q.shape[0], k), dtype=torch.float32)
    t._dequant_kernel[(triton.cdiv(q.shape[0], 8), k // 256)](
        q,
        s,
        lut,
        out,
        q.shape[0],
        k,
        q.stride(0),
        s.stride(0),
        CODEC=t._codec_id("iq2_xxs", s),
        BN=8,
        BK=256,
        num_warps=1,
    )
    return out


for n, k in [(5376, 4096), (4096, 5376), (2048, 1024), (1024, 2048)]:
    raw = make_raw((n,), k, "iq2_xxs", 0)
    q, s = t.repack_blocks(raw, "iq2_xxs")
    e, es = t.expand_iq2(q, s, lut)
    assert torch.equal(fp32_decode(e, es, k), ref_decode(raw, "iq2_xxs"))
    for m in (1, 4) if args.sweep else (1, 2, 4, 8, 32, 128):
        x = torch.randn(m, k, dtype=torch.bfloat16)
        cfg = t._linear_config(
            m, n, k, "iq2_xxs", p.multi_processor_count, p.shared_memory_per_block_optin
        )
        a = lambda: t._modelopt_block_linear(x, q, s, lut, "iq2_xxs")
        b = lambda: t._modelopt_block_linear(x, e, es, lut, "iq2_xxs")
        ya, yb = a(), b()
        torch.testing.assert_close(ya, yb, rtol=0, atol=ya.abs().max().item() / 256)
        av, bv = [], []
        for i in range(3):
            if i % 2:
                bv.append(us(b))
                av.append(us(a))
            else:
                av.append(us(a))
                bv.append(us(b))
        av, bv = statistics.median(av), statistics.median(bv)
        print(
            n,
            k,
            m,
            "fp32_decode_exact",
            "agreement",
            (ya == yb).float().mean().item(),
            round(av, 2),
            round(bv, 2),
            "speedup",
            round(av / bv, 3),
            flush=True,
        )
        if args.sweep:
            best = (bv, cfg)
            candidates = []
            for bn, w in ((4, 1), (8, 1), (16, 1), (32, 4)):
                for bk in (128, 256, 512):
                    for split in (1, 2, 3, 4, 7, 8, 16):
                        if k % (bk * split):
                            continue
                        c = dict(
                            BM=triton.next_power_of_2(m),
                            BN=bn,
                            BK=bk,
                            SPLIT=split,
                            num_warps=w,
                            num_stages=2,
                        )
                        fn = lambda c=c: t.launch_linear(x, e, es, lut, "iq2_xxs", c)
                        yy = fn()
                        torch.testing.assert_close(
                            yy, ya, rtol=0, atol=ya.abs().max().item() / 128
                        )
                        val = us(fn)
                        candidates.append((val, c))
            for val, c in sorted(candidates, key=lambda z: z[0])[:5]:
                print("BEST", n, k, m, round(val, 2), c, flush=True)
