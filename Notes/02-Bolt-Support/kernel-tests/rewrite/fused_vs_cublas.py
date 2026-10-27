"""Fused GEMM (BM64 / BM128) vs dequant+cuBLAS vs bf16 cuBLAS, for M > 64.

Settles the hard-rule 0.2.2 question (can the fused GEMM alone stay within 2x
of bf16 at M >= 2048?) and tunes the `M > 64` row of `_linear_config`.
Ratios are time / bf16 cuBLAS. Run from the vllm repo root:

    python <this file> [M ...]
"""

import sys

import torch

sys.path.insert(0, "benchmarks/kernels")
import benchmark_modelopt_block_triton as b
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

torch.set_default_device("cuda")
sms = torch.cuda.get_device_properties().multi_processor_count
Ms = [int(m) for m in sys.argv[1:]] or [96, 128, 256, 512, 2048, 8192]
for name, codec, N, K in b.DENSE:
    q, s = b.random_blocks((N,), K, codec)
    lut = t.iq2_lut(q.device) if codec == "iq2_xxs" else None
    w = t.dequantize_blocks(q, s, lut, codec)
    for M in Ms:
        if name == "lm_head" and M > 512:
            continue
        x = torch.randn(M, K, dtype=torch.bfloat16)
        ref = x @ w.T
        cub = b.bench_us(lambda: x @ w.T)
        deq = b.bench_us(lambda: x @ t.dequantize_blocks(q, s, lut, codec).T)
        row = [f"{name:12} M={M:5}", f"cublas {cub:8.1f}us", f"dequant {deq / cub:.2f}"]
        for smem in (101376, 232448):  # forces the BM64 / BM128 row
            cfg = t._linear_config(M, N, K, codec, sms, smem)
            us = b.try_time(lambda: t.launch_linear(x, q, s, lut, codec, cfg), ref)
            row.append(f"fused BM{cfg['BM']} " + (f"{us / cub:.2f}" if us else "OOR"))
        print(" | ".join(row), flush=True)
