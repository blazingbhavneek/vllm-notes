"""Dispatched op vs the better of fused / dequant+cuBLAS, BF16 and FP16."""

import sys
from functools import partial

import torch

sys.path.insert(0, "benchmarks/kernels")
from benchmark_modelopt_block_triton import BLOCK, DENSE, bench_us  # noqa: E402

from vllm.model_executor.layers.quantization import (  # noqa: E402
    modelopt_block_quant_triton as t,
)

torch.set_default_device("cuda")
torch.manual_seed(0)
p = torch.cuda.get_device_properties()
print(p.name, "bus", p.memory_bus_width)
print(f"{'layer':12s} {'dtype':8s} {'M':>4s} {'op':>7s} {'fused':>7s} {'deq':>7s} op/best")
for name, codec, N, K in DENSE:
    if name == "lm_head":
        continue
    size, nbytes = BLOCK[codec]
    raw = torch.randint(0, 256, (N, K // size, nbytes), dtype=torch.uint8)
    raw[..., :2] = (torch.randn(N, K // size) * 2e-3).half().unsqueeze(-1).view(
        torch.uint8
    )
    qw, sc = t.repack_blocks(raw, codec)
    lut = t.iq2_lut(qw.device) if codec == "iq2_xxs" else None
    for dtype in (torch.bfloat16, torch.float16):
        for M in (8, 32, 64, 128, 256, 512):
            x = torch.randn(M, K, dtype=dtype)
            cfg = t._linear_config(M, N, K, codec, p.multi_processor_count,
                                   p.shared_memory_per_block_optin, p.major)
            uf = bench_us(partial(t.launch_linear, x, qw, sc, lut, codec, cfg))
            ud = bench_us(lambda: x @ t.dequantize_blocks(qw, sc, lut, codec, dtype).T)
            uo = bench_us(partial(t._modelopt_block_linear, x, qw, sc, lut, codec))
            print(f"{name:12s} {str(dtype)[6:]:8s} {M:4d} {uo:7.1f} {uf:7.1f} "
                  f"{ud:7.1f} {uo / min(uf, ud):6.2f}", flush=True)
