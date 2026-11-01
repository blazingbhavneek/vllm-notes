"""Routed-MoE GEMV vs tl.dot tiles on Bolt's routed shape, per token count.

Run from the vllm root. MS=1,8,16,32,64,128 picks token counts, DTYPE=float16
for Turing. speedup > 1 means the GEMV wins at that M.
"""

import os, sys, statistics

sys.path.insert(0, ".")
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import make_moe, make_routing
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import (
    modelopt_block_triton_moe as moe,
)
from vllm.v1.worker.workspace import init_workspace_manager

torch.set_default_device("cuda")
torch.manual_seed(0)
init_workspace_manager(torch.device("cuda"))
# Bolt has 512 experts; E=512 and E=256 OOM on a 16 GB card. Fewer experts put
# more rows on each expert, which favours tl.dot: on the 5060 Ti, E=128 crosses
# over below M=64.
E = int(os.environ.get("E", "512"))
run, w1, w2 = make_moe(
    "iq2_xxs", MoEActivation.RELU2_NO_MUL, E=E, K=1024, inter=2048, top_k=9
)
del w1, w2
base = moe.moe_gemv_config
cfg = {"BN": 8, "BK": 256, "num_warps": 1, "num_stages": 2}
DT = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
MS = [int(m) for m in os.environ.get("MS", "16,32").split(",")]
p = torch.cuda.get_device_properties()
print(p.name, "bus", p.memory_bus_width, "gemv up to M =",
      next((m for m in range(1, 257) if base(m, E, 9, "iq2_xxs") is None), 257) - 1)


def us(fn):
    for _ in range(10):
        fn()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1000


for M in MS:
    x = torch.randn(M, 1024, dtype=DT)
    weights, ids = make_routing(M, E, 9)

    def dot():
        moe.moe_gemv_config = lambda *args: None
        return run(x, weights, ids)

    def gemv():
        moe.moe_gemv_config = lambda *args: cfg
        return run(x, weights, ids)

    a = dot().clone()
    b = gemv().clone()
    gap = ((a - b).abs().max() / a.abs().max()).item()
    assert gap < 1 / 64, gap
    aa = []
    bb = []
    for i in range(3):
        if i % 2:
            bb.append(us(gemv))
            aa.append(us(dot))
        else:
            aa.append(us(dot))
            bb.append(us(gemv))
    a, b = statistics.median(aa), statistics.median(bb)
    print(
        M,
        "dot_us",
        round(a, 2),
        "gemv_us",
        round(b, 2),
        "speedup",
        round(a / b, 3),
        "scaled_output_gap",
        gap,
        flush=True,
    )
moe.moe_gemv_config = base
