"""Check the remaining M=16/32 GEMV crossover on a full routed-MoE call."""

import sys, statistics

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
run, w1, w2 = make_moe(
    "iq2_xxs", MoEActivation.RELU2_NO_MUL, E=512, K=1024, inter=2048, top_k=9
)
del w1, w2
base = moe.moe_gemv_config
cfg = {"BN": 8, "BK": 256, "num_warps": 1, "num_stages": 2}


def us(fn):
    for _ in range(10):
        fn()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1000


for M in (16, 32):
    x = torch.randn(M, 1024, dtype=torch.bfloat16)
    weights, ids = make_routing(M, 512, 9)

    def dot():
        moe.moe_gemv_config = base
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
