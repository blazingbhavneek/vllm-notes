"""Error of the fused MoE relative to an unfused bf16 MoE, over 20 seeds.

The kernel test allows 2x; run from the vllm repo root to re-check the margin.
"""
import itertools, torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.worker.workspace import init_workspace_manager
import tests.kernels.quantization.test_modelopt_block_triton as T
from vllm.model_executor.layers.fused_moe.activation import MoEActivation as A

init_workspace_manager(torch.device("cuda"))
with set_current_vllm_config(VllmConfig()):
    for codec, act, top_k, M in itertools.product(("iq2_xxs", "q8_0"), (A.RELU2_NO_MUL, A.SILU), (2, 9), (1, 7, 64, 333)):
        run, w1, w2 = T.make_moe(codec, act, top_k=top_k)
        ratios = []
        for seed in range(20):
            torch.manual_seed(seed)
            x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
            tw, ti = T.make_routing(M, 16, top_k)
            want = T.ref_moe(x, w1, w2, tw, ti, act, torch.float64)
            bf = T.ref_moe(x, w1, w2, tw, ti, act, torch.bfloat16)
            got = run(x, tw, ti).double()
            ratios.append(((got - want).abs().max() / (bf - want).abs().max()).item())
        r = torch.tensor(ratios)
        print(f"{codec} {act.name} k={top_k} M={M}: median {r.median():.2f} max {r.max():.2f}")
