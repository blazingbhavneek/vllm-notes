"""Does `do_not_specialize=["EM", "num_valid_tokens"]` on `fused_moe_kernel` cost
upstream's plain BF16 MoE anything? PR #65003 adds it so a batch size change does
not trigger a new compile.

Run from the vLLM repo root:

    python <notes>/Notes/02-Bolt-Support/kernel-tests/round10/moe_specialize_ab.py

Arms: "pr" is the module's kernel; "spec" is the same function re-jitted without
`do_not_specialize` (upstream's decorator), swapped into the module global that
`invoke_fused_moe_triton_kernel` launches. GEMM1 only, tuned config if the repo
has one for the shape, else the default. Same routing and output for both.
Timing: CUDA-graph replay, `do_bench` (L2 flushed before each), median.
"""

import torch
import triton
import triton.language as tl

import vllm.model_executor.layers.fused_moe.fused_moe as fm
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)

# (name, E, 2 * intermediate, hidden, top_k)
MODELS = [
    ("mixtral-8x7b", 8, 2 * 14336, 4096, 2),
    ("qwen3-30b-a3b", 128, 2 * 768, 2048, 8),
    ("deepseek-v2-lite", 64, 2 * 1408, 2048, 6),
]
TOKENS = [1, 7, 8, 16, 17, 32, 64, 128, 256, 1024, 4096]


def bench_us(fn):
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return triton.testing.do_bench(graph.replay, return_mode="median") * 1e3


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    kernels = {"pr": fm.fused_moe_kernel, "spec": triton.jit(fm.fused_moe_kernel.fn)}
    print(torch.cuda.get_device_name())
    print(f"{'model':18s} {'M':>5s} {'pr_us':>9s} {'spec_us':>9s} {'pr/spec':>8s}")
    worst = 0.0
    for name, E, N, K, top_k in MODELS:
        w = torch.randn(E, N, K, dtype=torch.bfloat16) / K**0.5
        for M in TOKENS:
            a = torch.randn(M, K, dtype=torch.bfloat16)
            ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(M)]).int()
            c = torch.empty(M, top_k, N, dtype=torch.bfloat16)
            cfg = try_get_optimal_moe_config(w.shape, (E, K, N // 2), top_k, None, M)
            routing = _prepare_expert_assignment(ids, cfg, M, top_k, E, None)

            def run():
                invoke_fused_moe_triton_kernel(
                    a, w, c, None, None, None, *routing, False, top_k, cfg,
                    tl.bfloat16, False, False, False, False, False,
                )  # fmt: skip

            us, outs = {}, {}
            for arm, kernel in kernels.items():
                fm.fused_moe_kernel = kernel
                try:
                    run()
                    outs[arm] = c.clone()
                    us[arm] = bench_us(run)
                finally:
                    fm.fused_moe_kernel = kernels["pr"]
            assert torch.equal(outs["pr"], outs["spec"]), (name, M)
            ratio = us["pr"] / us["spec"]
            worst = max(worst, ratio)
            print(f"{name:18s} {M:5d} {us['pr']:9.2f} {us['spec']:9.2f} {ratio:8.3f}")
        del w
        torch.cuda.empty_cache()
    print(f"worst pr/spec: {worst:.3f}")


if __name__ == "__main__":
    main()
