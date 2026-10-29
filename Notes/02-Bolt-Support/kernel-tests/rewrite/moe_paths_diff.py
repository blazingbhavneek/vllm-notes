#!/usr/bin/env python
"""How far apart are the two MoE paths on identical routing, and which is right?

Prints output scale, the path-to-path gap, and each path's error against the
fp64 reference next to an unfused bf16 MoE's error (the bar the in-tree test
uses). Run from the vllm repo root:

  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/moe_paths_diff.py
"""
import sys

import torch

sys.path.insert(0, ".")
from tests.kernels.quantization.test_modelopt_block_triton import (  # noqa: E402
    make_moe,
    make_routing,
    ref_moe,
)
from vllm.model_executor.layers.fused_moe.experts import (  # noqa: E402
    modelopt_block_triton_moe,
)
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (  # noqa: E402
    moe_gemv_config as GATE,
)

from vllm.v1.worker.workspace import init_workspace_manager

torch.manual_seed(0)
init_workspace_manager(torch.device("cuda"))

for codec in ("iq2_xxs", "q8_0"):
    for M in (1, 8):
        run, w1, w2 = make_moe(codec, MoEActivation.RELU2_NO_MUL, top_k=9)
        x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
        tw, ti = make_routing(M, 16, 9)
        gemv = run(x, tw, ti).clone()
        modelopt_block_triton_moe.moe_gemv_config = lambda *a: None
        dot = run(x, tw, ti).clone()
        modelopt_block_triton_moe.moe_gemv_config = GATE
        want = ref_moe(x, w1, w2, tw, ti, MoEActivation.RELU2_NO_MUL, torch.float64)
        bf16 = ref_moe(x, w1, w2, tw, ti, MoEActivation.RELU2_NO_MUL, torch.bfloat16)
        bar = 2 * (bf16 - want).abs().max().item()
        rel = lambda a: ((a.double() - want).abs() / (want.abs() + 1.0)).max().item()
        print(
            f"{codec} M={M}: |out| max {want.abs().max():8.1f}  "
            f"paths differ by {float((gemv - dot).abs().max()):7.3f} "
            f"({(gemv - dot).abs().max() / want.abs().max() * 100:5.2f}% of scale)\n"
            f"  gemv err {rel(gemv):.5f}  dot err {rel(dot):.5f}  "
            f"bf16-cuBLAS bar {bar / want.abs().max():.5f}  "
            f"2x-bar abs {bar:8.3f}  gap {float((gemv - dot).abs().max()):.3f}"
        )
