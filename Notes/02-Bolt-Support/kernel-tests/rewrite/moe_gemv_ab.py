#!/usr/bin/env python
"""A/B the MoE decode GEMV path against the padded tl.dot path, per shape/M.

The dot side uses the in-tree `moe_config` tiles; the GEMV side sweeps
`moe_gemv_config` tiles.

  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/moe_gemv_ab.py [E xI ...]
"""
import itertools
import statistics
import sys
from functools import partial

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm.model_executor.layers.fused_moe.experts.modelopt_block_triton_moe as moe_mod
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t
from vllm.v1.worker.workspace import init_workspace_manager

TOP_K = 9
# BK and the pipeline depth matter as much as the tile: w1 has K=1024 and w2
# K=2048, so at BK=256 the K-loop is only 4-8 steps. Measured on the real
# (E=512, inter=2048) shape, BK=256/2 stages still wins; BK=512 is 1% better at
# M=1 and 4% worse at M=8, BK=1024 is 17% worse, BN=16 is 16% worse, and
# num_stages changes nothing. Keep the in-tree pick.
GEMV = [
    {"BN": bn, "BK": bk, "num_warps": w, "num_stages": st}
    for bn, bk, w, st in itertools.product((2, 4, 8, 16), (256, 512, 1024), (1,), (1, 2, 3))
]
MS = [1, 4, 8, 16, 32, 64, 128, 256, 512]
IN_TREE_GEMV = moe_mod.moe_gemv_config
IN_TREE_DOT = moe_mod.moe_config


def us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


def main() -> None:
    # (512, 2048) is the real Bolt 74B routed-expert shape; the others
    # are the sizes that were swept during tuning.
    shapes = [(512, 2048), (240, 2048), (432, 2560), (496, 2560)]
    if len(sys.argv) > 1:
        shapes = [(int(a.split("x")[0]), int(a.split("x")[1])) for a in sys.argv[1:]]
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    init_workspace_manager(torch.device("cuda"))
    relu2 = MoEActivation.RELU2_NO_MUL
    for E, inter in shapes:
        w1q, s1 = (
            x.contiguous()
            for x in t.repack_blocks(
                torch.randint(0, 256, (E, inter, 4, 66), dtype=torch.uint8), "iq2_xxs"
            )
        )
        w2q, s2 = (
            x.contiguous()
            for x in t.repack_blocks(
                torch.randint(0, 256, (E, 1024, inter // 256, 66), dtype=torch.uint8),
                "iq2_xxs",
            )
        )
        lut = t.iq2_lut(torch.device("cuda"))
        cfg = FusedMoEConfig(
            num_experts=E, experts_per_token=TOP_K, hidden_dim=1024,
            intermediate_size=inter, num_local_experts=E, num_logical_experts=E,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            activation=relu2, in_dtype=torch.bfloat16, device="cuda",
            routing_method=RoutingMethodType.TopK, max_num_tokens=max(MS),
        )  # fmt: skip
        quant = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(), _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype="iq2_xxs", scale=s1),
            _w2=FusedMoEQuantDesc(dtype="iq2_xxs", scale=s2),
        )
        kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            moe_mod.ModelOptBlockTritonExperts(cfg, quant, "iq2_xxs", lut),
        )
        gen = torch.Generator(device="cuda").manual_seed(0)
        print(f"\n=== MoE E={E} inter={inter} top_k={TOP_K} (IQ2_XXS) ===")
        print("   M   slots  gemv_us  ( GB/s) gemv_cfg                              "
              "  dot_us  ( GB/s) in-tree ratio")
        for M in MS:
            x = torch.randn(M, 1024, dtype=torch.bfloat16)
            ids = torch.rand(M, E, generator=gen, device="cuda").topk(TOP_K).indices
            ids = ids.to(torch.int32)
            wts = torch.rand(M, TOP_K, device="cuda").softmax(-1)
            run = partial(
                kernel.apply, x, w1q, w2q, wts, ids, activation=relu2,
                global_num_experts=E, expert_map=None,
                apply_router_weight_on_input=False,
            )  # fmt: skip
            best = {}
            for name, cands in (("gemv", GEMV), ("dot", [None])):
                best[name] = (float("inf"), None)
                for c in cands:
                    moe_mod.moe_gemv_config = (
                        lambda *_, c=c: c if name == "gemv" else None
                    )
                    moe_mod.moe_config = IN_TREE_DOT
                    try:
                        v = us(run)
                    except Exception as e:
                        print(f"  [{name}] M={M} {c} -> {type(e).__name__}", flush=True)
                        continue
                    if v < best[name][0]:
                        best[name] = (v, c)
                    print(f"  [{name}] M={M} {c} -> {v:.1f} us", flush=True)
            moe_mod.moe_gemv_config = IN_TREE_GEMV
            moe_mod.moe_config = IN_TREE_DOT
            g, d = best["gemv"], best["dot"]
            auto = IN_TREE_GEMV(M, E, TOP_K, "iq2_xxs") is not None
            active = ids.unique().numel()
            mb = active * 2 * inter * 1024 * 66 / 256 / 1e6
            print(f"{M:4d} {M * TOP_K:5d} {g[0]:8.1f} ({mb / g[0] * 1e3:4.0f}) "
                  f"{str(g[1]):44s} {d[0]:8.1f} ({mb / d[0] * 1e3:4.0f}) "
                  f"{'GEMV' if auto else 'dot '} {d[0] / g[0]:5.2f}x", flush=True)


with set_current_vllm_config(VllmConfig()):
    main()
