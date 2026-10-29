#!/usr/bin/env python
"""Where does the fused MoE's decode-time error come from? (real expert weights)

Compares, on identical routing:
  fp64        exact weights, exact math, no intermediate rounding
  ref_bf16    unfused bf16 MoE - this is the in-tree reference's error bar
  ref_fused   OUR pipeline simulated in torch: exact weights, but the GEMM1
              output and the per-slot GEMM2 output stored in bf16 like c1/c2
  gemv / dot  the Triton paths
If kernel ~= ref_fused, the error is the bf16 intermediate the fused-MoE
contract requires, not the quantized decode or the kernel's accumulation.

Run from the vllm repo root:
  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/moe_err_decomp.py [layer]
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_weights_acc as A  # noqa: E402
from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from tests.kernels.quantization.test_modelopt_block_triton import (  # noqa: E402
    ref_decode,
    ref_moe,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.experts import (  # noqa: E402
    modelopt_block_triton_moe as moe_mod,
)
from vllm.model_executor.layers.fused_moe.experts.modelopt_block_triton_moe import (
    ModelOptBlockTritonExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (  # noqa: E402
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (  # noqa: E402
    moe_gemv_config as GEMV_GATE,
)
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

L = sys.argv[1] if len(sys.argv) > 1 else "1"
E, TOP_K = 16, 9


def ref_fused(x, w1, w2, wts, ids):
    """Our pipeline in torch: exact weights, bf16 c1/c2 like the real buffers."""
    M = x.shape[0]
    out = torch.zeros(M, w2.shape[1], dtype=torch.float32, device=x.device)
    for t in range(M):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            h = (x[t].double() @ w1[e].double().T).to(torch.bfloat16)  # c1
            h = F.relu(h).square()  # activation reads/writes bf16
            y = (h.double() @ w2[e].double().T * wts[t, k].double()).to(torch.bfloat16)
            out[t] += y.double()  # finalize sums the slots
    return out.to(torch.bfloat16)


if __name__ == "__main__":
    torch.manual_seed(0)
    init_workspace_manager(torch.device("cuda"))
    base = f"language_model.backbone.layers.{L}.mixer.experts"
    up = torch.stack([A.read(f"{base}.{e}.up_proj.weight") for e in range(E)])
    dn = torch.stack([A.read(f"{base}.{e}.down_proj.weight") for e in range(E)])
    codec = A.CODEC[up.shape[3]]
    inter, K = up.shape[1], up.shape[2] * 256
    w1q, s1 = A.t.repack_blocks(up.contiguous(), codec)
    w2q, s2 = A.t.repack_blocks(dn.contiguous(), codec)
    lut = A.t.iq2_lut(w1q.device)
    w1, w2 = ref_decode(up, codec), ref_decode(dn, codec)
    relu2 = MoEActivation.RELU2_NO_MUL
    quant = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(), _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(dtype=codec, scale=s1),
        _w2=FusedMoEQuantDesc(dtype=codec, scale=s2),
    )  # fmt: skip
    kernel = mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        ModelOptBlockTritonExperts(
            make_dummy_moe_config(
                num_experts=E, experts_per_token=TOP_K, hidden_dim=K,
                intermediate_size=inter, activation=relu2,
            ),
            quant, codec, lut,
        ),
    )  # fmt: skip
    print(f"layer {L}: E={E} K={K} inter={inter} {codec}; err = max |x - fp64|")
    for M in (1, 2, 4, 8):
        gen = torch.Generator(device="cuda").manual_seed(M)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(M, E, device="cuda", generator=gen).topk(TOP_K).indices.to(torch.int32)
        wts = torch.rand(M, TOP_K, device="cuda").softmax(-1)
        args = (x, w1, w2, wts, ids, relu2)
        want = ref_moe(*args, torch.float64)
        err = lambda got: float((got.double() - want).abs().max())
        ulp = float(want.abs().max().to(torch.bfloat16)) * 2**-8
        errs = {"fp64": 0.0, "ref_bf16": err(ref_moe(*args, torch.bfloat16)),
                "ref_fused": err(ref_fused(x, w1, w2, wts, ids))}
        for name, gate in (("gemv", GEMV_GATE), ("dot", lambda *a: None)):
            moe_mod.moe_gemv_config = gate
            errs[name] = err(
                kernel.apply(
                    x, w1q, w2q, wts, ids, activation=relu2, global_num_experts=E,
                    expert_map=None, apply_router_weight_on_input=False,
                )
            )  # fmt: skip
        moe_mod.moe_gemv_config = GEMV_GATE
        print(f"  M={M}: " + "  ".join(f"{k}={v:.5f}" for k, v in errs.items())
              + f"   |out|={float(want.abs().max()):.3f} 1ulp={ulp:.5f}")
