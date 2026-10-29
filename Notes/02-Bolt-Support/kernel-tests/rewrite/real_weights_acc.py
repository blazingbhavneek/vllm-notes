#!/usr/bin/env python
"""Accuracy of the Triton block kernels on tensors read from the real checkpoint.

The in-tree tests use synthetic random blocks; a ModelOpt checkpoint's scales and
outlier distribution are not random bytes. Every row here compares the shipped
kernel against the exact GGML decode in fp64 and must stay within 1.5x the error
an unfused bf16 cuBLAS matmul makes on the same weights - the in-tree bar, on
real data. Covers all three dense paths (GEMV, dequant+matmul, tl.dot GEMM) and
both MoE paths on 16 stacked real experts.

Run from the vllm repo root:

  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/real_weights_acc.py [layer]
"""
import glob
import json
import sys

import torch

sys.path.insert(0, ".")
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
from vllm.model_executor.layers.fused_moe.experts import modelopt_block_triton_moe as moe_mod
from vllm.model_executor.layers.fused_moe.experts.modelopt_block_triton_moe import (
    ModelOptBlockTritonExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (  # noqa: E402
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (  # noqa: E402
    moe_gemv_config as GEMV_GATE,
)
from vllm.model_executor.layers.quantization import (  # noqa: E402
    modelopt_block_quant_triton as t,
)
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

CKPT = "/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
CODEC = {66: "iq2_xxs", 34: "q8_0"}
L = sys.argv[1] if len(sys.argv) > 1 else "1"

with open(glob.glob(f"{CKPT}/*.safetensors.index.json")[0]) as f:
    INDEX = json.load(f)["weight_map"]
_handles: dict = {}
FAILS: list[str] = []


def read(name: str, rows: int | None = None) -> torch.Tensor:
    """ggml block tensor [N, NB, block_bytes] uint8, optionally the first `rows`."""
    if name not in INDEX:
        raise KeyError(name)
    path = INDEX[name]
    if path not in _handles:
        from safetensors import safe_open

        _handles[path] = safe_open(f"{CKPT}/{path}", framework="pt")
    part = _handles[path]
    raw = part.get_tensor(name) if rows is None else part.get_slice(name)[0:rows]
    return torch.as_tensor(raw).cuda()


def row(label, ok, detail):
    print(f"{'ok  ' if ok else 'BAD '} {label:38s} {detail}", flush=True)
    if not ok:
        FAILS.append(label)


def dense(label, raw, ms=(1, 4, 8, 64)):
    codec = CODEC[raw.shape[2]]
    N, nb = raw.shape[0], raw.shape[1]
    K = nb * (256 if codec == "iq2_xxs" else 32)
    qweight, scales = t.repack_blocks(raw, codec)
    lut = t.iq2_lut(raw.device) if codec == "iq2_xxs" else None
    wref = ref_decode(raw, codec).double()
    wbf16 = wref.to(torch.bfloat16)
    exact = torch.equal(t.dequantize_blocks(qweight, scales, lut, codec).double(), wbf16.double())
    props = torch.cuda.get_device_properties(raw.device)
    for M in ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        out = torch.ops.vllm.modelopt_block_linear(x, qweight, scales, lut, codec)
        want = x.double() @ wref.T
        bar = 1.5 * ((x @ wbf16.T).double() - want).abs().max()
        ours = (out.double() - want).abs().max()
        cfg = t._linear_config(M, N, K, codec, props.multi_processor_count,
                               props.shared_memory_per_block_optin)
        path = ("dequant" if M >= t.DEQUANT_MIN_M[codec] and N * K <= t.DEQUANT_MAX_NUMEL
                else "gemv" if cfg["BM"] < 16 else "gemm")
        row(f"{label} M={M}", ours <= bar + 1e-6,
            f"err {ours:8.4f} bar {bar:8.4f} ratio {ours / max(bar, 1e-9):4.2f}  "
            f"{path} BN{cfg['BN']} BK{cfg['BK']} SPLIT{cfg['SPLIT']}  "
            f"dequant-exact={exact}")


def moe(layer, E=16, ms=(1, 8, 64)):
    base = f"language_model.backbone.layers.{layer}.mixer.experts"
    up = torch.stack([read(f"{base}.{e}.up_proj.weight") for e in range(E)])
    dn = torch.stack([read(f"{base}.{e}.down_proj.weight") for e in range(E)])
    codec = CODEC[up.shape[3]]
    inter, K = up.shape[1], up.shape[2] * 256
    w1q, s1 = t.repack_blocks(up.contiguous(), codec)
    w2q, s2 = t.repack_blocks(dn.contiguous(), codec)
    lut = t.iq2_lut(w1q.device)
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
                num_experts=E, experts_per_token=9, hidden_dim=K,
                intermediate_size=inter, activation=relu2,
            ),
            quant, codec, lut,
        ),
    )  # fmt: skip
    for M in ms:
        gen = torch.Generator(device="cuda").manual_seed(M)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ids = torch.rand(M, E, device="cuda", generator=gen).topk(9).indices
        ids = ids.to(torch.int32)
        wts = torch.rand(M, 9, device="cuda").softmax(-1)
        args = (x, w1, w2, wts, ids, relu2)
        want = ref_moe(*args, torch.float64)
        # 2x the error of an unfused bf16 MoE plus one bf16 ulp of the output,
        # the bar `assert_moe_close_to_bf16` uses: c1/c2 live in the model dtype,
        # so sub-ulp differences cannot appear in the result.
        bar = 2 * (ref_moe(*args, torch.bfloat16) - want).abs().max()
        bar = bar + want.abs().max().double() * 2**-8
        errs = {}
        for gate in (True, False):
            moe_mod.moe_gemv_config = GEMV_GATE if gate else lambda *a: None
            out = kernel.apply(
                x, w1q, w2q, wts, ids, activation=relu2, global_num_experts=E,
                expert_map=None, apply_router_weight_on_input=False,
            )  # fmt: skip
            errs["gemv" if gate else "dot"] = (out.double() - want).abs().max()
        moe_mod.moe_gemv_config = GEMV_GATE
        g, d = errs["gemv"], errs["dot"]
        row(f"moe E={E} layer {layer} M={M}", max(g, d) <= bar + 1e-9,
            f"gemv {g:8.5f} dot {d:8.5f} bar {bar:8.5f}  "
            f"ratio gemv {g / bar:4.2f} dot {d / bar:4.2f}")


def layer_with(suffix: str) -> str:
    """First layer index that has this tensor (attention lives in some mixers)."""
    return next(
        (str(i) for i in range(64)
         if f"language_model.backbone.layers.{i}.mixer.{suffix}" in INDEX),
        "",
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    init_workspace_manager(torch.device("cuda"))
    P = f"language_model.backbone.layers.{L}.mixer."
    dense(f"layer{L} shared_up   {CODEC[66]}", read(P + "shared_experts.up_proj.weight"))
    dense(f"layer{L} shared_down {CODEC[66]}", read(P + "shared_experts.down_proj.weight"))
    dense(f"layer{L} expert0_up  {CODEC[66]}", read(P + f"experts.0.up_proj.weight"), (1, 8))
    dense(f"layer{L} expert0_dn  {CODEC[66]}", read(P + f"experts.0.down_proj.weight"), (1, 8))
    dense(f"layer{L} fc1_latent  {CODEC[34]}", read(P + "fc1_latent_proj.weight"), (1, 8))
    dense(f"layer{L} fc2_latent  {CODEC[34]}", read(P + "fc2_latent_proj.weight"), (1, 8))
    A = layer_with("o_proj.weight")
    if A:
        dense(f"layer{A} o_proj     {CODEC[34]}",
              read(f"language_model.backbone.layers.{A}.mixer.o_proj.weight"), (1, 8))
        dense(f"layer{A} q_proj     {CODEC[34]}",
              read(f"language_model.backbone.layers.{A}.mixer.q_proj.weight"), (1, 8))
    dense("lm_head           " + CODEC[34], read("language_model.lm_head.weight", 4096), (1, 8))
    moe(L)
    print("\nFAILURES:", FAILS if FAILS else "none")
    raise SystemExit(1 if FAILS else 0)
