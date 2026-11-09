"""After _warm_moe, ReLU2 MoE steps through the real experts must not JIT.

Run from the vllm checkout: python ../Notes/02-Bolt-Support/kernel-tests/check_warmup_relu2.py
Pass `float16` as the first argument to check the FP16 (Turing) path.
Covers the decode GEMV (`_moe_gemv_kernel`) and the grouped GEMM (upstream
`fused_moe_kernel` with `BLOCK_CODEC`).
"""

import sys
import types

import torch

sys.path.insert(0, ".")
from tests.kernels.quantization.test_modelopt_block_triton import (  # noqa: E402
    make_moe,
    make_raw,
    make_routing,
)
from vllm.model_executor.layers.fused_moe import fused_moe  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.quantization import (  # noqa: E402
    modelopt_block_quant_triton as kernels,
)
from vllm.model_executor.warmup.modelopt_block_triton_warmup import (  # noqa: E402
    _warm_moe,
)
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

JITTED = (kernels._moe_gemv_kernel, fused_moe.fused_moe_kernel)


def compiled():
    return {
        (fn.__name__, k) for fn in JITTED for c in fn.device_caches.values() for k in c[0]
    }


init_workspace_manager(torch.device(0))
dtype = getattr(torch, sys.argv[1] if len(sys.argv) > 1 else "bfloat16")
codec, E, K, inter, top_k, max_tokens = "iq2_xxs", 16, 1024, 512, 2, 512
act = MoEActivation.RELU2_NO_MUL
run, _, _ = make_moe(codec, act, E=E, K=K, inter=inter, top_k=top_k)
w1, s1 = kernels.repack_blocks(make_raw((E, inter), K, codec, seed=7), codec)
w2, s2 = kernels.repack_blocks(make_raw((E, K), inter, codec, seed=8), codec)
module = types.SimpleNamespace(
    w13_weight=w1, w2_weight=w2, top_k=top_k, activation=act,
    apply_router_weight_on_input=False,
)  # fmt: skip
# `_warm_moe` calls `method.apply`; route it into the same experts kernel.
method = types.SimpleNamespace(
    codec=codec, apply=lambda layer, x, weights, ids, *_: run(x, weights, ids)
)
_warm_moe(module, method, kernels, max_tokens, set(), dtype)
torch.accelerator.synchronize()
before = compiled()

for M in (1, 3, 8, 9, 16, 31, 64, 200, 512):
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    run(x, *make_routing(M, E, top_k))
torch.accelerator.synchronize()
new = compiled() - before
print(f"{dtype}: warmed {len(before)}, compiled at inference: {sorted(n for n, _ in new)}")
assert not new, new
print("OK: no JIT during inference for the ReLU2 MoE paths")
