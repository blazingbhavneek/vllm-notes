"""After _warm_moe, a ReLU2 decode step through the real experts must not JIT.

Run from the vllm checkout: python ../Notes/02-Bolt-Support/kernel-tests/check_warmup_relu2.py
Pass `float16` as the first argument to check the FP16 (Turing) path.
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
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization import (  # noqa: E402
    modelopt_block_quant_triton as kernels,
)
from vllm.model_executor.warmup.modelopt_block_triton_warmup import (  # noqa: E402
    _warm_moe,
)
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402


def compiled(fn):
    return {k for c in fn.device_caches.values() for k in c[0]}


init_workspace_manager(torch.device(0))
dtype = getattr(torch, sys.argv[1] if len(sys.argv) > 1 else "bfloat16")
codec, E, K, inter, top_k = "iq2_xxs", 16, 1024, 512, 2
w1, s1 = kernels.repack_blocks(make_raw((E, inter), K, codec, seed=7), codec)
w2, s2 = kernels.repack_blocks(make_raw((E, K), inter, codec, seed=8), codec)
module = types.SimpleNamespace(
    w13_weight=w1, w13_weight_scale=s1, w2_weight=w2, w2_weight_scale=s2,
    top_k=top_k, apply_router_weight_on_input=False, expert_map=None,
    activation=MoEActivation.RELU2_NO_MUL,
)  # fmt: skip
method = types.SimpleNamespace(codec=codec, block_size=256)
_warm_moe(module, method, kernels, moe_align_block_size, 64, set(), dtype)
torch.accelerator.synchronize()
before = compiled(kernels._moe_gemv_kernel)

run, _, _ = make_moe(codec, MoEActivation.RELU2_NO_MUL, E=E, K=K, inter=inter, top_k=top_k)
for M in (1, 8, 64):  # GEMV covers 64 rows on GDDR GPUs
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    run(x, *make_routing(M, E, top_k))
torch.accelerator.synchronize()
new = compiled(kernels._moe_gemv_kernel) - before
print(f"{dtype}: warmed {len(before)}, compiled at inference: {len(new)}")
assert not new, new
print("OK: no JIT during inference for the ReLU2 GEMV path")
