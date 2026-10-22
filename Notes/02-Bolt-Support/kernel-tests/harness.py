"""Load the agent's Triton module standalone (no vLLM install) with gguf-py as oracle.

b12x is faked only to hand the kernel the IQ2_XXS grid; the grid itself comes
from gguf-py (llama.cpp's own Python port), so the oracle is independent.
"""

import importlib.util
import os
import sys
import types

import numpy as np
import torch
from gguf import GGMLQuantizationType as Q
from gguf import quants

VLLM = os.environ.get("VLLM_SRC", "/mnt/common/Code/vllmServer/vllm")
MOD = f"{VLLM}/vllm/model_executor/layers/quantization/modelopt_block_quant_hopper.py"

quants.IQ2_XXS.init_grid()
GRID = torch.from_numpy(quants.IQ2_XXS.grid.reshape(256, 8).astype(np.float32).copy())

_fake = types.ModuleType("b12x._lib.quant.iq2_xs")
_fake.iq2_xs_grid_cpu = lambda codec: GRID.clone()
for name in ("b12x", "b12x._lib", "b12x._lib.quant"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["b12x._lib.quant.iq2_xs"] = _fake

_spec = importlib.util.spec_from_file_location("hopper", MOD)
hopper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hopper)

BYTES = {"iq2_xxs": 66, "q8_0": 34}
WEIGHTS = {"iq2_xxs": 256, "q8_0": 32}
QTYPE = {"iq2_xxs": Q.IQ2_XXS, "q8_0": Q.Q8_0}


def make_packed(shape, codec, scale=2e-3, seed=0, specials=False):
    """Random packed blocks [..., NB, bytes] with finite fp16 block scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(0, 256, (*shape, BYTES[codec]), dtype=torch.uint8,
                           device="cuda", generator=g)
    d = (torch.randn(shape, device="cuda", generator=g) * scale).half()
    if specials:  # zeros, negatives, subnormals, large
        flat = d.view(-1)
        flat[0::7] = 0.0
        flat[1::11] = -flat[1::11].abs()
        flat[2::13] = 3e-6  # fp16 subnormal
        flat[3::17] = 60000.0
    packed[..., :2] = d.unsqueeze(-1).view(torch.uint8)
    return packed


def oracle_decode(packed, codec):
    """gguf-py reference: [..., NB, bytes] -> float32 [..., NB*weights]."""
    lead = packed.shape[:-2]
    arr = packed.reshape(-1, BYTES[codec]).cpu().numpy()
    out = quants.dequantize(arr, QTYPE[codec]).astype(np.float32)
    return torch.from_numpy(out).reshape(*lead, -1).cuda()
