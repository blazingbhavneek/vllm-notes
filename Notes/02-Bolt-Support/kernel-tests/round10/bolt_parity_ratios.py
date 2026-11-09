"""Error ratio ours / cuBLAS (both vs fp64 exact) on Bolt shapes and the shared-expert stack."""
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from tests.kernels.quantization.test_modelopt_block_triton import (  # noqa: E402
    BOLT_LINEAR, linear, make_raw, ref_decode,
)
torch.manual_seed(0)
print(torch.cuda.get_device_name())
for layer, (codec, N, K) in BOLT_LINEAR.items():
    raw = make_raw((N,), K, codec, seed=1)
    w = ref_decode(raw, codec)
    row = []
    for M in (1, 8, 64, 512):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        want = x.double() @ w.double().T
        ours = (linear(x, raw, codec).double() - want).abs().max()
        cub = ((x @ w.bfloat16().T).double() - want).abs().max()
        row.append(f"M={M}: {ours / cub:.2f}")
    print(f"{layer:16s}", "  ".join(row))
raw_up = make_raw((5376,), 4096, "iq2_xxs", seed=11)
raw_down = make_raw((4096,), 5376, "iq2_xxs", seed=12)
up, down = ref_decode(raw_up, "iq2_xxs"), ref_decode(raw_down, "iq2_xxs")
row = []
for M in (1, 64, 512):
    x = torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16)
    ours = linear(F.relu(linear(x, raw_up, "iq2_xxs")).square(), raw_down, "iq2_xxs")
    cub = F.relu(x @ up.bfloat16().T).square() @ down.bfloat16().T
    exact = F.relu(x.double() @ up.double().T).square() @ down.double().T
    e_o = (ours.double() - exact).abs().max() / exact.abs().max()
    e_c = (cub.double() - exact).abs().max() / exact.abs().max()
    row.append(f"M={M}: {e_o / e_c:.2f} (rel {e_o:.1e} vs {e_c:.1e})")
print(f"{'stack up->down':16s}", "  ".join(row))
