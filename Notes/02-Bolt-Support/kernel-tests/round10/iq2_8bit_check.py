"""Would an 8-bit tensor-core path help IQ2_XXS? Checks exactness and the ceiling."""

import sys

import torch
from triton.testing import do_bench_cudagraph

sys.path.insert(0, ".")
from tests.kernels.quantization.test_modelopt_block_triton import (  # noqa: E402
    make_raw,
    ref_decode,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (  # noqa: E402
    dequantize_blocks,
    iq2_lut,
    repack_blocks,
)

# 1. Exactness. IQ2_XXS weight = d * (0.5 + s) * 0.25 * grid * sign, with one
# (d, s) per 32 weights. Divide out that per-32 scale; the rest should be integers.
raw = make_raw((256,), 4096, "iq2_xxs", seed=0)
w = ref_decode(raw, "iq2_xxs").double()  # [256, 4096]
blocks = raw.reshape(-1, 66)
d = blocks[:, :2].contiguous().view(torch.float16).double()  # per 256
aux = blocks[:, 2:].contiguous().view(torch.int32)[:, 1::2]  # 8 per block
s = ((aux >> 28) & 15).double()
rs = (d * (0.5 + s) * 0.25).reshape(256, -1)  # [256, 128] = per 32 weights
lv = w / rs.repeat_interleave(32, dim=1)
ok = rs.repeat_interleave(32, dim=1) != 0
levels = torch.unique(lv[ok].round())
print("integer levels after removing the per-32 scale:", levels.tolist())
print("max |level - round(level)|:", (lv[ok] - lv[ok].round()).abs().max().item())
mags = torch.tensor([8.0, 25.0, 43.0])
print("e4m3 round trip of 8, 25, 43:",
      mags.to(torch.float8_e4m3fn).float().tolist(),
      "| int8:", mags.to(torch.int8).float().tolist())

# 2. Ceiling at prefill: raw BF16 vs INT8 vs FP8 matmul on this GPU, against the
# PR's IQ2 dense path (dequantize to BF16 + cuBLAS) on Bolt's shared_up shape.
print(f"\nGPU {torch.cuda.get_device_name()}  (us, CUDA graph)")
N, K = 5376, 4096
qw, sc = repack_blocks(make_raw((N,), K, "iq2_xxs", seed=1), "iq2_xxs")
lut = iq2_lut(qw.device)
wb = dequantize_blocks(qw, sc, lut, "iq2_xxs", torch.bfloat16)
wi8 = torch.randint(-43, 44, (N, K), dtype=torch.int8, device="cuda")
w8 = wb.to(torch.float8_e4m3fn)
one = torch.ones((), device="cuda")
print(f"{'M':>5s} {'PR iq2':>8s} {'dequant':>8s} {'bf16 mm':>8s} "
      f"{'int8 mm':>8s} {'fp8 mm':>8s} {'bf16/int8':>9s} {'bf16/fp8':>8s}")
for M in (256, 1024, 4096):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    xi8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
    x8 = x.to(torch.float8_e4m3fn)
    t = {
        "pr": lambda: torch.ops.vllm.modelopt_block_linear(x, qw, sc, lut, "iq2_xxs"),
        "deq": lambda: dequantize_blocks(qw, sc, lut, "iq2_xxs", torch.bfloat16),
        "bf16": lambda: x @ wb.T,
        "int8": lambda: torch._int_mm(xi8, wi8.T),
        "fp8": lambda: torch._scaled_mm(x8, w8.T, one, one, out_dtype=torch.bfloat16),
    }
    r = {k: do_bench_cudagraph(f, rep=300) * 1e3 for k, f in t.items()}
    print(f"{M:5d} {r['pr']:8.1f} {r['deq']:8.1f} {r['bf16']:8.1f} {r['int8']:8.1f} "
          f"{r['fp8']:8.1f} {r['bf16']/r['int8']:8.2f}x {r['bf16']/r['fp8']:7.2f}x")
