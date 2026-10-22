"""Correctness of decode_blocks / decode_expert_gather / block_linear vs gguf-py."""

import itertools

import torch
import torch.nn.functional as F

import os

from harness import WEIGHTS, hopper, make_packed, oracle_decode

if os.environ.get("FORCE_STAGES"):
    _orig = hopper._launch_gemm

    def _patched(*a, **kw):
        kw["stages"] = int(os.environ["FORCE_STAGES"])
        return _orig(*a, **kw)

    hopper._launch_gemm = _patched

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


# 1. decode is bit-exact in fp32 and round-to-nearest in bf16
for codec in ("iq2_xxs", "q8_0"):
    packed = make_packed((37, 21), codec, specials=True)
    want = oracle_decode(packed, codec)
    got32 = hopper.decode_blocks(packed, codec, torch.float32)
    got16 = hopper.decode_blocks(packed, codec)
    fin = torch.isfinite(want)
    check(f"decode fp32 bit-exact {codec}",
          torch.equal(got32[fin], want[fin]) and torch.equal(fin, torch.isfinite(got32)),
          f"max|d|={(got32[fin]-want[fin]).abs().max().item():.3g}")
    check(f"decode bf16 == round(oracle) {codec}",
          torch.equal(got16[fin], want[fin].bfloat16()))

# 2. expert gather equals plain decode of the selected experts (exactly)
for codec in ("iq2_xxs", "q8_0"):
    packed = make_packed((6, 40, 4), codec, seed=1)
    ids = torch.tensor([5, 0, 3, 3], device="cuda", dtype=torch.int32)
    got = hopper.decode_expert_gather(packed, ids, codec)
    want = oracle_decode(packed, codec).bfloat16()[ids.long()]
    check(f"expert gather exact {codec}", torch.equal(got, want))

# 3. block_linear vs fp64 reference on bf16-rounded oracle weights.
# Metric: kernel error must be within 2x of cuBLAS on the same bf16 operands.
SHAPES = {
    "q8_0": [(1024, 128), (4096, 32), (257, 3), (4096, 128), (6144, 128)],
    "iq2_xxs": [(5376, 16), (4096, 21), (257, 3), (1024, 4), (4096, 16)],
}
MS = [1, 2, 7, 16, 17, 64, 100, 256, 1000]
for codec, (n, nb) in itertools.chain.from_iterable(
    ((c, s) for s in SHAPES[c]) for c in SHAPES
):
    packed = make_packed((n, nb), codec, seed=n + nb)
    w = oracle_decode(packed, codec).bfloat16()
    k = nb * WEIGHTS[codec]
    worst = 0.0
    nfail = len(fails)
    for m in MS:
        x = torch.randn(m, k, device="cuda").bfloat16()
        ref = (x.double() @ w.double().T)
        try:
            got = hopper.block_linear(x, packed, codec).double()
        except Exception as e:  # noqa: BLE001
            check(f"block_linear {codec} N={n} NB={nb} M={m}", False,
                  type(e).__name__ + ": " + str(e)[:90])
            continue
        cub = F.linear(x, w).double()
        scale = ref.abs().max().item()
        e_k = (got - ref).abs().max().item() / scale
        e_c = (cub - ref).abs().max().item() / scale
        ratio = e_k / max(e_c, 1e-12)
        worst = max(worst, ratio)
        if not (torch.isfinite(got).all() and e_k <= max(2 * e_c, 4e-3)):
            check(f"block_linear {codec} N={n} NB={nb} M={m}", False,
                  f"rel={e_k:.3g} cublas={e_c:.3g}")
    check(f"block_linear {codec} N={n} NB={nb} all M", len(fails) == nfail,
          f"worst err/cublas_err={worst:.2f}")

# 4. 3-D input and non-contiguous input
packed = make_packed((512, 4), "iq2_xxs", seed=9)
w = oracle_decode(packed, "iq2_xxs").bfloat16()
x = torch.randn(3, 5, 1024, device="cuda").bfloat16()
got = hopper.block_linear(x, packed, "iq2_xxs")
check("block_linear 3-D input shape", got.shape == (3, 5, 512))
check("block_linear 3-D values",
      torch.allclose(got.float(), F.linear(x, w).float(), rtol=2e-2, atol=2e-2))
xt = torch.randn(1024, 16, device="cuda").bfloat16().T  # [16, 1024] non-contig
p2 = make_packed((64, 4), "iq2_xxs", seed=3)
got = hopper.block_linear(xt, p2, "iq2_xxs")
want = F.linear(xt, oracle_decode(p2, "iq2_xxs").bfloat16())
check("block_linear non-contiguous x",
      torch.allclose(got.float(), want.float(), rtol=2e-2, atol=2e-2))

print("\nFAILURES:", fails if fails else "none")
