"""Perf of the draft's block_linear / moe_forward (baseline for the rewrite).

Loads the draft module from $VLLM_SRC (a worktree of commit 2064cc2efb). On GPUs with
99 KB shared memory (sm_86/89/120) set FORCE_STAGES=1; on H200 leave it unset.
"""

import os

STAGES = int(os.environ.get("FORCE_STAGES", "2"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import test_moe  # noqa: E402,F401  (installs fake vllm ops + stage patch for MoE)
from harness import BYTES, WEIGHTS, hopper, make_packed, oracle_decode  # noqa: E402

if os.environ.get("FORCE_STAGES"):
    _orig = hopper._launch_gemm

    def _patched(*a, **kw):
        kw["stages"] = STAGES
        return _orig(*a, **kw)

    hopper._launch_gemm = _patched


def frozen_bm128(x, packed, codec):
    """What torch.compile freezes in: trace-time M>=65 -> BM=128 plain, no GEMV/split."""
    n, nb = packed.shape[:2]
    m, k = x.shape
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    flat = packed.reshape(n, -1)
    grid = (triton.cdiv(n, 64), triton.cdiv(m, 128))
    if codec == "iq2_xxs":
        hopper._iq2_gemm_kernel[grid](x, flat, y, hopper._iq2_grid("iq2_xxs", x.device),
                                      m, n, k, nb, flat.shape[-1], BN=64, BM=128,
                                      num_warps=8, num_stages=STAGES)
    else:
        hopper._q8_gemm_kernel[grid](x, flat, y, m, n, k, nb, flat.shape[-1],
                                     BN=64, BM=128, num_warps=8, num_stages=STAGES)
    return y


props = torch.cuda.get_device_properties(0)
print(f"GPU {props.name} SMs={props.multi_processor_count}")
SHAPES = [("q8_0", 131072, 128, "lm_head"), ("q8_0", 4096, 128, "attn-ish"),
          ("q8_0", 1024, 128, "fc1_latent"), ("q8_0", 4096, 32, "fc2_latent"),
          ("iq2_xxs", 5376, 16, "shared up"), ("iq2_xxs", 4096, 21, "shared down")]
print(f"{'codec':8} {'shape':12} {'M':>5} {'fused us':>9} {'GB/s':>6} {'TFLOP':>6} "
      f"{'cuBLAS us':>9} {'frozen us':>9}")
for codec, n, nb, tag in SHAPES:
    packed = make_packed((n, nb), codec, seed=1)
    w = oracle_decode(packed, codec).bfloat16() if n * nb < 10**7 else \
        torch.randn(n, nb * WEIGHTS[codec], device="cuda").bfloat16()
    k = nb * WEIGHTS[codec]
    for m in (1, 4, 16, 64, 256, 1024, 4096):
        x = torch.randn(m, k, device="cuda").bfloat16()
        t = do_bench(lambda: hopper.block_linear(x, packed, codec)) * 1e3
        tc = do_bench(lambda: F.linear(x, w)) * 1e3
        tf = do_bench(lambda: frozen_bm128(x, packed, codec)) * 1e3 if m <= 64 else float("nan")
        gbs = (n * nb * BYTES[codec] + 2 * m * (k + n)) / t / 1e3
        tfl = 2 * m * n * k / t / 1e6
        print(f"{codec:8} {tag:12} {m:5d} {t:9.1f} {gbs:6.0f} {tfl:6.1f} {tc:9.1f} {tf:9.1f}")

print("\nMoE (IQ2_XXS, relu2), Bolt's largest layer: E=432 K=1024 I=2560 topk=9")
E, K, I, topk = 432, 1024, 2560, 9
w13p = make_packed((E, I, K // 256), "iq2_xxs", seed=3, scale=1e-3)
w2p = make_packed((E, K, I // 256), "iq2_xxs", seed=4, scale=1e-3)
_align = test_moe.fake_align
_memo = {}


def memo_align(topk_ids, *a, **kw):
    key = topk_ids.data_ptr()
    if key not in _memo:
        _memo[key] = _align(topk_ids, *a, **kw)
    return _memo[key]


import sys  # noqa: E402

sys.modules["vllm.model_executor.layers.fused_moe.moe_align_block_size"].moe_align_block_size = memo_align
for m in (1, 16, 256, 2048):
    x = torch.randn(m, K, device="cuda").bfloat16()
    tw, ti = torch.topk(torch.softmax(torch.randn(m, E, device="cuda"), -1), topk)
    ti = ti.int()
    t = do_bench(lambda: hopper.moe_forward(x, w13p, w2p, tw, ti)) * 1e3
    touched = torch.unique(ti).numel()
    byts = touched * 2 * I * (K // 256) * 66
    flops = 2 * m * topk * 2 * I * K
    # dense bf16 upper bound: one cuBLAS GEMM with the same FLOPs
    a = torch.randn(m * topk, K, device="cuda").bfloat16()
    b = torch.randn(2 * I, K, device="cuda").bfloat16()
    tc = do_bench(lambda: F.linear(a, b)) * 1e3
    print(f"M={m:5d} experts={touched:3d} fused {t:8.1f} us  {byts / t / 1e3:6.0f} GB/s "
          f"{flops / t / 1e6:6.1f} TFLOP/s | same-FLOP dense cuBLAS {tc:8.1f} us")
