# Usage: python w4a8_error_study.py DIR   (DIR from csf_slab_decode.py, plus
#   bf16_down_e0.bin fetched with st_slice.py for the BF16 comparison). Needs numpy.

# How much error does a Hopper FP8 (W4A8) path add for one real NVFP4 expert?
# Compares W4A16 (exact NVFP4 weights, BF16-ish activations) with FP8 options.
import sys
import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)

def e4m3_table():
    v = []
    for code in range(256):
        s, e, m = code >> 7, (code >> 3) & 15, code & 7
        if e == 15 and m == 7:
            v.append(np.nan); continue
        val = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        v.append(-val if s else val)
    return np.array(v)

E4M3 = e4m3_table()
POS = np.unique(E4M3[(E4M3 >= 0) & ~np.isnan(E4M3)])  # sorted non-negative values

def round_e4m3(x):
    """Round to nearest E4M3 (ties to even), saturating at 448."""
    a = np.minimum(np.abs(x), 448.0)
    i = np.clip(np.searchsorted(POS, a), 1, len(POS) - 1)
    lo, hi = POS[i - 1], POS[i]
    pick_hi = (hi - a < a - lo) | ((hi - a == a - lo) & (i % 2 == 0))
    return np.sign(x) * np.where(pick_hi, hi, lo)

d = sys.argv[1]
s = E4M3[np.load(f"{d}/qloc_down.scale_decoded.npy")]          # [N, K/16]
packed = np.fromfile(f"{d}/qloc_down.weight.bin", np.uint8).reshape(2560, 320)
q = E2M1[np.stack((packed & 15, packed >> 4), -1).reshape(2560, 640)]  # low nibble first
N, K = q.shape
s16 = np.repeat(s, 16, axis=1)
w_exact = q * s16                               # global scale left out (common factor)

nz = s[s > 0]
row_span = np.log2(np.where(s > 0, s, np.nan).max(1) / np.where(s > 0, s, np.nan).min(1))
print(f"expert down_proj N={N} K={K}; scales: min {nz.min():.4g} max {nz.max():.4g}")
print(f"binades spanned by the scales inside one row: median {np.nanmedian(row_span):.2f}, max {np.nanmax(row_span):.2f}")

def fused(block):
    """FP4 x scale fused into an FP8 payload, with one power-of-two exponent per
    (row, K block). block = K for one exponent per row."""
    nb = K // block
    mx = (np.abs(q) * s16).reshape(N, nb, block).max(-1)          # largest |w| per block
    e = np.ceil(np.log2(np.where(mx > 0, mx, 1) / 448.0))          # keep payload <= 448
    p2 = np.repeat(2.0 ** e, block, axis=1)
    payload = round_e4m3(q * s16 / p2)
    return payload * p2

rng = np.random.default_rng(0)
x = rng.standard_normal((256, K)) * np.exp(rng.standard_normal((1, K)))  # outlier-ish channels
y_ref = x @ w_exact.T

def act_fp8(x, block=128):
    xb = x.reshape(x.shape[0], -1, block)
    sc = np.abs(xb).max(-1, keepdims=True) / 448.0
    return (round_e4m3(xb / sc) * sc).reshape(x.shape)

def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

print(f"\n{'variant':58s} {'weight err':>10s} {'inexact':>8s} {'GEMM err':>9s}")
for name, w in [("per-row exponent (whole scale fused, no promotion)", fused(K)),
                ("per-(row, K/128) exponent (promotion every 128)", fused(128)),
                ("per-(row, K/32) exponent (b12x-style residual)", fused(32))]:
    inexact = np.mean(w != w_exact)
    print(f"{name:58s} {rel(w, w_exact):10.4%} {inexact:8.2%} {rel(x @ w.T, y_ref):9.4%}")
xa = act_fp8(x)
print(f"{'FP8 activations only (per token x 128), exact weights':58s} {'-':>10s} {'-':>8s} {rel(xa @ w_exact.T, y_ref):9.4%}")
w = fused(128)
print(f"{'FP8 activations + per-(row, K/128) fused weights':58s} {rel(w, w_exact):10.4%} {np.mean(w != w_exact):8.2%} {rel(xa @ w.T, y_ref):9.4%}")

# Context: how far are the 4-bit weights from the original BF16 weights?
import os
bf16_path = f"{d}/bf16_down_e0.bin"
if os.path.exists(bf16_path):
    raw16 = np.fromfile(bf16_path, np.uint16).reshape(N, K)
    w_bf16 = (raw16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    gs = float(np.fromfile(f"{d}/qloc_down.ws2.bin", np.float32)[0])
    w_q = w_exact * gs
    print(f"\nFP4-Local (QAD) weights vs original BF16: {rel(w_q, w_bf16):.2%} relative error")
    sp = E4M3[np.fromfile(f"{d}/qptq_down.scale.bin", np.uint8).reshape(N, K // 16)]
    pp = np.fromfile(f"{d}/qptq_down.weight.bin", np.uint8).reshape(N, K // 2)
    qp = E2M1[np.stack((pp & 15, pp >> 4), -1).reshape(N, K)]
    w_ptq = qp * np.repeat(sp, 16, axis=1) * float(np.fromfile(f"{d}/qptq_down.ws2.bin", np.float32)[0])
    print(f"vanilla PTQ NVFP4 weights vs original BF16: {rel(w_ptq, w_bf16):.2%} relative error")
    print(f"GEMM vs BF16 weights: QAD W4A16 {rel(x @ w_q.T, x @ w_bf16.T):.2%}, "
          f"QAD W4A8 (fused, FP8 act) {rel(xa @ (w*gs).T, x @ w_bf16.T):.2%}, "
          f"PTQ W4A16 {rel(x @ w_ptq.T, x @ w_bf16.T):.2%}")
