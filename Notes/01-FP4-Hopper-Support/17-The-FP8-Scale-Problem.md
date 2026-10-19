# 17 · The FP8 scale problem

**What you'll learn:** why "unpack FP4 to FP8 inside the kernel" can't be exact for NVFP4; every way around it, and why most of them fail; the one approach that works on Hopper (a power-of-two shift per row in the epilogue plus one rounding in registers); and the measurements on real FP4-Local weights that say how much accuracy it costs. This is the theory behind Phase D. Phases A to C don't need it.

---

## 1. The goal, in one formula

For one output element of an NVFP4 GEMM:

```
y[m, n] = s_global · Σ_k  x[m, k] · q[n, k] · s[n, g(k)]
           FP32           act      FP4       E4M3 scale of k's 16-group
```

On Hopper we want the multiply-add to run on **FP8 tensor cores** (`wgmma`, 2× the BF16 rate), with:

- weights kept as FP4 in VRAM (memory goal), unpacked to FP8 in registers;
- activations quantized to FP8 at runtime.

## 2. Why the obvious way doesn't work

The tensor core multiplies two FP8 numbers. So the weight operand must be **one FP8 number per weight**: some FP8 encoding of `q · s`.

- `q` alone fits FP8 exactly (note 02). But a `wgmma` instruction covers K = 32 elements, which is **two** 16-groups with **two different** scales. The hardware can't apply a different scale to each half (no block-scale inputs on Hopper).
- Applying the scale **after** the multiply ("promotion", note 04) would need a separate instruction per 16 values. Impossible: one FP8 instruction is already 32 wide.
- So the scale must go **into the operand**: `w' = q · s`, rounded to FP8. And `q · s` needs up to 6 significant bits, while E4M3 has 4. **Rounding is unavoidable.**

✅ From `scripts/fp4_scale_exactness.py` (every FP4 value × every E4M3 mantissa):

| | Value |
|---|---|
| exact combinations | 41 of 56 (73.2 %) |
| always exact | FP4 values 0.5, 1, 2, 4 (powers of two) |
| sometimes rounded | FP4 values 1.5, 3, 6 |
| worst rounding error | 4.76 % |
| RMS rounding error (uniform over combinations) | 1.75 % |

## 3. The second problem: range

`q · s` can be as big as 6 × 448 = 2688, above E4M3's maximum of 448. It can also be tiny. Fix: divide by a power of two `2^E` before rounding, and multiply it back later. Dividing by a power of two is exact (it only changes the exponent), so it costs no accuracy, as long as everything stays inside E4M3's normal range.

How big must the "window" be? ✅ Measured:

- Qwen FP4-Local L0 E0 `down_proj`: scales from 14 to 448. Inside one row they span a median of **1.54** doublings (max **3.36**).
- GLM FP4-Local L3 E0 `gate_proj`: median **2.0**, max **3.0** doublings per row.
- FP4 values add another 3.6 doublings (0.5 to 6).
- Total ≤ 7 doublings. E4M3's normal range is about 14.8 doublings (2⁻⁶ to 448).

So **one power of two per row** keeps every product in range. Nothing overflows or becomes subnormal.

## 4. The approach that works on Hopper

**At load time (or in the Phase C scale decoder), for each weight row `n`:**

```
E[n] = ceil( log2( max_k |q[n,k] · s[n,g(k)]| / 448 ) )        one small integer per row
```

**In the kernel, per weight (in registers):**

```
w'[n,k] = round_to_e4m3( q[n,k] · s[n,g(k)] / 2^E[n] )          FP8 operand
```

Concretely: decode the nibble to FP16 (exact), multiply by `s / 2^E[n]` in FP16 (exact: FP16 has 11 significant bits), convert to E4M3 with round-to-nearest-even. A handful of instructions per element; Humming already does a similar fused conversion for MXFP4 (`use_fused_e8m0_scale`).

**Activations:** quantize `x` to FP8 per token, per 128 columns (DeepGEMM style): `x ≈ a · t[m, b]`.

**Accumulate:** for each 128-wide K block `b`, run 4 `wgmma` (4 × 32) into a temporary accumulator, then promote:

```
acc[m,n] += t[m,b] · tmp[m,n]                  (the promotion DeepGEMM already does)
```

**Epilogue:**

```
y[m,n] = s_global · 2^E[n] · acc[m,n]           one extra multiply per output
```

Variants:

- **Per-token activation scale** (one per token): no promotion at all; `t[m]` joins the epilogue. Faster; a bit less accurate on outlier-heavy activations.
- **Exponent per (row, 128-block)** instead of per row: free if you promote every 128 anyway (fold `2^E[n,b]` into the promotion multiply). Gives more headroom if some row has an unusually wide spread.

### The same idea in b12x (Blackwell)

b12x's "W4A8 on NVFP4" (note 14) does the same rounding, but puts the power of two on each 32-wide block and lets the **hardware** apply it (block-scaled MMA). On Hopper there's no such hardware, so the exponent moves to the epilogue (per row) or to the promotion step (per 128). ✅ The measurements below show this makes **no difference** to accuracy: as long as nothing leaves the normal range, rounding depends only on the significant bits, not on which power of two you divided by.

### vLLM precedent

`cutlass_w4a8` (Hopper, INT4 weights × FP8 activations) also applies its group scales in FP8 *before* the `wgmma`, after converting them to FP8, with a per-channel scale in the epilogue (note 07). So "round weight × scale into FP8" is already accepted in vLLM for INT4.

## 5. Approaches that don't work (and why)

| Idea | Why not |
|---|---|
| Unpack to FP8 and apply the scale after the MMA | needs promotion every 16 values; one FP8 instruction is 32 wide |
| Promotion every 32 values with one combined scale | the two 16-groups in a 32-chunk have different scales |
| Split each weight into two FP8 numbers (hi + lo) to stay exact | two MMAs → no speed gain over BF16 |
| INT8 instead of FP8 | same per-16 scale problem, and `2·q × (8 + m)` exceeds INT8's range |
| Convert NVFP4 → MXFP4 at load (power-of-two scales, exact in FP8) | rounding the **scales** to powers of two is a much bigger change than rounding products; it also changes what QAD trained |
| Store FP8 weights in VRAM (W8A8) | doubles expert memory (56 → 112.5 GiB for Qwen); breaks the memory goal |
| Run FP4 activations (W4A4) | Hopper has no FP4 tensor cores (note 13) |

## 6. How much accuracy does it cost? (✅ measured)

Setup (`scripts/w4a8_error_study.py`): Qwen FP4-Local L0 E0 `down_proj` ([2560, 640]), decoded CSF scales; 256 synthetic activation rows with outlier-heavy channels; errors are relative RMS against the exact W4A16 result.

| Variant | Weight error | Weights changed | GEMM error |
|---|---:|---:|---:|
| Fused FP8 weights, exponent per row | 2.11 % | 22.3 % | 2.13 % |
| Fused FP8 weights, exponent per (row, 128) | 2.11 % | 22.3 % | 2.13 % |
| Fused FP8 weights, exponent per (row, 32) (b12x-style) | 2.11 % | 22.3 % | 2.13 % |
| FP8 activations only (per token × 128), exact weights | — | — | 2.16 % |
| **Both (full W4A8)** | 2.11 % | 22.3 % | **3.04 %** |

For scale, the error from 4-bit quantization itself, against the original BF16 weights (same matrix):

| | vs BF16 original |
|---|---:|
| FP4-Local (QAD) weights | 9.56 % |
| Vanilla PTQ weights | 8.13 % |
| GEMM: QAD W4A16 | 9.55 % |
| GEMM: QAD **W4A8** | **10.00 %** |
| GEMM: PTQ W4A16 | 8.11 % |

How to read this:

1. The weight rounding (2.1 %) is about the same size as the error from FP8 activations, which every W8A8 deployment already accepts.
2. On top of FP4's own ~9.6 % error, W4A8 adds less than half a percentage point on this matrix.
3. **But** QAD tuned the model to the exact W4A16 math (note 10). One matrix can't tell you how errors add up over 48 layers × 3 projections, or how they shift routing. **End-to-end KL divergence and GSM8K are required** before turning this on by default (note 21).

## 7. When would W4A8 actually be faster?

Only when the expert GEMMs are compute-bound (note 04): tokens per expert ≳ 100 on an H200.

| Prefill chunk (`--max-num-batched-tokens`) | Qwen tokens / expert | GLM tokens / expert |
|---:|---:|---:|
| 4096 | 80 | 114 |
| 8192 | 160 | 228 |
| 16384 | 320 | 455 |

So W4A8 is a **prefill-only** option (decode stays W4A16), and it only pays off with big chunks. The ceiling is 2× on the expert GEMMs; expect less because of the extra unpack work. Phase D starts with a benchmark that proves the gain exists before any kernel work (note 18).

## Key takeaways

- NVFP4 × FP8 can't be exact on Hopper: the per-16 E4M3 scale has to be folded into the FP8 weight, which needs rounding.
- A per-row power-of-two shift (applied in the epilogue) fixes the range problem at no accuracy cost; one rounding per weight remains.
- Measured cost on one real expert: ~2 % per GEMM, similar to FP8 activations. It needs end-to-end evals, and it only helps long prefills.

---

**Next:** [18 · Design for Hopper support](18-Design-Hopper-Support.md)
