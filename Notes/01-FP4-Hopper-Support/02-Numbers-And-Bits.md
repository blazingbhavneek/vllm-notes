# 02 · Numbers and bits

**What you'll learn:** how a computer stores a number in 32, 16, 8 or 4 bits; how to read the formats used in this project (BF16, FP16, FP8 E4M3, FP8 E5M2, UE8M0, FP4 E2M1) by hand; and what "exactly representable" means. Note 17 depends on this.

---

## 1. Bits and bytes

- A **bit** is 0 or 1.
- A **byte** is 8 bits. It can hold 2⁸ = 256 different patterns.
- A **nibble** is 4 bits, half a byte. It can hold 16 patterns.

A number format is a rule that says which pattern means which number. The same byte `0x38` means 56 as an integer, but **1.0** as an FP8 E4M3 number. The bits are the same; only the rule differs.

## 2. Floating point in one picture

A floating-point number has three parts:

```
value = sign × 2^(exponent − bias) × 1.mantissa
```

- **sign**: 1 bit. 0 means positive, 1 means negative.
- **exponent**: tells you the size ("which power of two"). The stored bits are shifted by a fixed **bias** so that small and large values both fit.
- **mantissa** (also called fraction): the digits after the binary point. The leading `1.` is not stored; it's implied. That's why a format with *m* mantissa bits has *m + 1* **significant bits**.

Example in FP32: 6.0 = +1 × 2² × 1.1 (binary 1.1 means 1 + 1/2 = 1.5, and 1.5 × 4 = 6).

**Subnormals.** When the exponent bits are all zero, the implied `1.` becomes `0.`. That lets the format store numbers smaller than the smallest normal number, with fewer significant bits. You'll see this in the E4M3 table below.

## 3. The formats in this project

| Name | Bits: sign / exponent / mantissa | Significant bits | Largest value | Used for |
|---|---|---:|---:|---|
| FP32 | 1 / 8 / 23 | 24 | 3.4 × 10³⁸ | Accumulators inside kernels, global scales |
| BF16 | 1 / 8 / 7 | 8 | 3.4 × 10³⁸ | Normal model weights and activations |
| FP16 | 1 / 5 / 10 | 11 | 65 504 | Older default; still used in some kernels |
| FP8 E4M3 | 1 / 4 / 3 | 4 | **448** | FP8 weights and activations; NVFP4 block scales |
| FP8 E5M2 | 1 / 5 / 2 | 3 | 57 344 | Sometimes gradients; rarely here |
| UE8M0 | 0 / 8 / 0 | 1 (power of two only) | 2¹²⁷ | MXFP8 / MXFP4 block scales |
| FP4 E2M1 | 1 / 2 / 1 | 2 | **6** | NVFP4 and MXFP4 weights |

"E4M3" literally means "4 exponent bits, 3 mantissa bits". "UE8M0" means "unsigned, 8 exponent bits, 0 mantissa bits": it can only be a power of two.

## 4. FP4 E2M1: all 16 values

FP4 is so small you can list every value. Bias is 1.

| Bits (s e e m) | Value | Bits (s e e m) | Value |
|---|---:|---|---:|
| 0 00 0 | 0 | 1 00 0 | −0 |
| 0 00 1 | 0.5 | 1 00 1 | −0.5 |
| 0 01 0 | 1 | 1 01 0 | −1 |
| 0 01 1 | 1.5 | 1 01 1 | −1.5 |
| 0 10 0 | 2 | 1 10 0 | −2 |
| 0 10 1 | 3 | 1 10 1 | −3 |
| 0 11 0 | 4 | 1 11 0 | −4 |
| 0 11 1 | 6 | 1 11 1 | −6 |

So FP4 can only say: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6. That's it. To store a weight like 0.0173, you must first divide it by a **scale** so it lands near one of these values (note 03).

**Packing.** Two FP4 values share one byte. In NVFP4 checkpoints the **low nibble holds the first (even) element** and the high nibble holds the next one. A weight matrix with K columns is stored as `uint8 [N, K/2]`.

## 5. FP8 E4M3: reading a byte by hand

Bias is 7. Layout: `s eeee mmm`.

- Normal (exponent bits 1 to 15): value = (−1)ˢ × 2^(e−7) × (1 + m/8)
- Subnormal (exponent bits 0): value = (−1)ˢ × 2^(−6) × (m/8)
- The pattern `s 1111 111` is NaN. There is no infinity. That's why the largest value is 2⁸ × 1.75 = **448**.

Worked examples:

| Byte | Bits | Value |
|---|---|---|
| `0x38` | 0 0111 000 | 2⁰ × 1.0 = **1.0** |
| `0x3C` | 0 0111 100 | 2⁰ × 1.5 = **1.5** |
| `0x40` | 0 1000 000 | 2¹ × 1.0 = **2.0** |
| `0x7E` | 0 1111 110 | 2⁸ × 1.75 = **448** (the largest) |
| `0x01` | 0 0000 001 | 2⁻⁶ × 1/8 = **2⁻⁹ ≈ 0.00195** (the smallest above 0) |

An important property for CSF (note 11): **for positive E4M3 values, a bigger byte means a bigger number.** So "byte + small offset" stays in the same neighborhood of values. Each step of +8 in the byte doubles the value.

## 6. UE8M0: a pure power of two

One byte `e` means 2^(e − 127). Byte 127 is 1.0, byte 128 is 2.0, byte 126 is 0.5. It can't store 1.5. MX formats (MXFP8, MXFP4) use it as the block scale. Multiplying by a power of two only changes the exponent, so it's **always exact** (as long as you don't run out of exponent range).

## 7. "Exactly representable"

A value is **exactly representable** in a format if some bit pattern of that format equals it with no rounding.

The rule of thumb: **a format can hold a product exactly only if it has at least as many significant bits as the product needs** (and enough exponent range).

- FP4 → FP8 E4M3: every FP4 value (0.5, 1, 1.5, 2, 3, 4, 6) fits in E4M3. ✅ **Exact.**
- FP4 → BF16: ✅ exact.
- FP4 × E4M3 scale → BF16: the FP4 part has 2 significant bits and the scale has 4, so the product needs up to 2 + 4 = 6. BF16 has 8. ✅ **Exact.** This is why W4A16 kernels lose nothing when they unpack.
- FP4 × E4M3 scale → FP8 E4M3: needs up to 6, E4M3 has 4. ❌ **Not always exact.**

Worked example of the last case:

```
FP4 value    1.5   = 1.1    (binary)
E4M3 scale   1.125 = 1.001  (binary)
product      1.6875 = 1.1011 (binary)  ← needs 5 significant bits
nearest E4M3 values: 1.625 (1.101) and 1.75 (1.110)
```

1.6875 sits exactly between them. Rounding picks one (rule below), and the result is off by 3.7 %.

How often does this happen? A script that tries every FP4 value against every E4M3 mantissa ([scripts/fp4_scale_exactness.py](scripts/fp4_scale_exactness.py)) finds: ✅

- 41 of 56 combinations (73.2 %) are exact. FP4 values 0.5, 1, 2 and 4 are powers of two, so they're always exact.
- The values 1.5, 3 and 6 can need rounding.
- Worst rounding error: 4.76 %. Average (root-mean-square) over all combinations: 1.75 %.

Note 17 explains what this means for a Hopper FP8 kernel.

## 8. Rounding: "round to nearest, ties to even"

When a value doesn't fit, hardware picks the **nearest** representable value. If it's exactly halfway, it picks the one whose last mantissa bit is 0 ("even"). In the example above, 1.625 is `1.101` (last bit 1) and 1.75 is `1.110` (last bit 0), so 1.6875 rounds to **1.75**.

## 9. Range: overflow and underflow

Significant bits aren't the only limit. The exponent limits the range.

- **Overflow:** FP4's biggest value times E4M3's biggest value is 6 × 448 = 2688. That's bigger than 448, so it can't be stored in E4M3. A kernel must divide everything by a power of two first and multiply back later. Dividing by a power of two is exact.
- **Underflow:** very small products fall into the subnormal range, where E4M3 has fewer significant bits, or they become 0.

On real FP4-Local weights, the scales in one matrix row span at most 3.4 powers of two (✅ measured on one Qwen and one GLM expert, note 17). E4M3's normal range spans about 14.8 powers of two (2⁻⁶ to 448). So with one power-of-two "shift" per row, nothing overflows or underflows. Only the significant-bit rounding remains.

## 10. Try it yourself

```python
# FP4 E2M1 decoding of one byte that holds two values (low nibble first).
E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
byte = 0x7A                       # binary 0111 1010
lo, hi = byte & 0xF, byte >> 4    # lo = 1010 = 10, hi = 0111 = 7
print(E2M1[lo], E2M1[hi])         # -1.0  6.0

# FP8 E4M3 decoding of one byte.
def e4m3(b):
    s, e, m = b >> 7, (b >> 3) & 0xF, b & 0x7
    v = (m / 8) * 2**-6 if e == 0 else (1 + m / 8) * 2**(e - 7)
    return -v if s else v
print(e4m3(0x38), e4m3(0x7E))     # 1.0  448.0
```

## Key takeaways

- FP4 has only 15 distinct values (from −6 to 6), so it needs scales.
- E4M3 has 4 significant bits and tops out at 448.
- FP4 → FP8 and FP4 × scale → BF16 are exact. FP4 × E4M3 scale → FP8 is **not always exact**. That one fact shapes the whole Phase D design.

---

**Next:** [03 · What quantization is](03-What-Quantization-Is.md)
