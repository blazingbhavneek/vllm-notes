# 03 · What quantization is

**What you'll learn:** what weights and activations are, how a "scale" lets a 4-bit number stand in for a real weight, every scale layout in this project (NVFP4, MXFP8, block FP8), the WxAy naming (W4A16, W8A8, W4A4, W4A8), how memory is counted, and how PTQ, QAT and QAD differ.

---

## 1. What is inside a model

A language model is mostly **linear layers**. A linear layer multiplies its input by a weight matrix:

```
y = x · Wᵀ
x : [M, K]   M input rows (tokens), K input features
W : [N, K]   N output features ("rows" of the weight), K input features
y : [M, N]
```

- **Weights** (`W`) are fixed numbers learned in training. They are most of the model's size.
- **Activations** (`x`, `y`) are the numbers flowing through the model for the current tokens. They change with every request.
- The **KV cache** stores attention keys and values for past tokens. It's a separate memory cost (notes 19 and 21).

In a Mixture-of-Experts (MoE) model, each "expert" is a small MLP with three matrices: `gate_proj` and `up_proj` (shape `[I, H]`) and `down_proj` (shape `[H, I]`), where H is the hidden size and I is the expert's intermediate size. vLLM fuses gate and up into one matrix called **w13**, and calls down **w2**.

## 2. The core idea: store a small number and a scale

A weight like `w = 0.0173` doesn't fit in FP4, which only has 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 (note 02). So we store:

```
q = round_to_fp4(w / s)     ← stored, 4 bits
s                           ← stored, the scale
w ≈ q × s                   ← "dequantize" when we need the value
```

If `s = 0.0058`, then `w / s = 2.98`, which rounds to `q = 3`, and `q × s = 0.0174`. Close.

One scale for the whole matrix is a bad idea: one large weight (an "outlier") forces a big scale, and then all the small weights round to 0. So formats share one scale among a **small group** of weights.

## 3. Scale granularity

| Granularity | One scale per | Example |
|---|---|---|
| Per tensor | whole matrix | FP8 "per-tensor"; NVFP4's second-level global scale |
| Per channel | output row (N) | FP8 per-channel |
| Per group | row × K-group of g weights | NVFP4 (g = 16), MXFP8 (g = 32), GPTQ INT4 (g = 128) |
| Per block | 2-D tile, for example 128 × 128 | FP8_PB_WO (DeepSeek-style block FP8) |

## 4. The formats in this project, exactly as stored in the checkpoints

### NVFP4 (two-level scales)

```
w ≈ q × s_block × s_global
q        : FP4 E2M1, two per byte          weight          uint8   [N, K/2]
s_block  : FP8 E4M3, one per 16 weights    weight_scale    f8_e4m3 [N, K/16]
s_global : FP32, one per matrix            weight_scale_2  f32     []
s_input  : FP32, for FP4 activations       input_scale     f32     []   (W4A4 only)
```

How a quantizer picks them (NVIDIA ModelOpt's recipe):

1. `s_global = max|W| / (6 × 448)`. This makes the biggest block scale land near E4M3's maximum, 448.
2. For each group of 16: `s_block = round_to_e4m3( max|group| / (6 × s_global) )`.
3. `q = round_to_fp4( w / (s_block × s_global) )`.

✅ Real numbers: a Qwen FP4-Local expert has `s_global = 5.99 × 10⁻⁵` and block scales from 14 to 448.

### MXFP8

```
w ≈ q × 2^(e − 127)
q : FP8 E4M3                         weight         f8_e4m3 [N, K]
e : UE8M0 (power of two), per 32     weight_scale   uint8   [N, K/32]
```

### MXFP4 (for reference)

FP4 E2M1 values with a UE8M0 scale per 32. Not used by these checkpoints, but b12x and Marlin support it, and it's the "easy" FP4 for an FP8 kernel because a power-of-two scale is exact (note 17).

### FP8 128 × 128 block ("FP8_PB_WO" in ModelOpt)

```
w ≈ q × s
q : FP8 E4M3                                  weight         f8_e4m3 [N, K]
s : FP32, one per 128 × 128 tile              weight_scale   f32     [N/128, 1, K/128, 1] (GLM files) 
```

✅ vLLM runs this as W8A8: activations are quantized at runtime with one scale per token per 128 columns (`modelopt.py:2712-2723`, "PbWo: 128x128 block-static weight, dynamic per-block activation"). This is the DeepSeek-V3 style that Hopper runs natively with DeepGEMM or CUTLASS.

## 5. Bits per weight (how much memory each format costs)

| Format | Value bits | Scale bits per weight | Total bits per weight |
|---|---:|---:|---:|
| BF16 | 16 | 0 | 16 |
| FP8 block 128×128 | 8 | 32 / 16384 ≈ 0.002 | ≈ 8.0 |
| MXFP8 | 8 | 8 / 32 = 0.25 | 8.25 |
| NVFP4 | 4 | 8 / 16 = 0.5 | **4.5** |
| NVFP4 + CSF (FP4-Local) | 4 | ≈ 0.27 (about 54 % of 0.5) | **≈ 4.27** |
| MXFP4 | 4 | 8 / 32 = 0.25 | 4.25 |

Worked example, Qwen3.8-Flash-Next routed experts (✅ from the checkpoint headers):

```
48 layers × 512 experts × 3 matrices × 2560 × 640 = 120.8 billion weights
BF16:        120.8e9 × 2 bytes    = 225 GiB
FP4 values:  120.8e9 × 0.5 byte   = 56.25 GiB
NVFP4 scales (raw): 120.8e9 / 16  = 7.03 GiB
NVFP4 scales (CSF):                = 3.69 GiB   (saves 3.34 GiB)
```

## 6. The WxAy names

"W" is the weight format, "A" is the activation format, and the number is bits.

| Name | Weights in memory | Activations | Multiply happens in | Hardware |
|---|---|---|---|---|
| **W16A16** | BF16 | BF16 | BF16 | any |
| **W4A16** (weight-only) | FP4 or INT4 | BF16 | BF16, after unpacking weights inside the kernel | any modern GPU (Marlin, Humming) |
| **W8A16** | FP8 | BF16 | BF16, after unpacking | any (Marlin, Humming) |
| **W8A8** | FP8 | FP8 (quantized at runtime) | FP8 tensor cores | Ada (sm_89), Hopper, Blackwell |
| **W4A8** | FP4 or INT4 | FP8 | FP8 tensor cores, after unpacking weights to FP8 | Hopper (vLLM `cutlass_w4a8` for INT4), Blackwell |
| **W4A4** (NVFP4 both) | FP4 | FP4 (quantized at runtime) | FP4 tensor cores | **Blackwell only** |

Important consequences:

- An "NVFP4" checkpoint is a *W4A4* recipe (it even ships `input_scale`). But any GPU can run it as **W4A16**: keep the 4-bit weights, ignore `input_scale`, and multiply in BF16. That's what vLLM does on Hopper today (note 16).
- "W4A16_NVFP4" in a checkpoint means the producer *intended* weight-only. The weight bytes are the same NVFP4 layout.

## 7. Static vs dynamic activation scales

- **Static:** the scale is measured once on sample data ("calibration") and saved (`input_scale`). Fast at runtime, but wrong if real inputs look different.
- **Dynamic:** the kernel computes the scale from the actual activations every step, for example one scale per token, or per token per 128 columns. A bit more work, more robust.

On Hopper, FP8 activations are usually **dynamic per token** or **per token per 128** (DeepGEMM style).

## 8. Dequantize where? The key to keeping memory small

"Dequantize" means turning `q` and `s` back into a real number. You can do it in two places:

1. **Materialized:** unpack the whole matrix once into BF16 or FP8 and store it in GPU memory. Simple, but the memory saving is gone.
2. **In-kernel:** keep the 4-bit weights in memory. Inside the matrix-multiply kernel, load a small tile into registers, unpack it there, multiply, and throw the unpacked values away. **Memory stays 4-bit.** This is what Marlin, Humming and b12x do.

Notice: whether the kernel unpacks to BF16 or to FP8 *inside registers* makes **no difference to memory**. It only changes how fast the multiply is, and how exact it is (note 17). The memory savings come from what's *stored*: FP4 weights, compressed scales, and no materialized copies.

## 9. Where accuracy goes

Quantization adds rounding error to every weight. Two things make it worse:

- **Outliers**, large values that stretch a group's scale.
- **Error adding up** through many layers.

✅ Measured on one real Qwen expert matrix (note 17): the FP4-Local weights differ from the original BF16 weights by 9.6 % (relative, root-mean-square). The vanilla PTQ NVFP4 weights differ by 8.1 %. QAD weights are *further* from the original matrix, but the model *as a whole* behaves closer to the original. That's the point of QAD (next section).

## 10. PTQ, QAT and QAD

| Method | What happens | Cost | Example here |
|---|---|---|---|
| **PTQ** (post-training quantization) | Take the trained model, pick scales from a little calibration data, round. No training. | Minutes to hours | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| **QAT** (quantization-aware training) | Keep training while simulating quantization in the forward pass, so weights learn to survive rounding. | Expensive; can be unstable for models that went through SFT, RL and merging | — |
| **QAD** (quantization-aware distillation) | A full-precision **teacher** model produces outputs; a quantized **student** is trained to match them with a KL-divergence loss. | Medium (hundreds of millions of tokens) | FP4-Local (both models) |

📄 NVIDIA's QAD report (arXiv [2601.20088](https://arxiv.org/abs/2601.20088)) says QAD is more stable than QAT for models trained with SFT, RL and merging, and works without the original training data. FP4-Local used 500 million tokens for 3 epochs (Qwen model card). Note 10 has details.

GGUF GSQ-RCO is a *third* kind: a smarter PTQ that learns the rounding choices and picks a different format per tensor (note 09).

## 11. How accuracy is measured

- **Benchmarks:** the share of correct answers on a test set (GSM8K math, GPQA science, LiveCodeBench code, and so on). Easy to understand, but noisy.
- **KL divergence (KLD):** compare the quantized model's next-token probabilities with the original's on the same text, in "nats per token". 0 means identical. More sensitive than benchmarks. 📄 Local Inference Lab measured GLM-5.3-Flash: vanilla NVFP4 0.0675 nats/token vs QAD step 2500 0.0641, under the same expert routing as BF16.
- **Perplexity:** how surprised the model is by real text. Lower is better.

For a kernel change, the rule is: **the new kernel must give the same answers as the reference kernel, within numerical noise.** If it changes the math on purpose (like FP8 rounding in Phase D), you must measure KLD and benchmarks.

## Key takeaways

- A quantized weight is a small number times a scale. Small groups (16 or 32) keep accuracy.
- NVFP4 = FP4 values + E4M3 scale per 16 + FP32 global scale. It costs 4.5 bits per weight, or about 4.27 with CSF.
- Any GPU can run an NVFP4 checkpoint as W4A16. Only Blackwell can run it as W4A4.
- Memory savings come from what's stored. Unpacking inside the kernel keeps them.

---

**Next:** [04 · How a GPU does matrix math](04-How-A-GPU-Does-Matrix-Math.md)
