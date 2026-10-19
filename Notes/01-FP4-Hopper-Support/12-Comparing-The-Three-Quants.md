# 12 · Comparing the three quants

**What you'll learn:** how the three quantized versions of Qwen3.8-Flash-Next differ, side by side: how each was made, what it stores, how big it is, how accurate it is, and what it takes to run it on Hopper. Everything here is explained in notes 08 to 11; this note puts it in one place.

The three:

- **A. Vanilla NVFP4:** `nvidia/Qwen3.8-Flash-Next-NVFP4` (note 08)
- **B. GGUF GSQ-RCO:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (note 09)
- **C. FP4-Local (QAD):** `nvidia/Qwen3.8-Flash-Next-FP4-Local` (note 10), and its GLM sibling

---

## 1. How they were made

| | A. Vanilla NVFP4 | B. GSQ-RCO GGUF | C. FP4-Local |
|---|---|---|---|
| Made by | NVIDIA (ModelOpt 0.46) | IST-DASLab | NVIDIA + Local Inference Lab (ModelOpt 0.47, Quatrain) |
| Method | **PTQ**: MSE-calibrated scales, round once | **Smarter PTQ**: GSQ learns roundings and scales; RCO picks a format per tensor under a size budget | **QAD**: 4,000 training updates on 500 M tokens × 3, student matches the BF16 teacher |
| Weights changed? | no (only rounded) | no (only rounded, but rounding chosen by optimization) | **yes** (BF16 master weights trained, then rounded) |
| Calibration / training data | cnn_dailymail + Nemotron v2 | (per the GSQ / RCO papers) | Nemotron v2 / v3 prompts + teacher answers |

## 2. What they store

| | A. Vanilla NVFP4 | B. GSQ-RCO GGUF | C. FP4-Local |
|---|---|---|---|
| Routed experts | NVFP4: FP4 E2M1 + E4M3 scale / 16 + FP32 global, **raw scales** | one GGUF type per layer: Q2_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL… (2–4.5 bpw) | NVFP4, **CSF-compressed scales** |
| Expert bits per weight | 4.5 | ≈ 2–3.5 | ≈ 4.27 |
| Attention | BF16 | per tensor (Q4_K, IQ4_XS, Q6_K…) | MXFP8 |
| Shared experts | BF16 | per tensor | MXFP8 |
| N-gram table (PLE) | FP8 per tensor, 47.7 GiB | IQ4_NL, 28.8 GB | NVFP4, 26.9 GiB |
| MTP | experts FP8 128 × 128 | (not in the GGUF files) ❓ | experts `W4A16_NVFP4` |
| Activations intended | FP4 (W4A4) | any (software) | FP4 (W4A4) |
| Container | safetensors (ModelOpt config) | GGUF (2 shards + `mmproj`) | safetensors under `tensors/` (ModelOpt mixed config) |

## 3. Size and memory (Qwen)

| | A. Vanilla NVFP4 | B. GSQ-RCO (IQ3_S / Q2_0) | C. FP4-Local |
|---|---:|---:|---:|
| Download | 123.5 GiB | 83.6 GB / 66.4 GB | 95.1 GiB |
| Must be on the GPU (PLE in CPU memory or on disk) | ≈ 75.8 GiB | 54.8 GB / 37.6 GB | ≈ 71.5 GiB after Phase A; ≈ 68.2 GiB after Phase C |
| Fits on one H200 (≈ 131 GiB usable)? | yes, with room for KV cache | yes | yes |

(GPU-resident = total − PLE table. Vanilla: 123.5 − 47.7 = 75.8 GiB. FP4-Local keeps CSF in VRAM only after Phase C: 95.1 − 26.9 = 68.2 GiB. In Phase A the scales are expanded at load, which adds 7.03 − 3.69 = 3.3 GiB, giving 71.5 GiB. Note 19 has the full budget.)

## 4. Accuracy (📄 model cards; different harnesses, so compare only within a row)

| | Baseline | Quant |
|---|---|---|
| A. Vanilla NVFP4 | GPQA-D 92.0 (FP8) | 91.5 |
| B. GSQ-RCO IQ3_S | GPQA-D 91.92 (BF16) | 92.93 |
| B. GSQ-RCO Q2_0 | GPQA-D 91.92 (BF16) | 89.39 |
| C. FP4-Local | GPQA-D 92.2 (BF16) | 91.5 |

For GLM, LIL's KL-divergence study (note 10) is the best evidence that QAD beats PTQ: −20 % KLD under natural routing.

## 5. Hardware and software support

| | A. Vanilla NVFP4 | B. GSQ-RCO GGUF | C. FP4-Local |
|---|---|---|---|
| Model card hardware | Blackwell (B200 / B300) | anything llama.cpp runs on | Blackwell (RTX PRO 6000, DGX Spark) |
| Upstream vLLM loads it? | ✅ (`--quantization modelopt`) | via `vllm-gguf-plugin`, and `qwen4_exp` support isn't confirmed ❓ | ❌ (CSF unknown; shards in `tensors/`) |
| Native FP4 tensor-core path | Blackwell only | none, on any GPU (integer / software decode) | Blackwell only |
| Kernel on Hopper | Humming (or Marlin) W4A16 ⚠️ | GGUF MMVQ / MMQ, or unpack to BF16 for IQ types | same as A once it loads ⚠️ |
| Main Hopper gap | testing only | plugin kernel quality, model adapter | **CSF loading** (+ testing), then CSF in VRAM, then speed |

## 6. The short version

- **A and C use the same number format** (NVFP4). They differ in how the numbers were chosen (PTQ vs QAD) and how the scales are stored (raw vs CSF). On Hopper, once C can be loaded, it runs through exactly the same kernels as A.
- **B is a different world.** Different file format, different kernels, different engine plugin. It's smaller, and it's not tied to any GPU generation, but vLLM's GGUF path is the weakest of the three.
- **Why "Blackwell only" for A and C?** Only because the *fast W4A4* path needs FP4 tensor cores. Weight-only W4A16 works on Hopper. Note 13 explains this fully.

---

**Next:** [13 · Why Blackwell only](13-Why-Blackwell-Only.md)
