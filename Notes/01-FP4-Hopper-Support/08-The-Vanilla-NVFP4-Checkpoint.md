# 08 · The vanilla NVFP4 checkpoint

**What you'll learn:** what NVIDIA's plain ("vanilla") NVFP4 checkpoints of both models quantize and what they leave alone, how they were made, and exactly which kernel vLLM picks for each part on Hopper today. This is the baseline that FP4-Local improves on.

---

## 1. Two vanilla checkpoints

| | `nvidia/Qwen3.8-Flash-Next-NVFP4` | `nvidia/GLM-5.3-Flash-NVFP4` |
|---|---|---|
| Producer | ModelOpt `0.46.0.dev281` | ModelOpt `0.47.0.dev393` |
| `quant_algo` | `MIXED_PRECISION` | `NVFP4` (plus `kv_cache_quant_algo: FP8`) |
| vLLM config class | `ModelOptMixedPrecisionConfig` (`modelopt.py:1562`) | `ModelOptNvFp4Config` (`modelopt.py:728`) |
| `get_min_capability` | 75 | 75 |
| Size | 123.5 GiB ✅ | 190.4 GiB ✅ |
| Method | PTQ, no fine-tuning 📄 | PTQ, recipe `nvfp4_experts_dense_mlp-kv_fp8_cast` 📄 |
| Calibration data | cnn_dailymail + Nemotron-Post-Training-Dataset-v2 📄 | same 📄 |
| Tested on | B200 / B300, TP 8 📄 | GB200, TP 4 📄 |

Both say "NVIDIA Blackwell" in the model card. But both use `get_min_capability = 75` in vLLM, so vLLM *will* try to run them on Hopper.

## 2. What each one quantizes (✅ from `config.json`)

### Qwen vanilla

| Part | Format |
|---|---|
| Routed experts, layers 0–47 | **NVFP4 W4A4** (weights group 16, static FP4 input scale), "MSE-calibrated weight scales" 📄 |
| MTP routed experts | **FP8_PB_WO** (128 × 128 block FP8 weights, dynamic per-128 FP8 activations) |
| PLE n-gram table | **per-tensor FP8**, copied byte for byte from `Qwen/Qwen3.8-Flash-Next-FP8` 📄 |
| Attention, Gated DeltaNet, shared experts, routers, hyper-connection mixers, embedding, LM head, vision | **BF16** (292 entries in `ignore`) |

### GLM vanilla

| Part | Format |
|---|---|
| Routed experts, layers 3–44 | **NVFP4 W4A4** |
| Dense MLP, layers 0–2 | **NVFP4 W4A4** |
| All `self_attn*` (KDA and DSA), routers, shared experts, the whole MTP layer 45, vision, embedding, LM head | **BF16** |
| KV cache | FP8 |

Note: the GLM model card's prose says shared experts are quantized, but its `ignore` list excludes `shared_experts*`. Trust the config.

## 3. Accuracy (📄 model cards)

| Model | Baseline | GPQA Diamond: baseline → NVFP4 |
|---|---|---|
| Qwen | FP8 | 92.0 → 91.5 |
| GLM | BF16 | 92.17 → 92.11 |

## 4. How vLLM runs it on Hopper today, part by part

### NVFP4 routed experts → Humming W4A16

1. `select_nvfp4_moe_backend` (`fused_moe/oracle/nvfp4.py:191`) builds the list FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN, HUMMING, EMULATION.
2. `prioritize_humming` moves HUMMING in front of MARLIN, on sm_90 only.
3. If the model sets `swiglu_limit` (GLM does), only backends that apply the clamp stay. Marlin and Humming both stay.
4. The FlashInfer and CUTLASS FP4 backends need sm_100 or newer, so they fail `is_supported_config`. HUMMING is the first one that passes. `(kNvfp4Static, kNvfp4Dynamic)` is in its supported list (`fused_humming_moe.py:357`).
5. Humming decides the real activation type itself (`check_and_fallback_input_schema`, `humming/schema.py:261`):
   - FP4 activations need sm_100. ❌
   - FP8 E4M3 fails, because weight groups under 32 are refused with 8-bit activations (`humming/schema/humming.py:581` in the Humming repo). ❌
   - E3M4 needs sm_100, and INT8 doesn't match float weights. ❌
   - **BF16 passes. ✅ So the result is W4A16**, and `input_scale` goes unused.
6. If you pass `--moe-backend marlin`, you get `MarlinExperts`, which is also W4A16 (note 07).

⚠️ This chain comes from reading the code. Nobody has run it on an H200 for these models yet. Check the log line `Using 'humming' NvFp4 MoE backend` (note 21).

### NVFP4 dense MLP (GLM layers 0–2) → Humming W4A16

`init_nvfp4_linear_kernel` (`kernels/linear/__init__.py:1139`) goes through `_POSSIBLE_NVFP4_KERNELS` (line 584). On sm_90, Humming is moved ahead of Marlin (`__init__.py:419`). The FlashInfer, CUTLASS, cuDNN and TRT-LLM FP4 kernels need Blackwell.

### FP8 block (Qwen MTP experts) → native FP8 on Hopper

128 × 128 block FP8 with per-128 activations is DeepSeek-V3 style. Hopper runs it natively (DeepGEMM or CUTLASS block FP8 for linears; Humming lists `(kFp8Static128BlockSym, kFp8Dynamic128Sym)` for MoE).

### BF16 parts → ordinary BF16 kernels

### PLE FP8 table → lookup kernel, kept in CPU memory by default

See note 05.

## 5. Why this matters for FP4-Local

- The vanilla routed experts use exactly the same NVFP4 layout as FP4-Local's experts, **except the scales are stored raw** (`weight_scale` E4M3 `[N, K/16]`). So the vanilla Qwen checkpoint is the best **baseline** on Hopper: if it runs correctly, the FP4-Local expert path differs only in how the scales are loaded.
- Phase 0 of the plan (note 18) is: run the vanilla checkpoints on an H200 first, record speed and GSM8K, and use them as the reference.
- The vanilla Qwen is 123.5 GiB on disk. With the PLE table in CPU memory (vLLM's default), about 75.8 GiB sits on the GPU, so it fits one H200 (≈ 131 GiB usable). GLM vanilla (190.4 GiB) needs 2 H200s. FP4-Local (95.1 GiB; with PLE in CPU memory, about 72 GiB on the GPU in Phase A and 68 GiB in Phase C) leaves even more room for KV cache (note 19).

## Key takeaways

- Vanilla NVFP4 = PTQ, W4A4 routed experts, raw E4M3 scales, most other layers BF16.
- On Hopper vLLM should pick Humming and run it as W4A16. ⚠️ Not yet run end to end.
- It's the reference baseline for every FP4-Local test on Hopper.

---

**Next:** [09 · GGUF and GSQ-RCO](09-GGUF-And-GSQ-RCO.md)
