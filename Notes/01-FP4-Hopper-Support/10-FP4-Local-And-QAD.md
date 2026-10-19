# 10 · FP4-Local and QAD

**What you'll learn:** what the "FP4-Local" checkpoints are and who made them; how quantization-aware distillation (QAD) works, step by step; what's stored in each checkpoint; how accurate they are; how they're served on Blackwell today; and the licenses. This is the third quant, and the one this project targets.

---

## 1. The family

"FP4-Local" is NVIDIA's name for 4-bit checkpoints tuned for **local** hardware: one RTX PRO 6000 Blackwell (96 GB), a DGX Spark (128 GB), or two of either. They were built together with **Local Inference Lab** (LIL, a non-profit community organization), which wrote the training workflow, the CSF compression and the serving containers.

| Repo | Made by | Expert scales | Size | Released |
|---|---|---|---:|---|
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | NVIDIA + LIL | slab `csf` | 95.1 GiB | 5 Oct 2026 📄 |
| `nvidia/GLM-5.3-Flash-FP4-Local` | NVIDIA + LIL | `csf_32x2_v1` | 165.2 GiB | Oct 2026 |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` | LIL (the original) | slab `csf` on `main`; other branches differ (note 06) | 165.5 GiB | |

📄 Model cards: ModelOpt v0.47.0, model version "FP4 QAD 1.0". Both NVIDIA cards start with: "the NVFP4 scale factors in this checkpoint are in a new lossless compression format, but upstream vLLM support for this format has not yet landed. **You must use one of the containers provided below.**"

## 2. QAD, step by step

QAD = **quantization-aware distillation**. NVIDIA's technical report is arXiv [2601.20088](https://arxiv.org/abs/2601.20088), "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery".

The idea: instead of only rounding the weights (PTQ), keep training a 4-bit **student** so its outputs match the full-precision **teacher**.

1. **Teacher.** The original BF16 model generates answers to prompts (taken from NVIDIA's Nemotron datasets; "only prompts from the datasets were used"). For every token it records the next-token probabilities and the final hidden states.
2. **Student.** A copy of the model whose weights are kept in BF16 ("master weights"), but on every forward pass the trainable layers are **fake-quantized**: rounded to NVFP4 (or MXFP8) and back. So the student computes exactly what the 4-bit model will compute.
3. **Loss.** Compare student and teacher on the same tokens:
   - next-token probability matching (Qwen card: **total variation** distance; the general QAD recipe uses KL divergence);
   - hidden-state matching;
   - extra losses on the "end of thinking" and "end of response" decisions, so the 4-bit model stops when the teacher would.
4. **Update.** Gradients flow through the rounding as if it weren't there (the "straight-through estimator") and update the BF16 master weights. Next forward pass, they round to slightly different 4-bit values.
5. **Export.** At the end, round the master weights one last time and save NVFP4 values, E4M3 block scales and FP32 global scales. Then compress the block scales with CSF (note 11). CSF is lossless, so the compression doesn't change the model.

### What was trained (📄 model cards)

| | Qwen FP4-Local | GLM FP4-Local |
|---|---|---|
| Trainer | "Quatrain" distillation trainer | Quatrain |
| Schedule | 2,500 "trunk" updates, then 1,500 "joint-refinement" updates | QAD step 2,500 (LIL report) |
| Data | 500 M tokens, 3 epochs | Nemotron prompts, teacher responses |
| Trained parts | routed experts (NVFP4); in joint refinement also the n-gram tables and simulated MXFP8 shared experts | routed experts (NVFP4); routers, shared experts, dense MLPs and norms are also adapted |
| Frozen | attention (uses its MXFP8 weights) | vision (mostly) |
| Routing during training | | the student uses **its own** expert routing |

### Why QAD weights look "worse" but work better

✅ Measured on Qwen layer 0, expert 0, `down_proj` (scripts in `scripts/`, details in note 17):

| Weights | Error vs the BF16 original matrix |
|---|---:|
| Vanilla PTQ NVFP4 | 8.1 % |
| FP4-Local QAD | 9.6 % |

The QAD matrix is *further* from the original. That's expected: QAD doesn't try to copy each matrix. It moves the weights so the **whole model's output** matches the teacher. Errors in one layer get balanced by changes in others.

**Important consequence for us:** the accuracy of FP4-Local depends on the *exact* 4-bit math it was trained against. A Hopper kernel that changes the math (for example rounding weight × scale into FP8) changes what QAD optimized for. That's why Phase D needs end-to-end accuracy checks (note 17).

## 3. What's inside (summary of notes 05 and 06)

| Component | Qwen FP4-Local | GLM FP4-Local (NVIDIA) |
|---|---|---|
| Routed experts | NVFP4 W4A4, scales `csf` | NVFP4 W4A4, scales `csf_32x2_v1` |
| Attention / linear attention | MXFP8 | KDA q/k/v/o MXFP8; DSA projections FP8 128 × 128 block; small KDA projections, indexer, `kv_b_proj` BF16 |
| Shared experts | MXFP8 | FP8 128 × 128 block |
| Dense MLP | (none) | MXFP8 (layers 0–2) |
| MTP experts | `W4A16_NVFP4`, raw scales | `W4A16_NVFP4`, raw scales |
| N-gram table (PLE) | NVFP4 | (none) |
| LM head | BF16 | MXFP8 |
| Vision | attention MXFP8, MLP `W4A16_NVFP4` | same idea |
| Where the files are | `tensors/*.safetensors` | `model-*.safetensors` + `tensors/native-QB.safetensors` + `model-inputscales.safetensors` |

## 4. Accuracy

📄 Model cards (GPQA Diamond has 198 questions; ToolEvalBench is a tool-calling benchmark):

| Model | GPQA Diamond: BF16 → FP4-Local | ToolEvalBench: BF16 → FP4-Local |
|---|---:|---:|
| Qwen | 92.2 % → 91.5 % | 89.4 % → 88.7 % |
| GLM | 92.1 % → 91.3 % | 91.9 % → 92.4 % |

📄 LIL's KL-divergence report for GLM QAD step 2,500 (524,020 held-out tokens, `rtx6kpro/kld/glm-5.3-flash-qad-step2500.md`):

| Setting | Published NVFP4 (PTQ) | QAD step 2,500 | Change |
|---|---:|---:|---:|
| Same expert routes as BF16 ("exact-route replay") | 0.0675 nats/token | 0.0641 | −5.1 % |
| Each model routes for itself ("natural route") | 0.1622 | 0.1291 | −20.4 % |

So QAD's biggest win is that the 4-bit model's **routing** stays closer to BF16. A behavior test on 7,168 tasks found the two within ±1 point (equivalent).

## 5. How it's served today (Blackwell only)

From the model cards. Containers: `eugr/spark-vllm-b12x:latest` (DGX Spark) and `ghcr.io/local-inference-lab/vllm:release-10152026` (RTX PRO 6000). Both are a **community fork** of vLLM plus b12x. The important flags:

```sh
--quantization modelopt_mixed      # exists upstream too
--load-format b12x                 # fork only (or instanttensor)
--moe-backend b12x --linear-backend b12x
--gdn-decode-kernel b12x           # Qwen
--attention-backend B12X           # GLM
--kv-cache-dtype fp8
export VLLM_B12X_MOE_FP4_FORCE_A16=1   # run experts as W4A16 ...
export B12X_W4A16_A4_PREFILL=1         # ... but use FP4 activations for long prefill
export CUTE_DSL_ARCH=sm_120a           # or sm_121a on Spark
```

Two interesting details:

- Even on Blackwell, the recommended setup runs the experts as **W4A16 for decode**. FP4 activations are only used for prefill. That's the same split this project proposes for Hopper (W4A16 decode, maybe W4A8 prefill).
- The GLM MTP draft model is told to use `"moe_backend": "humming"`. So Humming already runs FP4-Local's `W4A16_NVFP4` MTP experts in production on Blackwell.

## 6. Licenses (read before using in CI or tests)

| Repo | License |
|---|---|
| NVIDIA FP4-Local (both) | NVIDIA Open Model Agreement; base model terms also apply (Qwen Community License 1.0, or MIT for GLM) |
| LIL GLM | **Local Inference Lab License 1.0**: Apache-2.0 text plus extra restrictions. The NOTICE says: "Reuploading this work, or any Substantially Similar Copy of it, is not permitted. Link to the Original Location instead." It also requires attribution on landing pages. Revisions up to `9ca06e2` were MIT. |

Practical rule: **use the NVIDIA repos for tests and docs.** Never re-upload LIL tensors, even small slices, into test fixtures. Generate synthetic test data instead (note 21).

## Key takeaways

- FP4-Local = NVFP4 experts trained with QAD to match BF16, plus 8-bit attention, plus lossless CSF scale compression.
- QAD is tuned to the exact NVFP4 math. Changing the math (FP8 rounding) needs fresh accuracy checks.
- It's served today only through a community fork with b12x on sm_120 / sm_121. Upstream vLLM can't load it.

---

**Next:** [11 · CSF scale compression](11-CSF-Scale-Compression.md)
