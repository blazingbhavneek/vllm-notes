# 09 · GGUF and GSQ-RCO

**What you'll learn:** what a GGUF file is; how the main GGUF block formats (Q4_0, the K-quants, the IQ quants) store weights; what GSQ and RCO do; what IST-DASLab's `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` actually contains; and how (and how well) vLLM can run it. This is the second of the three quants you asked about.

---

## 1. GGUF in one paragraph

**GGUF** is the file format of **llama.cpp**, the most popular engine for running models on one machine. A GGUF file holds metadata (architecture, tokenizer, hyperparameters) and a list of tensors. **Each tensor can use a different storage type.** The types are "block" formats: weights are cut into small blocks along a row, and each block stores small integers plus its own scale. GGUF formats were designed for **software** unpacking (CPU SIMD, or CUDA integer dot products), not for any GPU's tensor-core number formats.

## 2. The block formats you'll meet

Values are for one block. "bpw" is bits per weight including scales. These numbers are the standard llama.cpp values.

| Type | Block | What's stored | bpw |
|---|---:|---|---:|
| **Q8_0** | 32 | 32 × INT8 + one FP16 scale `d`. `w = d × q` | 8.5 |
| **Q4_0** | 32 | 32 × 4-bit + one FP16 scale. `w = d × (q − 8)` | 4.5 |
| **Q5_0** | 32 | like Q4_0, with 5-bit values | 5.5 |
| **Q2_K … Q6_K** ("K-quants") | 256 ("super-block") | the 256 values are split into sub-blocks of 16 or 32; each sub-block has a small quantized scale (and for some types a minimum); the super-block has FP16 scales for those scales | Q2_K 2.625, Q3_K 3.44, Q4_K 4.5, Q5_K 5.5, Q6_K 6.56 |
| **IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S** ("i-quants") | 256 | groups of 4 or 8 weights are replaced by an **index into a fixed codebook** (a lookup table of allowed patterns), plus signs and scales. More accuracy per bit, but decoding needs table lookups | ≈ 2.06, 2.31, 2.5, 3.06, 3.44 |
| **IQ4_XS** | 256 | 16 non-uniform levels (a fixed non-linear grid) per weight, 6-bit sub-block scales | 4.25 |
| **IQ4_NL** | 32 | 16 non-uniform levels, one FP16 scale per 32 | 4.5 |
| **Q2_0** | small (fits 640-wide rows) | a newer simple 2-bit type **without** a codebook (📄 model card: it "avoids lookup-table formats") | ≈ 2.4 in this release |

Two practical rules:

- **Blocks of 256 need rows divisible by 256.** Qwen's `down_proj` experts have K = 640, and 640 / 256 = 2.5. So for `ffn_down_exps`, only the small-block types fit (in this release: Q2_0 and IQ4_NL). 📄
- **Codebook types are slow to decode.** Every few weights need a table lookup. On this model that cost dominates: Q2_0 gives **3.4 × the prompt throughput** of IQ2_XS (367 vs 108 tokens/s in llama.cpp) at a slightly smaller size. 📄

## 3. GSQ: better rounding

**GSQ** (Gumbel-Softmax Quantization, arXiv [2604.18556](https://arxiv.org/abs/2604.18556), code [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ)) is a post-training method. Plain rounding sends each weight to the *nearest* grid value. GSQ instead **learns** which grid value each weight should use, and learns the group scales too, by gradient descent on a smooth relaxation (the "Gumbel-Softmax" trick turns a hard "pick one of N" choice into a soft, differentiable one). The result is still a normal GGUF block format, so llama.cpp can run it unchanged. It's PTQ: the model's weights aren't retrained, only their quantized form is optimized.

## 4. RCO: choosing a format per tensor

**RCO** (Riemannian Constrained Optimization, arXiv [2605.00649](https://arxiv.org/abs/2605.00649), code [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO)) answers: "I have N tensors and K formats, and a total size budget. Which format should each tensor get?" It writes the size budget as a smooth surface (a Riemannian manifold) and does gradient descent on the model's task loss while staying exactly on budget.

The process for this release (📄 model card):

1. **Database.** Quantize every tensor with GSQ at every candidate type.
2. **Search.** RCO picks one type per tensor so the file hits the target bits per weight.
3. **Assemble.** Stitch the chosen versions into one standard GGUF.

The search covers **352 tensors**: 304 dense ones (attention, SSM projections, shared experts, embedding, output head) and 48 fused routed-expert tensors, **one per layer per matrix type**. GGUF stores all 512 experts of a layer in one 3-D tensor (`blk.N.ffn_gate_exps`), so all experts of a layer share one type. It can't vary per expert.

## 5. What's in the release (📄 + ✅ allocation files)

| Build | bpw (transformer weights) | Shard 1 (must be resident) | Shard 2 (n-gram table, IQ4_NL) | Total |
|---|---:|---:|---:|---:|
| Q2_0 | 2.40 | 37.6 GB | 28.8 GB | 66.4 GB |
| IQ2_XS | 2.50 | 39.2 GB | 28.8 GB | 68.0 GB |
| IQ3_XXS | 3.00 | 47.0 GB | 28.8 GB | 75.8 GB |
| IQ3_S | 3.50 | 54.8 GB | 28.8 GB | 83.6 GB |
| `mmproj` (vision, BF16) | 16 | 0.91 GB | | 0.91 GB |

Shard 2 is the same in every build. It holds the 51.2-billion-entry n-gram table at IQ4_NL (4.5 bpw). It's a lookup, so llama.cpp can leave it memory-mapped on disk (`-lm mmap --lazy-mode on`).

How the routed experts were allocated (✅ from `tensor-allocation/*.rco-allocation.txt`, 48 layers):

| Build | `gate_exps` / `up_exps` | `down_exps` |
|---|---|---|
| Q2_0 | all Q2_0 | all Q2_0 |
| IQ2_XS | mostly IQ2_S | Q2_0 |
| IQ3_XXS | mix of IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS | 18 × IQ4_NL + 30 × Q2_0 |
| IQ3_S | 20 × IQ2_S, 10 × IQ3_S, 17 × IQ3_XXS, 1 × IQ4_XS | 39 × IQ4_NL + 9 × Q2_0 |

Example: the Q2_0 build has 1223 tensors: BF16 483, F32 292, Q2_0 202, Q3_K 91, IQ4_XS 56, Q4_K 38, Q4_0 16, Q5_K 16, Q6_K 12, IQ4_NL 7, Q5_0 7, Q8_0 2, F16 1. Its layer 0 uses Q4_K for `attn_gate`, IQ4_XS for `attn_qkv`, Q5_0 / IQ4_XS / Q3_K for the shared-expert down / gate / up, and keeps the hyper-connection mixers and routers in BF16.

## 6. Accuracy (📄 model card, against BF16)

| Variant | AIME25 | GPQA-Diamond | LiveCodeBench v6 | Task average |
|---|---:|---:|---:|---:|
| BF16 | 100.00 | 91.92 | 87.43 | 93.12 |
| Q2_0 | 96.67 | 89.39 | 81.14 | 89.07 |
| IQ2_XS | 96.67 | 87.37 | 83.43 | 89.16 |
| IQ3_XXS | 100.00 | 91.41 | 86.29 | 92.57 |
| IQ3_S | 100.00 | 92.93 | 86.86 | 93.26 |

Note: these numbers aren't directly comparable with FP4-Local's (different harness, GPQA settings and baseline run), so don't rank across tables.

## 7. Running it in vLLM

- GGUF support moved out of vLLM into **[vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)** in June 2026. The docs still call it "highly experimental and under-optimized".
- The plugin's CUDA kernels come from llama.cpp build b2899 (May 2024). It has a one-token kernel (MMVQ) for every type, but its batched kernel (MMQ) covers only the older Q and K types. For **IQ types**, any step with more than about 8–16 tokens **unpacks the whole weight to BF16 every step** and then runs a normal matmul. That's slow, and it's the batched case vLLM exists for.
- Mixed types inside one fused layer (q/k/v, gate/up) cause a per-step copy (`quantization/linear.py:247-257` in the plugin).
- ❓ Whether the plugin has an adapter for `qwen4_exp` (hyper-connections, PLE, QSA, MTP) isn't verified. It has adapters for Qwen3.5 / 3.6. Assume it doesn't until someone tests it.
- These problems and the fixes are tracked as G1–G4 in `issue-tracker.md`. They're a **separate project** from FP4-Local on Hopper.

On Hopper specifically: GGUF kernels don't use FP4 or FP8 tensor-core formats on *any* GPU. They use integer dot products or unpack to BF16. So "Blackwell only" never applied to GGUF. Its limits are kernel quality, not hardware.

## Key takeaways

- GSQ-RCO is standard GGUF: better rounding (GSQ) plus a per-tensor format choice under a size budget (RCO).
- It's much smaller than FP4-Local (37.6–54.8 GB resident vs about 68 GiB on GPU), because the experts go down to 2–3.5 bits.
- It runs anywhere llama.cpp runs. In vLLM it depends on a plugin with old kernels, and Qwen3.8-Flash-Next support isn't confirmed.

---

**Next:** [10 · FP4-Local and QAD](10-FP4-Local-And-QAD.md)
