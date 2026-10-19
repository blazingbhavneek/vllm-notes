# 19 · VRAM budget on H200

**What you'll learn:** how much GPU memory each model needs on an H200 in each phase, where it goes, how much is left for the KV cache, and how many GPUs to plan for. Weight numbers are ✅ from the checkpoint headers. KV and state numbers are estimates computed from the configs (marked ≈); vLLM prints the real number at startup ("GPU KV cache size: … tokens").

---

## 1. The H200

- 141 GB = **131.3 GiB**.
- vLLM uses `--gpu-memory-utilization` of it (default 0.9): about **118 GiB**.
- Out of that come: weights, the KV cache and linear-attention states, activations, CUDA graphs and kernel workspaces (budget about 4–6 GiB for these).
- Host (CPU) RAM matters too: Qwen's PLE table sits in **pinned host memory** by default (26.9 GiB).

## 2. Qwen3.8-Flash-Next FP4-Local on 1 × H200

| Component | Phase A (scales decoded at load) | Phase C (CSF kept in VRAM) |
|---|---:|---:|
| Routed-expert FP4 values | 56.25 | 56.25 |
| Routed-expert scales | 7.03 (raw) | 3.69 (CSF) + 0.29 scratch (2 × 146 MB) |
| Attention + Gated DeltaNet (MXFP8 + BF16 parts) | 3.2 | 3.2 |
| Embedding + LM head (BF16) | 2.4 | 2.4 |
| MTP experts (`W4A16_NVFP4`) | 1.3 | 1.3 |
| Everything else (norms, routers, mixers, shared experts, vision) | 1.4 | 1.4 |
| **Weights on GPU** | **≈ 71.6 GiB** | **≈ 68.5 GiB** |
| PLE n-gram table | 26.9 GiB **host** memory | 26.9 GiB host memory |

What's left for everything else at 0.9 utilization: about 118 − 71.6 − 5 ≈ **41 GiB** (Phase A).

### KV cache and states (≈)

- **KV cache:** only the 12 QSA layers (plus the MTP layer) keep a per-token KV cache: 2 KV heads × 256 × (K + V) = 1,024 values per layer per token. In FP8: about **13 KiB per token** in total (with a little extra for the indexer cache).
- **Gated DeltaNet state:** a fixed size per sequence: 36 layers × 48 heads × 128 × 128 × 4 bytes (FP32) ≈ **108 MiB per sequence**, whatever its length.

So 41 GiB holds roughly **3 million tokens** of FP8 KV cache (minus 108 MiB per running sequence). That's far more than needed. **Qwen isn't memory-bound on an H200, even in Phase A.** Phase C matters more for smaller GPUs and for GLM.

Sanity check against the model card: one RTX PRO 6000 (89.4 GiB) holds "850K tokens of FP8 KV cache" with about 68 GiB of weights. 850K × 13 KiB ≈ 10.5 GiB, which fits the leftover. ✅ Consistent.

### Options

- `--kv-cache-dtype fp8` halves the KV cache (Hopper supports FP8 KV).
- Putting PLE on the GPU (`--engram-config '{"cpu_offload": false}'`) costs 26.9 GiB of VRAM. It still fits (≈ 98.5 GiB of weights). Measure whether it's faster.

## 3. GLM-5.3-Flash FP4-Local on 2 × H200

| Component | Phase A | Phase C |
|---|---:|---:|
| Routed-expert FP4 values (incl. MTP layer) | 145.1 | 145.1 |
| Routed-expert scales | 17.7 (raw) + 0.4 MTP | 9.6 (CSF) + 0.4 MTP + 0.84 scratch (2 × 0.42) |
| Attention (KDA + DSA + mHC) | 6.3 | 6.3 |
| Embedding, LM head, shared experts, dense MLP, vision, other | 3.8 | 3.8 |
| **Total weights** | **≈ 173.3 GiB** | **≈ 166.0 GiB** |
| **Per GPU with TP 2 (or EP 2)** | **≈ 86.7 GiB** | **≈ 83.0 GiB** |

Left per GPU at 0.9 utilization: about 118 − 86.7 − 5 ≈ **26 GiB** (Phase A), about 30 GiB (Phase C).

### KV cache and states (≈, ❓ measure)

- The 11 DSA layers (+ MTP) use MLA: one 512-wide compressed KV per token per layer, plus an indexer key cache. With FP8 that's on the order of 0.5–0.7 KiB per token per layer. MLA caches are usually **replicated** on each TP rank, not split.
- KDA state per sequence: 34 layers × 64 heads × 128 × 128 × 4 bytes ≈ 136 MiB (split across TP ranks by heads).
- The model card claims 1.8 M tokens of FP8 KV cache on 2 × RTX PRO 6000. That doesn't match this simple estimate, so vLLM's token accounting for this hybrid model (or a compressed index pool, `index_kpool = 4`) must differ. ❓ Read the startup log instead of trusting the estimate.

### GPU count for each GLM checkpoint

| Checkpoint | Weights | H200s needed |
|---|---:|---:|
| FP4-Local (Phase C) | 166 GiB | 2 |
| FP4-Local (Phase A) | 173 GiB | 2 |
| Vanilla NVFP4 | 190.4 GiB | 2 |
| Official FP8 | 305.8 GiB | 4 (TP has to divide 64 heads, so not 3) |
| BF16 | 598.5 GiB | 8 (≈ 75 GiB per GPU) |

## 4. Summary: what each phase buys

| | Qwen (1 × H200) | GLM (2 × H200, per GPU) |
|---|---:|---:|
| Vanilla NVFP4 weights on GPU | 75.8 GiB | 95.2 GiB |
| FP4-Local, Phase A | 71.6 GiB | 86.7 GiB |
| FP4-Local, Phase C | 68.5 GiB | 83.0 GiB |
| Phase D (W4A8) | same as C (in-register unpack only) | same as C |

Phase D adds **no** memory: it changes what happens in registers, not what's stored. Its activation-quantization buffers are small (one FP8 copy of a prefill chunk's activations: 8192 tokens × 4096 × 1 byte = 32 MiB for GLM).

---

**Next:** [20 · Implementation plan for the agent](20-Implementation-Plan-For-The-Agent.md)
