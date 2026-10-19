# 16 · What happens on Hopper today

**What you'll learn:** what happens, step by step, if you run `vllm serve nvidia/Qwen3.8-Flash-Next-FP4-Local` or `nvidia/GLM-5.3-Flash-FP4-Local` on an H200 with upstream vLLM today; and, once loading is fixed, which kernel each layer type would use and what might still break. This is the gap list that the plan (notes 18 and 20) closes.

⚠️ Everything in this note comes from reading the code at `3e96f9cf63`. Nobody has run these models on Hopper yet. Phase B's job is to turn each ⚠️ into ✅ or a bug report.

---

## 1. Today: loading fails (both models)

### Qwen FP4-Local

1. `modelopt_mixed` is picked; min capability 75 passes.
2. `_prepare_weights` globs `*.safetensors` in the snapshot root. All shards are in `tensors/`, so it finds nothing and raises `Cannot find any model weights with …` (`default_loader.py:259`).

**Stops here.**

### GLM FP4-Local

1. Same config path.
2. The glob finds the top-level `model-*.safetensors`, but **not** `tensors/native-QB.safetensors`.
3. Weight loading reaches the first `…weight_scale.csf32_bases` tensor. No parameter has that name, so it most likely raises `There is no module or parameter named …` (`models/utils.py:443`).
4. Even if it didn't, `w13_weight_scale` would stay uninitialized (garbage), and the `native-QB` BF16 weights would be missing.

**Stops here.**

## 2. Gaps, in the order you hit them

| # | Gap | Affects | Fixed by |
|---|---|---|---|
| L1 | Shards only in `tensors/` aren't found | Qwen | PR #60646 (fallback to the index) |
| L2 | An extra subfolder shard next to top-level shards is missed | GLM | not fixed by #60646 (its fallback runs only when the glob is empty); needs "always add index-listed files" |
| L3 | Slab CSF streams (`.nvfp4_csf_fixed`, `.nvfp4_csf_exceptions`) | Qwen, LIL GLM `main` | PR #60646 |
| L4 | `csf_32x2_v1` streams (`.csf32_*`, 5 per scale) | NVIDIA GLM | nobody yet |
| L5 | `weight_scale_encoding` never validated | both | small check: refuse unknown encodings with a clear error |
| L6 | `input_scale` in a separate file `model-inputscales.safetensors` | GLM | probably fine once L2 is fixed (it's in the index) ⚠️ |

## 3. After loading is fixed: layer by layer on an H200

### Qwen FP4-Local

| Layer | Format | Kernel vLLM should pick on sm_90 | Math | Status |
|---|---|---|---|---|
| Routed experts (48) | NVFP4 W4A4 (CSF → raw at load) | Humming MoE (`HummingGroupedExperts` / batched), or Marlin with `--moe-backend marlin` | W4A16, `input_scale` dropped | ⚠️ |
| MTP experts | `W4A16_NVFP4` | Humming MoE | W4A16 | ⚠️ |
| Shared experts | MXFP8 | Humming linear (W8A16) | W8A16 | ⚠️ |
| Gated DeltaNet `in_proj_qkv/z`, `out_proj` | MXFP8 | Humming linear | W8A16 | ⚠️ |
| Gated DeltaNet `in_proj_a/b` (N = 48) | MXFP8 | Humming linear | W8A16 | ❓ small N: check Humming accepts it, else it falls back |
| QSA `q/k/v/o`, indexer | MXFP8 | Humming linear | W8A16 | ⚠️ |
| QSA attention | | sm_90 config tables exist (`nvidia/ops/qsa.py:558`) | | ⚠️ |
| Gated DeltaNet recurrence | | FLA / Triton kernels | | ⚠️ |
| Hyper-connections | BF16 | CuTe DSL kernels (sm_90+) | | ⚠️ |
| PLE n-gram table | NVFP4 | Triton lookup (`ngram_embedding.py:521`), table in pinned CPU memory | | ⚠️ |
| Vision attention | MXFP8 | Humming linear | W8A16 | ⚠️ |
| Vision MLP, merger | `W4A16_NVFP4` | Humming linear (or Marlin) | W4A16 | ⚠️ |
| Embedding, LM head | BF16 | normal | | ✅ |

### GLM FP4-Local

| Layer | Format | Kernel on sm_90 | Math | Status |
|---|---|---|---|---|
| Routed experts (42) | NVFP4 W4A4 (CSF → raw at load), `swiglu_limit = 10` | Humming MoE or Marlin (both apply the clamp) | W4A16 | ⚠️ |
| MTP routed experts | `W4A16_NVFP4` | Humming MoE | W4A16 | ⚠️ |
| Shared experts | FP8 128 × 128 block | block-FP8 linear (DeepGEMM / CUTLASS) | W8A8 | ⚠️ |
| DSA `q_a`, `q_b`, `kv_a`, `o` | FP8 128 × 128 block | block-FP8 linear | W8A8 | ⚠️ |
| DSA `kv_b_proj`, indexer projections | BF16 (from `native-QB`) | normal | | ⚠️ needs L2 |
| DSA attention | | FlashMLA-sparse (`capability.major in [9, 10]`) | | ⚠️ |
| DSA indexer | FP8 cache by default | DeepGEMM `fp8_fp4_paged_mqa_logits` (FP8 mode) | | ⚠️ |
| KDA `q/k/v/o` | MXFP8 | Humming linear | W8A16 | ⚠️ |
| KDA `b/f_a/f_b/g_a/g_b` | BF16 (from `native-QB`) | normal | | ⚠️ needs L2 |
| KDA recurrence | | FlashKDA (sm_90 supported) or Triton | | ⚠️ |
| Dense MLP (layers 0–2) | MXFP8 | Humming linear | W8A16 | ⚠️ |
| LM head | MXFP8 | Humming linear | W8A16 | ⚠️ |
| MTP-layer attention + shared expert | MXFP8 (group 32) | Humming linear | W8A16 | ⚠️ |
| Vision | MXFP8 / `W4A16_NVFP4` | Humming linear | | ⚠️ |

## 4. Known risks on Hopper

1. **Humming on sm_90 is newer than Marlin.** It's the default because of `prioritize_humming`, so it gets less real-world testing on big MoE models. Always compare against `--moe-backend marlin --linear-backend marlin`.
2. **Open Marlin MoE bugs on Hopper:** #49070 (garbage output for an NVFP4 MoE on H100/H200; fix #48624 for unaligned sizes still open) and #48895 (wrong top-k weights in `moe_wna16_marlin_gemm`). Qwen's expert width 640 and GLM's 2048 are both multiples of 128, so the alignment bug may not trigger. Test it.
3. **Tensor parallelism splits expert widths.** Qwen's I = 640: TP 2 → 320 per GPU, TP 4 → 160, TP 8 → 80. Kernels often need multiples of 64 or 128 (b12x says "multiples of 64"). For Qwen on Hopper, prefer **TP 1** (it fits) or **expert parallelism** (`--enable-expert-parallel`), which keeps whole experts.
4. **Fused shared experts.** `maybe_fuse_shared_experts` can merge the shared expert into the routed-expert tensor. That breaks if they use different formats (MXFP8 vs NVFP4). ❓ Check that it stays off.
5. **Speed of W8A16 MXFP8 layers.** Attention and shared experts run with 16-bit math. That's fine for decode, but slower than W8A8 for long prefills. This is a known, separate optimization (not this project).
6. **Report from the field:** #57748 asks about GLM NVFP4 on 4 × H100 with no logs posted. No confirmed Hopper run exists.

## 5. What a correct run looks like (the target after Phase A and B)

- The log shows `Using 'humming' NvFp4 MoE backend` (or `marlin` if forced), and no errors about unknown parameters.
- `vllm serve … --max-model-len 32768` starts on one H200 for Qwen, two for GLM.
- GSM8K (5-shot, `lm_eval`) is within noise of the vanilla NVFP4 checkpoint and of the published Blackwell numbers.
- Humming and Marlin give the same answers (KLD between them near zero; note 21).

---

**Next:** [17 · The FP8 scale problem](17-The-FP8-Scale-Problem.md)
