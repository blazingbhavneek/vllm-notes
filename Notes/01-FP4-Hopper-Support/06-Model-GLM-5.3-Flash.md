# 06 · Model: GLM-5.3-Flash

**What you'll learn:** how GLM-5.3-Flash is built; the shapes of its tensors; which format each part uses in NVIDIA's and Local Inference Lab's FP4-Local checkpoints; and where its code is in vLLM. Numbers are ✅ from `config.json` and safetensors headers unless marked 📄.

---

## 1. The model in one paragraph

GLM-5.3-Flash (Hugging Face architecture `Glm5NextForConditionalGeneration`, vLLM model type `glm5_next`) is a multimodal MoE model from Z.ai. It has 45 decoder layers plus one MTP layer. Three of every four layers use **KDA** (Kimi Delta Attention, a linear attention with a fixed-size state). Every fourth layer uses **DSA** (DeepSeek Sparse Attention: MLA attention plus an "indexer" that picks the 2048 most useful past tokens). The first 3 layers have an ordinary dense MLP; the other 42 have an MoE block with **288 routed experts** (8 per token) and 1 shared expert. Like Qwen, it carries 4 residual streams, here called **mHC** (manifold-constrained hyper-connections).

📄 NVIDIA's model card: "320B in total and 18B activated". Context up to 1M tokens.

## 2. Key config numbers (`text_config`)

| Field | Value | Meaning |
|---|---:|---|
| `hidden_size` | 4096 | residual stream width (H) |
| `num_hidden_layers` | 45 | decoder layers; the MTP layer is stored as layer **45** |
| `layer_types` | 34 × `linear_attention` (KDA), 11 × `deepseek_sparse_attention` | DSA at layers 3, 7, …, 43 |
| `mlp_layer_types` | 3 × `dense`, 42 × `sparse` | `first_k_dense_replace = 3` |
| `intermediate_size` | 12 288 | dense MLP width (layers 0–2) |
| `n_routed_experts` / `num_experts_per_tok` | 288 / 8 | E / top_k |
| `moe_intermediate_size` | 2048 | expert width (I) |
| `n_shared_experts` | 1 | |
| `scoring_func` / `topk_method` | sigmoid / `noaux_tc` | DeepSeek-V3-style router with a correction bias |
| `routed_scaling_factor` | 2.5 | |
| `swiglu_limit` | 10.0 | experts clamp the SwiGLU input; the MoE kernel must support this |
| `hc_mult` / `hc_sinkhorn_iters` | 4 / 20 | mHC: 4 streams, mixing matrix normalized by Sinkhorn iterations |
| `linear_attn_config.num_heads` / `head_dim` | 64 / 128 | KDA shape |
| `num_attention_heads`, `q_lora_rank`, `kv_lora_rank` | 64, 1536, 512 | MLA shape for DSA |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 256 / 0 / 256 | no rotary part (`mla_use_nope`) |
| `index_topk`, `index_n_heads`, `index_head_dim` | 2048, 32, 128 | the DSA indexer |
| `vocab_size` | 154 880 | |

## 3. One decoder layer, step by step

1. **mHC mixing** (`hc_attn_fn [24, 16384]`, `hc_attn_base`, `hc_attn_scale`): combines the 4 streams (4 × 4096 = 16 384 numbers) into one input.
2. **Token mixing**, one of two kinds:
   - **KDA** (34 layers), tensors under `self_attn.*`:
     - `q_proj`, `k_proj`, `v_proj`: each `[8192, 4096]` (64 heads × 128).
     - `o_proj [4096, 8192]`.
     - `q_conv1d`, `k_conv1d`, `v_conv1d`: short convolutions, `[8192, 1, 4]`.
     - `f_a_proj [128, 4096]` and `f_b_proj [8192, 128]`: a low-rank "forget gate".
     - `g_a_proj` and `g_b_proj`: a low-rank output gate.
     - `b_proj [64, 4096]`: the per-head write strength.
     - `A_log`, `dt_bias`, `o_norm`.
   - **DSA** (11 layers), tensors under `self_attn.*`:
     - Queries: `q_a_proj [1536, 4096]`, then `q_b_proj [16384, 1536]` (64 heads × 256).
     - Keys and values: `kv_a_proj_with_mqa [512, 4096]`, then `kv_b_proj [32768, 512]` (64 heads × (256 key + 256 value)).
     - `o_proj`.
     - Indexer: `indexer.wq_b [4096, 1536]`, `indexer.wk [128, 4096]`, `indexer.weights_proj [32, 4096]`, `indexer.index_kpool_compress_gate [128, 4096]`, norms.
3. **mHC mixing** again (`hc_ffn_*`).
4. **MLP**:
   - Layers 0–2: dense `gate_proj` and `up_proj [12288, 4096]`, `down_proj [4096, 12288]`.
   - Layers 3–44: router `gate [288, 4096]` (BF16) plus `e_score_correction_bias [288]`; 288 routed experts with `gate_proj` and `up_proj [2048, 4096]` and `down_proj [4096, 2048]`; one shared expert of the same shape.

The **MTP layer** (layer 45) has DSA attention, its own 288 experts, `eh_proj [4096, 8192]`, `enorm`, `hnorm` and `shared_head`.

## 4. The routed experts

Count: 42 layers × 288 experts × 3 matrices × (4096 × 2048) = **304.4 billion weights**, plus 7.2 billion in the MTP layer's experts.

| Tensor (NVIDIA FP4-Local) | dtype | gate / up | down |
|---|---|---|---|
| `….weight` | U8 (FP4 pairs) | [2048, 2048] | [4096, 1024] |
| `….weight_scale.csf32_bases` | U8 | [2048] | [4096] |
| `….weight_scale.csf32_codes` | U8 | [262 144] | [262 144] |
| `….weight_scale.csf32_page_bases` | U32 | [32] | [32] |
| `….weight_scale.csf32_offsets` | U16 | [8224] | [8224] |
| `….weight_scale.csf32_exceptions` | U16 | a few hundred to ~1500 | similar |
| `….weight_scale_2` | F32 | scalar | scalar |
| `….input_scale` | F32 (in `model-inputscales.safetensors`) | scalar | scalar |

The uncompressed scale would be E4M3 `[2048, 256]` (gate, up) or `[4096, 128]` (down). Note 11 decodes the five CSF streams.

## 5. Formats in the NVIDIA FP4-Local checkpoint (✅ `quantized_layers`)

| Format | Count | Which tensors |
|---|---:|---|
| `NVFP4` + `weight_scale_encoding: "csf_32x2_v1"` | 42 | all routed experts of layers 3–44 |
| `FP8_PB_WO` (128 × 128 block FP8) | 170 | shared experts (42 × 3); DSA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` (11 × 4) |
| `MXFP8` (`weight_block_size [1, 32]`) | 146 | KDA `q/k/v/o_proj` (34 × 4); dense MLP of layers 0–2 (3 × 3); `lm_head` |
| `MXFP8` (`group_size 32`) | 59 | vision attention (24 × 2); MTP layer's shared expert (3) and attention projections (8) |
| `W4A16_NVFP4` | 77 | vision MLP (24 × 3) and merger (4); MTP layer's routed experts (1) |
| not listed (BF16 / FP32) | — | embedding, routers, norms, mHC; KDA `b_proj`, `f_a/f_b`, `g_a/g_b`; DSA `kv_b_proj` and indexer projections |

✅ Several of the BF16 tensors (KDA small projections, DSA indexer, `kv_b_proj`) are stored in a **subdirectory** shard, `tensors/native-QB.safetensors`. That matters for the loader (note 16).

### Local Inference Lab's original checkpoint is different

`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` has several branches (✅ from the Hub refs API):

| Branch | Experts | Non-expert layers |
|---|---|---|
| `main` (`dec48ab`) | NVFP4, slab `csf` | 579 × MXFP8 (almost everything, including KDA small projections and shared experts) |
| `qad2-csf` | NVFP4, slab `csf` | |
| `qad2-kda-w8a16-csf` (`96f27f8`) | NVFP4, `csf_32x2_v1` | **identical** recipes to NVIDIA's FP4-Local |
| `qad2-kda-w8a16` (`bcaca2f`) | NVFP4, **uncompressed** scales | same as above |
| `qad2-kda-bf16`, `qad2`, `fp8` | variants | |

✅ The expert scales in NVIDIA's FP4-Local decode to exactly the bytes in LIL's `qad2-kda-w8a16` branch (checked on `layers.3.mlp.experts.0.gate_proj`). So NVIDIA's GLM FP4-Local is LIL's `qad2-kda-w8a16-csf` build. The uncompressed branch is handy as a test reference.

## 6. What each checkpoint weighs (GiB, ✅)

| Checkpoint | Total |
|---|---:|
| `zai-org/GLM-5.3-Flash-BF16` | 598.5 |
| `zai-org/GLM-5.3-Flash` (the official release is **FP8**, block-scaled) | 305.8 |
| `nvidia/GLM-5.3-Flash-NVFP4` (vanilla PTQ) | 190.4 |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` (`main`) | 165.5 |
| `nvidia/GLM-5.3-Flash-FP4-Local` | **165.2** |

NVIDIA FP4-Local by component:

| Component | GiB |
|---|---:|
| Routed experts, FP4 values (including the MTP layer) | 145.1 |
| Routed experts, CSF scales (5 streams) | 9.6 |
| MTP-layer expert scales (uncompressed) | 0.4 |
| Attention (KDA + DSA + mHC attention side) | 6.3 |
| Embedding (BF16) | 1.2 |
| Shared experts (FP8 block) | 1.0 |
| LM head (MXFP8) | 0.6 |
| Dense MLP (layers 0–2) | 0.5 |
| Vision | 0.4 |
| Other | 0.1 |
| **Total** | **165.2** |

Without CSF, the decoder expert scales would be 17.7 GiB instead of 9.6 GiB: **8.1 GiB more**.

## 7. Where the code lives in vLLM

| Path | What |
|---|---|
| `vllm/models/glm5next/common/model.py` | the model; `load_weights` at lines 859 and 1135 |
| `vllm/models/glm5next/common/kda.py` | KDA; FlashKDA "requires CUDA SM90/SM10x/SM12x, bfloat16, head_dim=128" (around line 141–157), otherwise a Triton path |
| `vllm/models/glm5next/common/attention.py` | DSA attention; uses `Glm5NextIndexerBackend` |
| `vllm/models/glm5next/nvidia/sparse_indexer.py` | indexer; uses DeepGEMM `fp8_fp4_paged_mqa_logits` |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py:204` | FlashMLA-sparse: `capability.major in [9, 10]`, so Hopper is supported |
| `vllm/models/glm5next/common/mtp.py` | MTP layer |

## Key takeaways

- GLM-5.3-Flash is about 320 B parameters. Its routed experts are 94 % of the FP4-Local file.
- NVIDIA's FP4-Local uses the **csf_32x2_v1** scale format. Nothing in upstream vLLM (nor PR #60646) reads it yet.
- Its shared experts and DSA projections are 128 × 128 block FP8, which Hopper runs natively and fast.
- Its experts use a SwiGLU clamp (`swiglu_limit = 10`). Both Marlin and Humming support it.

---

**Next:** [07 · Marlin and Humming kernels](07-Marlin-And-Humming-Kernels.md)
