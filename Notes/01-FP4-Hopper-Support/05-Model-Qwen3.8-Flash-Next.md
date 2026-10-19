# 05 · Model: Qwen3.8-Flash-Next

**What you'll learn:** how Qwen3.8-Flash-Next is built, layer by layer; the shape of every important tensor; where the bytes go in each checkpoint; and where the model's code lives in vLLM. All numbers come from the checkpoint `config.json` files and safetensors headers (✅) unless marked 📄.

---

## 1. The model in one paragraph

Qwen3.8-Flash-Next (Hugging Face architecture name `Qwen4ExpForConditionalGeneration`, vLLM model type `qwen4_exp`) is a multimodal Mixture-of-Experts model. It has 48 decoder layers. Three of every four layers use **Gated DeltaNet** (a "linear attention" that keeps a fixed-size state instead of a growing KV cache). Every fourth layer uses **Qwen Sparse Attention** (QSA), full attention that only looks at a selected subset of past tokens. Every layer has an MoE block with **512 routed experts** (10 used per token) plus one **shared expert**. Instead of a single residual stream, it carries **4 parallel streams** ("hyper-connections"). Layer 2 adds a huge **n-gram embedding** table (PLE). One extra **MTP** layer predicts the next-next token for speculative decoding. A vision encoder handles images and video.

📄 NVIDIA's model card: "125B in total and 6B activated, plus 51B n-gram embedding and 4B MTP". Context: 262K tokens natively, extensible to 1M.

## 2. Key config numbers (`text_config`)

| Field | Value | Meaning |
|---|---:|---|
| `hidden_size` | 2560 | width of one residual stream (H) |
| `num_hidden_layers` | 48 | decoder layers |
| `layer_types` | 36 × `linear_attention`, 12 × `full_attention` | every 4th layer (3, 7, …, 47) is full attention |
| `num_experts` | 512 | routed experts per layer (E) |
| `num_experts_per_tok` | 10 | experts used per token (top_k) |
| `moe_intermediate_size` | 640 | expert hidden width (I) |
| `shared_expert_intermediate_size` | 640 | shared expert width |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 | QSA attention shape |
| `linear_num_key_heads` / `linear_num_value_heads` / head dims | 16 / 48 / 128 | Gated DeltaNet shape |
| `hc_count` / `hc_lowrank` | 4 / 320 | 4 residual streams; mixers use rank 320 |
| `ple_layer_ids` | [2] | the n-gram embedding is injected at layer 2 |
| `ngram_size` / `heads_per_ngram` | 3 / 8 | 2-gram and 3-gram lookups, 8 heads each → 16 lookups per token |
| `split_ngram_parts` | 128 | the table is split into 128 shards in the checkpoint |
| `ple_embedding_dtype` | `nvfp4` (FP4-Local only) | the n-gram table itself is NVFP4 |
| `vocab_size` | 248 320 | |
| `mtp_num_hidden_layers` | 1 | one MTP draft layer (full attention) |
| `vision_config.depth` / `hidden_size` | 27 / 1152 | vision encoder |

## 3. One decoder layer, step by step

The layer gets 4 residual streams, each 2560 wide (10 240 numbers per token in total).

1. **Hyper-connection mixer (attention side).** Small BF16 matrices (`input_mix_weight_down [320, 10240]`, `input_mix_weight_up [10240, 320]`, `block_inject_weight [4, 10240]`) combine the 4 streams into one 2560-wide input and later inject the result back.
2. **Token mixing**, one of two kinds:
   - **Gated DeltaNet** (36 layers), tensors under `linear_attn.*`:
     - `in_proj_qkv [10240, 2560]`: queries and keys (16 heads × 128 each) and values (48 heads × 128).
     - `in_proj_z [6144, 2560]`: an output gate.
     - `in_proj_a [48, 2560]`, `in_proj_b [48, 2560]`: per-head decay and write strength.
     - `conv1d [10240, 1, 4]`, `A_log`, `dt_bias`, `norm`: small state-update parameters.
     - `out_proj [2560, 6144]`.
   - **Qwen Sparse Attention** (12 layers), tensors under `self_attn.*`:
     - `q_proj [12288, 2560]`: 24 heads × 256, ×2 because it also produces a sigmoid output gate.
     - `k_proj [512, 2560]`, `v_proj [512, 2560]`: 2 KV heads × 256.
     - `o_proj [2560, 6144]`.
     - `indexer.index_qk_proj [640, 2560]`: a small "indexer" that scores past tokens, so attention reads only the best 2048 (`indexer_budget`).
3. **Hyper-connection mixer (MLP side).** The same structure again.
4. **MoE block**, tensors under `mlp.*`:
   - `gate [512, 2560]` (BF16): the **router**. It scores the 512 experts. The top 10 are used.
   - 512 **routed experts**, each with `gate_proj [640, 2560]`, `up_proj [640, 2560]` and `down_proj [2560, 640]`.
   - 1 **shared expert** (always used), same shapes, plus `shared_expert_gate [1, 2560]`.

Layer 2 also has the **PLE** block (section 5).

## 4. The routed experts: where the bytes are

Every routed expert is a tiny SwiGLU MLP:

```
h   = silu(x · gate_projᵀ) * (x · up_projᵀ)     x: [tokens, 2560] → h: [tokens, 640]
out = h · down_projᵀ                              → [tokens, 2560]
```

Count: 48 layers × 512 experts × 3 matrices × (2560 × 640) = **120.8 billion weights**. That's most of the model. In FP4-Local each expert matrix is stored as:

| Tensor | dtype | Shape (gate/up) | Shape (down) |
|---|---|---|---|
| `…experts.{e}.gate_proj.weight` | U8 (2 FP4 per byte) | [640, 1280] | — |
| `…experts.{e}.down_proj.weight` | U8 | — | [2560, 320] |
| `….weight_scale.nvfp4_csf_fixed` | U8 | [40, 1296] | [160, 336] |
| `….weight_scale.nvfp4_csf_exceptions` | U32 | a few hundred to ~1000 entries | ~200 entries |
| `….weight_scale_2` | F32 | scalar | scalar |
| `….input_scale` | F32 | scalar | scalar |

The uncompressed scale would be E4M3 `[640, 160]` for gate/up and `[2560, 40]` for down. Note 11 explains the CSF tensors.

## 5. PLE: the 51-billion-parameter n-gram table

At layer 2, the model looks up extra vectors keyed by the last 2 and 3 tokens:

1. For every token, compute 16 hash IDs (2-gram and 3-gram, 8 "heads" each).
2. Look up 16 vectors of 160 numbers each (16 × 160 = 2560) in a giant table.
3. Mix them in with `ple.key_proj [10240, 2560]`, `ple.value_proj [2560, 2560]`, a short convolution and norms.

The table has about 320 million rows × 160 = **51.2 billion numbers**, split into 128 shards in the checkpoint:

| Checkpoint | Table format | Table size |
|---|---|---:|
| vanilla NVFP4 (`nvidia/Qwen3.8-Flash-Next-NVFP4`) | FP8 | 47.7 GiB |
| FP4-Local | **NVFP4** (`shard_i.weight` U8 [2 500 012, 80] + `shard_i.weight_scale` E4M3 [2 500 012, 10]) | **26.9 GiB** |
| GGUF GSQ-RCO | IQ4_NL (shard 2 of each GGUF) | 28.8 GB |

Important facts:

- ✅ It's a **lookup, not a matrix multiply.** Each token reads only 16 rows. So it doesn't need to be in GPU memory at all.
- ✅ vLLM keeps it in **pinned CPU memory by default**. `EngramConfig.cpu_offload` defaults to `True` (`vllm/config/engram.py`), and the GPU reads the 16 rows over PCIe using unified addressing. You can change this with `--engram-config '{"cpu_offload": false}'`.
- ✅ The NVFP4 lookup-and-unpack is a **Triton kernel** (`_lookup_nvfp4_ple_embedding_kernel`, `vllm/models/qwen4_exp/common/ngram_embedding.py:521`). Triton runs on Hopper. So PLE isn't a Hopper problem.

## 6. MTP and vision

- **MTP layer** (`mtp.layers.0.*`): one full-attention layer with its own 512 experts. In FP4-Local its experts are `W4A16_NVFP4` (FP4 weights, BF16 activations, scales **not** CSF-compressed). It's used for speculative decoding: it guesses the next tokens, and the main model checks them.
- **Vision encoder** (`model.visual.*`): 27 blocks. In FP4-Local the attention linears are MXFP8 and the MLP linears are `W4A16_NVFP4`.

## 7. What each checkpoint stores, by component (GiB, ✅ from headers)

| Component | BF16 base | Vanilla NVFP4 | **FP4-Local** |
|---|---:|---:|---:|
| Routed experts (48 layers) | ≈ 225 | 63.3 (FP4 56.25 + scales 7.03) | **59.9** (FP4 56.25 + CSF 3.69) |
| PLE n-gram table | ≈ 95 | 47.7 (FP8) | **26.9** (NVFP4) |
| Attention + Gated DeltaNet (+ attention-side mixers) | | 5.6 (BF16) | **3.2** (MXFP8 + BF16 parts) |
| MTP experts | | 2.3 (FP8 block) | **1.3** (W4A16 NVFP4) |
| Embedding + LM head | | 2.4 (BF16) | **2.4** (BF16) |
| Shared experts | | 0.44 (BF16) | **0.23** (MXFP8) |
| Vision | | 0.84 | **0.31** |
| Everything else (norms, routers, MLP-side mixers, MTP attention) | | 0.89 | 0.89 |
| **Total** | **335.3** | **123.5** | **95.1** |

## 8. The format of every layer in FP4-Local (✅ `quantized_layers` in `config.json`)

| Format | Count | Which tensors |
|---|---:|---|
| `NVFP4` + `weight_scale_encoding: "csf"` | 48 | `model.language_model.layers.N.mlp.experts` (all routed experts) |
| `MXFP8` (group 32) | 438 | shared experts (48 × 3), Gated DeltaNet projections (36 × 5), QSA projections and indexer (12 × 5), vision attention (27 × 2) |
| `NVFP4` (group 16) | 1 | the PLE n-gram embedding |
| `W4A16_NVFP4` | 58 | vision MLP (27 × 2), vision merger (2), MTP experts (2 entries) |
| not listed | — | embedding, LM head, routers, norms, hyper-connection mixers, conv weights: BF16 or FP32 |

## 9. Where the code lives in vLLM

The model has moved to a new layout under `vllm/models/`:

| Path | What |
|---|---|
| `vllm/models/qwen4_exp/nvidia/model.py` | the model for NVIDIA GPUs (`load_weights` at line 642 uses `AutoWeightsLoader`) |
| `vllm/models/qwen4_exp/nvidia/qsa.py`, `nvidia/ops/qsa.py` | sparse attention; `_select_sm90_config` at `ops/qsa.py:558` is tuned for Hopper (H20) |
| `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` | small decode GEMMs; has `QWEN4_EXP_SM90_GEMM_PLANS` (line 72) |
| `vllm/models/qwen4_exp/nvidia/hyperconnection.py`, `ops/hc.py`, `ops/cute_dsl/` | hyper-connection kernels (CuTe DSL needs sm_90+) |
| `vllm/models/qwen4_exp/common/ngram_embedding.py` | PLE table, including the NVFP4 lookup kernel |
| `vllm/models/qwen4_exp/nvidia/mtp.py` | MTP draft layer |
| `vllm/config/engram.py` | PLE storage options (`cpu_offload`, `dp_shared_memory`, `use_thp`) |

## Key takeaways

- 63 % of FP4-Local's bytes are routed experts. They're the only W4A4 NVFP4 part, and the only part with CSF scales.
- The 27 GiB PLE table is a lookup. vLLM keeps it in CPU memory by default, so it costs no VRAM.
- The model's own code already has Hopper (sm_90) paths. The quantized layers are the gap.

---

**Next:** [06 · Model: GLM-5.3-Flash](06-Model-GLM-5.3-Flash.md)
