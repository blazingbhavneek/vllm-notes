# 15 · How vLLM loads and runs these checkpoints

**What you'll learn:** the exact path a ModelOpt checkpoint takes through vLLM, from `vllm serve` to the first forward pass, with file and line numbers. You need this map to know *where* each Hopper change goes. All references are to vLLM `main` at `3e96f9cf63` (19 Oct 2026). Line numbers drift; search for the function name if a number is off.

---

## The seven steps at a glance

```
1. vllm serve           → ModelConfig picks the quantization config class
2. build the model      → every layer asks the config: "which quant method am I?"
3. create_weights       → each method registers empty parameters (names + shapes)
4. find + read files    → DefaultModelLoader globs safetensors and streams (name, tensor)
5. load_weights         → each tensor is copied into its parameter (TP slicing here)
6. post-load processing → pick the kernel, repack weights/scales for it
7. forward              → the kernel runs; weights stay packed in VRAM
```

CSF has to be handled in step 4, 5 or 6 (section 8).

## Step 1: choosing the quantization config

- `config.json` has `quantization_config.quant_method = "modelopt"` and `quant_algo = "MIXED_PRECISION"` (FP4-Local, vanilla Qwen) or `"NVFP4"` (vanilla GLM).
- `ModelOptMixedPrecisionConfig.override_quantization_method` (`vllm/model_executor/layers/quantization/modelopt.py:1646`) returns `"modelopt_mixed"` when the algo is `MIXED_PRECISION`. The list of valid names is in `vllm/config/model.py:1396`.
- vLLM checks `get_min_capability()`: **75** for `modelopt_mixed` (`modelopt.py:1635`) and for NVFP4 (`:771`), 80 for MXFP8 (`:1219`). An H200 (90) passes.
- The mixed config reads `quantized_layers` (the per-layer recipes) and builds one sub-config per algo: `fp8_config`, `nvfp4_config`, `w4a16_nvfp4_config`, `mxfp8_config`, and a block-FP8 config (`modelopt.py:1562-1615`).

⚠️ Nothing in upstream reads `weight_scale_encoding`. A CSF recipe looks like a plain NVFP4 recipe to vLLM today.

## Step 2: each layer gets a quant method

When the model is built, every `LinearBase` and `RoutedExperts` module calls `quant_config.get_quant_method(layer, prefix)` (`modelopt.py:1832`):

1. Attention modules get the KV-cache method if `kv_cache_quant_algo` is set.
2. Excluded layers get `UnquantizedLinearMethod`.
3. `_resolve_quant_algo(prefix)` (`:1736`) finds the recipe: direct name, then fused names via `packed_modules_mapping` (`gate_up_proj` → `gate_proj` + `up_proj`), then "any child starts with the prefix" for experts.
4. **Linear layers** → `build_linear_method` (`:2756`) → the generic `ModelOptLinearMethod` (`:2542`). `resolve(algo)` (`:2689`) turns the algo into a (weight key, activation key) pair:

| Algo | Weight key | Activation key |
|---|---|---|
| `FP8_PB_WO` | `kFp8Static128BlockSym` | `kFp8Dynamic128Sym` |
| `MXFP8` | `kMxfp8Static` | `kMxfp8Dynamic` |
| `NVFP4` | `kNvfp4Static` | `kNvfp4Dynamic` |
| `W4A16_NVFP4` | `kNvfp4Static` | none (`input_scale` dropped) |

5. **Routed experts** → `FP8_PB_WO` → `Fp8MoEMethod`; `NVFP4` or `W4A16_NVFP4` → `ModelOptNvFp4FusedMoE` (`:839`); `MXFP8` → `ModelOptMxFp8FusedMoE` (`:1280`).

In `ModelOptNvFp4FusedMoE.__init__`, `use_a16` is true only for `W4A16_NVFP4` (`:856`), and the MoE backend is chosen right away: `select_nvfp4_moe_backend(...)` (`:873`, defined at `fused_moe/oracle/nvfp4.py:191`; walk-through in note 08).

## Step 3: create_weights registers empty parameters

`ModelOptNvFp4FusedMoE.create_weights` (`modelopt.py:911`) registers, per layer (E experts, H hidden, I intermediate):

| Parameter | Shape | Dtype |
|---|---|---|
| `w13_weight` | [E, 2I, H/2] | uint8 (FP4 pairs) |
| `w2_weight` | [E, H, I/2] | uint8 |
| `w13_weight_scale` | [E, 2I, H/16] | float8_e4m3fn |
| `w2_weight_scale` | [E, H, I/16] | float8_e4m3fn |
| `w13_weight_scale_2`, `w2_weight_scale_2` | [E, 2] / [E] | float32 |
| `w13_input_scale`, `w2_input_scale` | per expert | float32 |

There is **no parameter** for `…weight_scale.nvfp4_csf_fixed`, `…csf32_codes`, and so on. That's the core of the loading problem.

## Step 4: finding and reading the files

`DefaultModelLoader` (`vllm/model_executor/model_loader/default_loader.py`):

1. `_prepare_weights` (`:147`): download (or find) the snapshot, then `glob.glob(os.path.join(hf_folder, "*.safetensors"))` (`:229-230`). **Only the top folder is searched.**
2. `filter_duplicate_safetensors_files` (`:252`) keeps only files named in `model.safetensors.index.json`.
3. `_get_weights_iterator` (`:265`) streams `(name, tensor)` pairs from each file.

Consequences:

- **Qwen FP4-Local** keeps every shard under `tensors/` → the glob finds **nothing** → load fails ("Cannot find any model weights"). ⚠️
- **NVIDIA GLM FP4-Local** has top-level `model-*.safetensors` *and* `tensors/native-QB.safetensors` → the glob finds the top-level files, **silently misses** `native-QB` (the BF16 KDA small projections, the DSA indexer and `kv_b_proj`) → those parameters stay uninitialized. ⚠️
- PR #60646 adds a fallback that reads shard paths from the index, but **only when the glob found nothing** (`if not hf_weights_files`). That fixes Qwen but **not GLM**. The fix needs to always add index-listed files. ⚠️ (From reading the diff.)

## Step 5: load_weights copies tensors into parameters

- `Qwen4ExpForCausalLM.load_weights` (`vllm/models/qwen4_exp/nvidia/model.py:642`) renames a few things, then uses `AutoWeightsLoader`, which walks the module tree and calls each parameter's `weight_loader`.
- For experts, the `RoutedExperts` weight loader maps `experts.{e}.gate_proj.weight_scale` into slice `e`, shard 0 of `w13_weight_scale`, applying tensor-parallel slicing along the intermediate dimension.
- Qwen's loader ignores names ending in `_weight_scale` and `_input_scale` when they have no matching parameter (`_QWEN4_EXP_IGNORED_MISSING_SUFFIXES`, `model.py:143`), but **not** `.nvfp4_csf_fixed` or `.csf32_*`. So the CSF streams most likely raise "There is no module or parameter named …". ⚠️ Confirm on a real run.
- GLM: `vllm/models/glm5next/common/model.py:859` and `:1135`.

## Step 6: post-load processing picks and prepares the kernel

### Routed experts (NVFP4)

`ModelOptNvFp4FusedMoE.process_weights_after_loading` (`modelopt.py:1027`):

1. If `use_a16`, fill input scales with 1.0.
2. `convert_to_nvfp4_moe_kernel_format` (`fused_moe/oracle/nvfp4.py:324`) **repacks** for the chosen backend:
   - **HUMMING** (`:425`): `convert_to_humming_moe_kernel_format(layer, {"quant_method": "modelopt", "quant_algo": "nvfp4"})` replaces the weight and scale parameters with Humming's own layout. Input scales are set to `None`.
   - **MARLIN** (`:460`): `prepare_nvfp4_moe_layer_for_marlin` (`quantization/utils/marlin_utils_fp4.py:329`) repacks weights and converts the scales to S0E5M3 (note 07). Input scales `None`.
3. `make_nvfp4_moe_quant_config` (`:526`) and `make_nvfp4_moe_kernel` (`:610`) build the runnable kernel object.

**Key fact for Phase C:** after this step, the scales in VRAM are **no longer in checkpoint layout**. They're permuted and re-encoded for Humming or Marlin. A runtime CSF decoder must write the kernel's layout, or the kernel must read raw layout. Note 18 deals with this.

### Linear layers (MXFP8, NVFP4, FP8 block)

`ModelOptLinearMethod.process_weights_after_loading` (`modelopt.py:2625`) → `select_linear_kernel` (`:2404`) →

- MXFP8: `init_mxfp8_linear_kernel` (`kernels/linear/__init__.py:947`), candidates `_POSSIBLE_MXFP8_KERNELS` (`:560`).
- NVFP4: `init_nvfp4_linear_kernel` (`:1139`), candidates `_POSSIBLE_NVFP4_KERNELS` (`:584`).
- In both, `prioritize_humming(kernels, compute_capability)` (`:419`) moves Humming ahead of Marlin **on sm_90 only** (`quantization/utils/humming/priority.py:11`).
- FP8 block: the block-FP8 W8A8 kernels (DeepGEMM / CUTLASS on Hopper).

You can override the choice with `--linear-backend marlin` and `--moe-backend marlin` (or `humming`).

## Step 7: forward

The MoE layer calls the kernel object with the router output; the kernel reads packed FP4 weights and scales and unpacks them in registers (note 07). Linear layers call their kernel's `apply`. Nothing is unpacked into VRAM.

## 8. Where CSF can be handled: three options

| Option | Where | What it does | Pros | Cons |
|---|---|---|---|---|
| **A. Loader-level decode** | step 4, wrap the weight iterator (PR #60646) | turn CSF streams into a normal `weight_scale` tensor on the CPU before any model code sees it | model-agnostic; zero changes to quant methods or kernels; tiny diff | VRAM holds raw scales (+3.3 / +8.1 GiB); a bit of CPU time at load |
| **B. Quant-method-level** | steps 3 and 6 (LIL fork's approach) | register CSF parameters in `create_weights`, decode in `process_weights_after_loading` | can keep the compressed form for later | touches every NVFP4 MoE method; per-backend work |
| **C. Runtime decode** | step 7 | keep CSF in VRAM, decode into scratch per layer before the kernel | saves the 3.3 / 8.1 GiB | needs a GPU decoder that writes the kernel's layout; must work with CUDA graphs |

The plan (note 18): **A first** (Phase A), then **C** (Phase C), reusing A's decoder as the test reference.

## Key takeaways

- The quant config, the loader and the kernel oracles are separate pieces; CSF breaks the loader and the parameter names, not the kernels.
- Two loader bugs block FP4-Local: shards in `tensors/` (both models) and an extra subfolder shard next to top-level shards (GLM).
- After loading, Humming and Marlin rewrite the scales into their own layouts. Runtime CSF decoding must target those layouts.

---

**Next:** [16 · What happens on Hopper today](16-What-Happens-On-Hopper-Today.md)
