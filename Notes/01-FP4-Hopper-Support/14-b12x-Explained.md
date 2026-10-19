# 14 · b12x explained

**What you'll learn:** what b12x is, where it lives now, what it does for FP4-Local on Blackwell (MoE, CSF decode, W4A8), why none of its kernels can run on Hopper, and which of its *ideas* we should copy for Hopper.

---

## 1. What b12x is

**b12x** is a kernel library for NVIDIA **sm_120 / sm_121** GPUs (RTX 5090, RTX PRO 6000 Blackwell, DGX Spark), written in **CuTe DSL** (CUTLASS's Python kernel language). It's made by the Local Inference Lab community and powers the FP4-Local containers.

| Fact | Value |
|---|---|
| Standalone repo | archived 6 Oct 2026 at `2cc7f66` ("replace README with FlashInfer migration guide") |
| New home | FlashInfer `flashinfer/experimental/b12x`, merged in [FlashInfer #5767](https://github.com/flashinfer-ai/flashinfer/pull/5767), commit `fc8fdc17` |
| Install | `flashinfer-python[b12x]` (Python imports stay `import b12x`) |
| Requires (at the merge) | PyTorch 2.12+, CUTLASS DSL 4.7.1 |
| In upstream vLLM | `uv pip install "vllm[b12x]"`, `--linear-backend b12x`, `--moe-backend b12x` (`docs/features/quantization/b12x.md`: "NVIDIA SM120 and SM121 GPUs") |

What it contains (folders under `b12x/`): `moe/` (fused MoE, EP MoE, calibration, checkpoints), `gemm/`, `attention/` (MLA, sparse, paged), `sequence/` (linear-attention kernels), `norm/`, `comm/pcie/` (all-reduce over PCIe without NVLink), `loader/`, `preparation/`, `quantization/`.

## 2. How a b12x MoE call works

Every b12x MoE kernel follows the same logical steps (`docs/moe-execution-model.md`):

1. Route each token to `top_k` experts.
2. Compute expert-local FC1 (gate + up).
3. Apply the activation (and quantize the intermediate if needed).
4. Compute FC2 (down).
5. Reduce weighted routes back into token order.

The numeric recipes it supports:

| Recipe | Activations | Weight scale in the file | Scale used during compute |
|---|---|---|---|
| NVFP4 W4A4 | FP4 E2M1, E4M3 per K/16 | E4M3 per K/16 | E4M3 per K/16 |
| **W4A8 on NVFP4** | MXFP8 (E4M3, E8M0 per K/32) | E4M3 per K/16 | **E8M0 per K/32 base × E4M3 per K/16 residual** |
| native W4A8-MX | MXFP8 | E8M0 per K/32 | E8M0 per K/32 |
| W8A8-MX, W6A8-MX | MXFP8 | E8M0 per K/32 | E8M0 per K/32 |
| W4A16 | BF16 | E4M3 per K/16 or E8M0 per K/32 | source-preserving |

## 3. How b12x handles CSF (keeping scales small in VRAM)

From `docs/lossless-scale-compression.md` and `b12x/_lib/quant/nvfp4_csf.py`:

1. **vLLM reads the checkpoint.** The LIL fork's `nvfp4_csf_loader` (and `mxfp4_csf_loader`) read the CSF streams and slice them per tensor-parallel rank. b12x itself "has no CSF checkpoint reader".
2. **Preparation (once, before CUDA-graph capture).** vLLM passes `CsfScalePlanes(fixed, exceptions)` per expert. `prepare_weights` uploads the **compressed** planes to the GPU and pre-splits the exception ranges so each thread block knows which exceptions are its own.
3. **Scratch.** The caller provides a decoded-scale scratch buffer. Layers that run one after another can **share one scratch**.
4. **Per MoE call**, `Nvfp4CsfDecoder.decode(ids, …)` (`nvfp4_csf.py:841`) expands scales into the scratch, choosing by batch size:

| Number of routed (token, expert) pairs | Mode | What it does |
|---|---|---|
| fewer than 64 | 0 | decode only the routed experts, skipping duplicates directly |
| 64 or more | 2 | first mark which experts are used (`mark_active_experts`), then decode those |
| at least 16 × number of experts | 3 | long prefill: decode **all** experts |

5. `expand_scales` (`moe/fused_moe/api.py:176`) can decode the **next** layer's scales on a side stream while the current layer computes.

Gate/up and down scales are decoded together in one launch ("paired decoder").

Result: VRAM holds the compressed streams (3.7 GiB for Qwen instead of 7.0) plus one small scratch.

## 4. The "W4A8 on NVFP4" recipe (important for Hopper Phase D)

Source: `b12x/moe/_shared/kernels/reference.py:1046` (`decompose_nvfp4_scales_to_mx_residual`) and `:1154` (`_dequant_w4a8_weight_nvfp4_residual`).

**At load time**, for each pair of neighboring E4M3 scales `(s0, s1)` (together covering K = 32):

1. `E = max(floor(log2 s0), floor(log2 s1))`, the shared power of two, stored as UE8M0.
2. Residuals `r0 = s0 / 2^E`, `r1 = s1 / 2^E`, stored as E4M3. The larger one is in [1, 2).

**In the kernel**, for each FP4 weight `q` in a 16-group with residual `r`:

1. Decode the nibble to FP16 (exact).
2. Multiply by `r` in FP16 (exact: FP16 has 11 significant bits; the product needs at most 6).
3. **Round to E4M3** (round to nearest even). ← the only lossy step
4. Feed that E4M3 value to the tensor core. The **hardware** block-scaled MMA multiplies by `2^E` for each K/32.

The code warns that "the partner residual can underflow E4M3 when the pair's exponents differ by more than ~2^9", and provides `nvfp4_mx_residual_quality_report` to measure it. Its docstring says: "Run on real checkpoints before claiming full-fidelity NVFP4 serving."

On Hopper there's no block-scaled MMA to apply `2^E` per K/32. Note 17 shows how to move `E` to the epilogue instead.

## 5. Other b12x facts worth knowing

- **Recommended FP4-Local settings:** `VLLM_B12X_MOE_FP4_FORCE_A16=1` (W4A16 experts) plus `B12X_W4A16_A4_PREFILL=1` (FP4 activations only for prefill).
- **Prefill scratch:** `docs/w4a16-prefill-reduction.md` cuts W4A16 prefill scratch from 1,071,913,568 bytes to 308,550,240 bytes by fusing the route sum.
- **Disk-backed tables:** `B12X_DISK_BACKEND=io_uring` and `VLLM_PLE_TABLE_MEMORY=disk` keep Qwen's PLE table on SSD (DGX Spark recipe).
- **Constraints:** the NVFP4 CSF reader supports GLM-5.3-Flash and Qwen3.8-Flash-Next shapes, and "TP extents must contain a multiple of 64 intermediate channels". b12x MoE has no expert parallelism.

## 6. Why b12x can't run on Hopper

- Its kernels use sm_120 instructions: block-scaled `mma.sync` with FP4 and FP8 operands and hardware scale vectors. Hopper doesn't have them.
- It's compiled for `sm_120a` / `sm_121a` (`CUTE_DSL_ARCH`). The "a" means architecture-specific features that exist only on that chip family.
- Its tuning (tile sizes, shared-memory budgets) targets the ~100 KB shared memory of sm_120, not Hopper's 228 KB with TMA and `wgmma`.

Porting b12x to Hopper would mean rewriting every kernel. **Don't.** vLLM already has Hopper kernels (Humming, Marlin, DeepGEMM, CUTLASS). Use those.

## 7. What to copy from b12x (ideas, not code)

| b12x idea | Use on Hopper |
|---|---|
| Keep CSF compressed in VRAM; decode into shared scratch per layer | Phase C |
| Decode only routed experts for small batches; all experts for long prefill | Phase C (start with "decode all" per layer; it's simpler and still saves memory) |
| Pre-split exception ranges per thread block at load | Phase C GPU decoder |
| Power-of-two base × E4M3 residual, rounded once to E4M3 in registers | Phase D (with the exponent moved to the epilogue) |
| `nvfp4_mx_residual_quality_report` | Phase D: measure the rounding on real checkpoints before claiming accuracy |

License: b12x is Apache-2.0 (check `LICENSE` in the FlashInfer tree before copying any code; reimplementing an idea needs no license).

---

**Next:** [15 · How vLLM loads and runs quantized models](15-How-vLLM-Loads-And-Runs-Quantized-Models.md)
