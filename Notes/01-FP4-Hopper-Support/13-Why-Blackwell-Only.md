# 13 · Why Blackwell only

**What you'll learn:** the three different reasons the FP4-Local model cards say "NVIDIA Blackwell"; which of them is a true hardware limit and which are only missing software; and, layer by layer, what Hopper can and can't do with these checkpoints.

---

## 1. What the cards say

- "Supported Hardware Microarchitecture Compatibility: NVIDIA Blackwell." 📄
- "Upstream vLLM support for this format has not yet landed. You must use one of the containers provided below." 📄
- Validated on RTX PRO 6000 Blackwell and DGX Spark only. 📄

"Blackwell only" mixes three different things.

## 2. Reason 1: the software exists only in a Blackwell-only stack (not a hardware limit)

- **CSF readers** are only in the Local Inference Lab fork of vLLM. Upstream has none (note 11).
- The containers use **b12x** kernels for experts, linears, attention and linear attention (`--moe-backend b12x --linear-backend b12x --attention-backend B12X --gdn-decode-kernel b12x`). b12x is written in CuTe DSL for **sm_120 / sm_121 only** (note 14).
- The containers are built for those chips: `TORCH_CUDA_ARCH_LIST=12.0a`, `CUTE_DSL_ARCH=sm_120a` (or `sm_121a`).

So even a B200 (sm_100) isn't covered by the community containers. ❓ (Not tested.)

**This is all software.** Upstream vLLM already has Hopper kernels for every number format in these checkpoints. What's missing is the loader, plus testing.

## 3. Reason 2: FP4 math needs FP4 tensor cores (a real hardware limit)

The checkpoints declare the routed experts as **W4A4**: FP4 weights *and* FP4 activations, multiplied by the tensor cores directly, with the hardware applying one scale per 16 values on each side.

| Feature | Hopper (sm_90) | Blackwell (sm_100 / sm_120) |
|---|---|---|
| FP4 × FP4 tensor-core multiply | ❌ | ✅ |
| Hardware block scales (one per 16 or 32 values, applied inside the MMA) | ❌ | ✅ |
| FP8 × FP8 multiply | ✅ (`wgmma`, K = 32) | ✅ |
| BF16 × BF16 multiply | ✅ | ✅ |

So **W4A4 is impossible on Hopper.** That's the only true hardware limit in this whole project.

But it doesn't matter much, because:

- vLLM can **ignore `input_scale`** and keep activations in BF16 (W4A16). The weights stay 4-bit in memory; the kernel unpacks them in registers. This is exact (note 02, section 7).
- Even on Blackwell, the recommended community setting runs the experts as W4A16 for decode (`VLLM_B12X_MOE_FP4_FORCE_A16=1`). FP4 activations are only used for long prefills (`B12X_W4A16_A4_PREFILL=1`).

## 4. Reason 3: nobody has tested it on Hopper (a validation gap)

No one has published Hopper runs of FP4-Local. Bugs may exist in Hopper-only code paths: Humming's sm_90 kernels, Marlin MoE on Hopper (see the open issues in note 07), FlashKDA on sm_90, QSA sm_90 tables. That's Phase B.

## 5. Layer by layer: what Hopper can do

| Layer type in FP4-Local | What it needs | Hopper can… | Best Hopper path today |
|---|---|---|---|
| NVFP4 routed experts (W4A4) | FP4 weights, FP4 activations | ✅ W4A16 exact; ✅ W4A8 with rounding (Phase D); ❌ W4A4 | Humming or Marlin MoE, W4A16 |
| `W4A16_NVFP4` (MTP experts, vision MLP) | FP4 weights, BF16 activations | ✅ exact | Humming or Marlin, W4A16 |
| NVFP4 PLE table (Qwen) | lookup + unpack | ✅ | Triton lookup kernel, CPU-resident table |
| MXFP8 linears (attention, shared experts, dense MLP, LM head) | FP8 weights + E8M0 scale / 32 | ✅ W8A16 | Humming W8A16 (default on sm_90) or Marlin |
| FP8 128 × 128 block (GLM shared experts, DSA projections) | FP8 weights, FP8 activations per 128 | ✅ native W8A8 | DeepGEMM / CUTLASS block FP8 |
| BF16 parts | BF16 | ✅ | normal kernels |
| CSF scale streams | a decoder | ✅ (any GPU; it's just bytes) | decode at load (Phase A), later on GPU (Phase C) |

Why MXFP8 runs as W8A16 on Hopper by default: Humming's input-schema code prefers 16-bit activations for group-scaled weights, and it only treats "MXFP8 with FP8 activations" as fine on sm_120 and newer (`humming/schema.py:342-354`). W8A8 for these layers would need a separate kernel decision. It's not part of this project.

## 6. What changes, and what stays the same, between Blackwell and Hopper

| | Blackwell (community stack) | Hopper (this project) |
|---|---|---|
| Expert weights in VRAM | FP4 | FP4 (same) |
| Expert scales in VRAM | CSF compressed | raw after Phase A, CSF after Phase C |
| Decode math | W4A16 | W4A16 (same) |
| Prefill math | W4A4 (FP4 activations) | W4A16, or W4A8 after Phase D |
| Accuracy | as trained | W4A16 identical in math to Blackwell's W4A16 decode; W4A8 adds a small rounding error (note 17) |

## Key takeaways

- Only one thing is truly impossible on Hopper: FP4 × FP4 math (W4A4).
- Everything else is software (CSF loader, testing) or speed (W4A8 prefill).
- Hopper W4A16 does the same math as Blackwell's recommended decode mode, and keeps 4-bit weights in memory.

---

**Next:** [14 · b12x explained](14-b12x-Explained.md)
