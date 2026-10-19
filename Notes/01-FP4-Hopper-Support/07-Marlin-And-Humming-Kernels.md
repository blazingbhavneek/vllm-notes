# 07 · Marlin and Humming kernels

**What you'll learn:** what Marlin is and how its "vanilla" 4-bit mode works (GPTQ / AWQ INT4); the exact bit tricks it uses to unpack NVFP4 to BF16; why its FP8 mode refuses Hopper; what Humming is and why vLLM prefers it on Hopper; and the one Humming rule that blocks NVFP4 × FP8 today. These two kernels are what will run FP4-Local on Hopper.

---

## 1. Marlin in one paragraph

**Marlin** ("Mixed Auto-Regressive LINear kernel") was written by IST-DASLab (Elias Frantar, Dan Alistarh and others) for 4-bit-weight × 16-bit-activation matrix multiplies. Its goal: be as fast as the memory allows. At small batch sizes it reaches close to the ideal 4× speedup over BF16, because it reads 4× fewer bytes. vLLM adopted it and grew it to many formats. Today vLLM's Marlin lives in `csrc/quantization/marlin/` (CUDA) and `vllm/model_executor/layers/quantization/utils/marlin_utils*.py` (Python).

## 2. The "vanilla" 4-bit quants Marlin was built for: GPTQ and AWQ

Most 4-bit models on Hugging Face are **INT4 with one scale per 128 weights** ("group size 128"), made by one of two PTQ methods:

- **GPTQ:** quantizes one column at a time and adjusts the not-yet-quantized columns to cancel the error, using second-order information (the "Hessian") from calibration data. Stored as symmetric INT4 (`uint4b8`: values 0–15 meaning −8 to 7) plus BF16 / FP16 group scales.
- **AWQ:** finds the input channels with large activations and scales their weights up before rounding, so they lose less precision. Stored as unsigned INT4 plus scales **and zero points** (`uint4` + zeros).

Marlin runs both as W4A16. `w = (q − zero) × scale`, unpacked in registers and multiplied in FP16 or BF16.

Neither FP4-Local nor the vanilla NVIDIA checkpoints use INT4. But NVFP4 runs through the same machinery, with a different unpack step.

## 3. How a Marlin GEMM works, step by step

1. **Repack at load time.** `ops.gptq_marlin_repack` rearranges the packed weights so that each thread's single 32-bit load contains exactly the 8 values it needs for its tensor-core instruction. Scales are permuted to match (`marlin_permute_scales`). This happens once, in `process_weights_after_loading`.
2. **Partition the work.** The output is cut into tiles. The K dimension is split across SMs in "stripes" when there aren't enough tiles. Partial results are summed with a global reduction (`use_atomic_add` or `use_fp32_reduce`).
3. **Pipeline the loads.** Each thread block copies the next weight and activation tiles into shared memory with `cp.async` while computing the current ones (several stages deep).
4. **Unpack in registers.** A few bit operations turn 8 packed values into 8 BF16 (or FP16) values. Then they're multiplied by the group scale.
5. **Multiply.** `mma.sync` (the warp-level tensor-core instruction, K = 16 for BF16) accumulates into FP32.
6. **Epilogue.** Apply the global scale (NVFP4) and write BF16 output.

Note: Marlin uses `mma.sync`, not Hopper's `wgmma`. That's fine for memory-bound decode. For compute-bound prefill on Hopper, `wgmma` kernels are faster.

## 4. Marlin's NVFP4 unpack, bit by bit

From `csrc/quantization/marlin/dequant.h:434-449` (FP4 → BF16):

```cpp
constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;   // 6
constexpr int MASK = 0x70007000;
int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);  // two values at once
q <<= 4;
int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);  // the next two
```

What it does, for one 16-bit half of the register:

1. The FP4 value sits in bits 15–12 as `s e e m`.
2. `q & 0x8000` keeps the sign bit where BF16 wants it (bit 15).
3. `(q & 0x7000) >> 6` moves `e e m` to bits 8–6. In BF16, bits 8 and 7 are the two lowest exponent bits and bit 6 is the top mantissa bit.
4. The result is a valid BF16 number equal to **FP4 value × 2⁻¹²⁶**. That factor comes from the different exponent biases (BF16 bias 127 minus FP4 bias 1).

Instead of multiplying every value by 2¹²⁶, Marlin folds that factor into the **global scale** once: `nvfp4_marlin_process_global_scale` multiplies it by `2^(126 − 7)` (`marlin_utils_fp4.py:143-155`). The `− 7` matches a trick on the scales, described next. Total cost per value: a couple of bit operations. No lookup table, no float conversion.

**Scales.** `nvfp4_marlin_process_scales` (`marlin_utils_fp4.py:59-121`) turns each E4M3 block scale into a special 8-bit format called **S0E5M3** (no sign, 5 exponent bits, 3 mantissa bits). It multiplies the scale by 2⁷ and shifts its FP16 bit pattern. NVFP4 scales are never negative, so the sign bit can be reused. In the kernel this scale unpacks to BF16 with bit operations too. If a checkpoint's scales are all small, it also rescales them by a power of two (`_nvfp4_compute_scale_factor`) and corrects the global scale to match.

**Exactness.** FP4 × E4M3 needs 6 significant bits (note 02). BF16 has 8. So Marlin's W4A16 NVFP4 results are exact apart from FP32 accumulation order.

## 5. Marlin's FP8 path, and why it's off on Hopper

Marlin can also unpack weights to FP8 and multiply with FP8 activations (W4A8). `dequant.h:474-491` has the FP4 → FP8 E4M3 unpack. It's the same shift trick, with a shift of 2 instead of 6. It's exact.

Two things stop it from being used here:

1. **NVFP4 + FP8 activations is refused outright.** `marlin_utils_fp4.py` raises `"NVFP4 weight + INT8/FP8 activation is not supported."` in five places (lines 192, 225, 350, 455, 663). Only **MXFP4** (power-of-two scales) is allowed with FP8 activations. The power-of-two scale is applied exactly by adding 6 to its exponent (`mxfp4_marlin_process_scales`, lines 124-140). For NVFP4's E4M3 scales, that exactness is impossible (note 17).
2. **Marlin's FP8 mode refuses sm_90.** `get_marlin_input_dtype` (`marlin_utils.py:720-737`) raises unless the GPU is sm_89 or sm_12x: "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than Marlin W4A16 on other devices)". Marlin uses `mma.sync`, and on Hopper FP8 `mma.sync` isn't faster than BF16 `mma.sync`. You need `wgmma` to get Hopper's FP8 speed.

## 6. Marlin for MoE

`fused_marlin_moe` runs a whole MoE block with Marlin:

1. The router picks the top-k experts for each token.
2. `moe_align_block_size` groups tokens by expert and pads each group to the kernel's block size.
3. **GEMM 1:** a Marlin GEMM over all experts at once (`w13`, gate + up).
4. **Activation:** `silu(gate) * up`, with the SwiGLU clamp if the model has one (GLM: `swiglu_limit = 10`).
5. **GEMM 2:** a Marlin GEMM with `w2` (down).
6. Multiply each expert's output by its router weight and sum per token.

In vLLM it's `MarlinExperts` (`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`). It accepts any of these weight formats: FP8 (block, channel, tensor), MXFP4, MXFP8, NVFP4, INT4, INT8 (`_supports_quant_scheme`, line 596). It **ignores the activation format**, so a W4A4 NVFP4 checkpoint runs as W4A16, and `input_scale` is dropped (`oracle/nvfp4.py:460-475` sets `a13_scale = a2_scale = None`).

Known open bugs on Hopper (📄 from GitHub, still open on 19 Oct 2026): [#49070](https://github.com/vllm-project/vllm/issues/49070) (MiniMax-M3-NVFP4 garbage output on H100/H200; one fix merged as #48929; the unaligned-size fix [#48624](https://github.com/vllm-project/vllm/pull/48624) is still open) and [#48895](https://github.com/vllm-project/vllm/issues/48895) (wrong top-k weights in `moe_wna16_marlin_gemm`).

## 7. Humming: the successor

**Humming** ([vllm-project/humming](https://github.com/vllm-project/humming), Apache-2.0) is a JIT-compiled GEMM library for quantized inference, by jinzhen-lin, who also maintains vLLM's Marlin. vLLM pins `humming-kernels[cu13]==0.1.17` (`requirements/cuda.txt:31`).

What makes it different from Marlin:

- **One backend per GPU generation:** `mma` (sm_75–89), **`wgmma` on sm_90 with TMA and warp specialization**, `umma` (sm_100), `mxmma` (block-scaled, sm_120).
- **Many formats:** any weight type up to 8 bits; FP16, BF16, FP8, INT8 or FP4 activations, as long as the activation format can represent the weight values (README support matrix).
- **JIT:** it compiles a kernel for each layer shape and data type at startup, with heuristics per GPU (`humming/tune/sm90.py`, `sm90_h20.py`, `sm120.py`, …).
- **Hopper-specific layouts:** a "packed-K" weight layout for `wgmma`, and "fused E8M0 scale". That second one applies an MXFP4 power-of-two scale exactly during the FP4 → FP8 unpack, so MXFP4 × FP8 runs at FP8 speed on Hopper.

### How vLLM uses Humming

| Piece | File |
|---|---|
| Linear kernels | `vllm/model_executor/kernels/linear/nvfp4/humming.py`, `mxfp8/humming.py`, … (`is_supported`: CUDA, installed, sm_75+) |
| MoE kernels | `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py` (`BatchedHummingGroupedExperts`, `HummingGroupedExperts`, `HummingIndexedExperts`) |
| Format translation | `vllm/model_executor/layers/quantization/utils/humming/schema.py` (turns a ModelOpt or compressed-tensors description into Humming's config) |
| **Priority** | `vllm/model_executor/layers/quantization/utils/humming/priority.py`: `_HUMMING_PREFERRED_CAPABILITIES = frozenset({90})` (line 11). `prioritize_humming` moves Humming directly in front of Marlin, **on sm_90 only**. |

So on an H200, every NVFP4 or MXFP8 layer tries Humming before Marlin (note 15 shows the exact call sites).

### Which activation format Humming picks

`schema.py:320-400`. For a weight with group scales and an 8-bit activation request, with fallback allowed, it says (line 356):

```python
if input_bits == 8 and is_groupwise and not is_mxfp8:
    # prefer wna16 for groupwise weight or groupwise input
    return HummingInputSchema()          # = 16-bit activations
```

So by default, NVFP4 and MXFP8 layers on Hopper run as **W4A16 / W8A16** in Humming.

### The rule that blocks NVFP4 × FP8 on Hopper

`humming/kernel/humming.py:396-415` (`check_scale`): when the GPU has no block-scaled MMA (that includes sm_90),

```python
if self.weight_scale_group_size > 0:
    assert self.weight_scale_group_size >= 256 // self.a_dtype.num_bits
```

For FP8 activations, `256 // 8 = 32`. NVFP4's group size is 16. So **Humming can't run NVFP4 weights with FP8 activations on Hopper** unless the scale is fused into the unpack. Its only fused path (`use_fused_e8m0_scale`, `humming/config/config.py:368-376`) requires E8M0 (power-of-two) scales. The Hopper FP8 project (Phase D) is essentially "add a fused E4M3 scale path" to Humming or a similar kernel. Note 17 explains the math, note 18 the design.

## 8. The other Hopper kernel worth knowing: CUTLASS W4A8

`csrc/quantization/cutlass_w4a8/` and `vllm/model_executor/kernels/linear/mixed_precision/cutlass.py` implement **INT4 weights × FP8 activations on sm_90**, based on CUTLASS example 55 (`55_hopper_int4_fp8_gemm.cu`). Rules (`cutlass.py:31-66`): sm_90 only, FP8 E4M3 activations (dynamic per token), INT4 weights, **group size 128 only**, BF16 output. The group scales are converted to **FP8** (`convert_bf16_scales_to_fp8`, plus a per-channel scale applied in the epilogue) and applied to the unpacked weight in FP8 before the `wgmma`. So this kernel already rounds "weight × scale" into FP8, the same compromise Phase D needs. It's a useful starting point or reference.

## Key takeaways

- Marlin and Humming keep weights 4-bit in memory and unpack them in registers. That's W4A16, and it's exact for NVFP4.
- On Hopper, vLLM tries Humming first, then Marlin.
- Neither can do NVFP4 × FP8 on Hopper today. Marlin refuses NVFP4 + FP8 and refuses FP8 on sm_90. Humming needs weight groups of at least 32 for FP8, unless the scale is a power of two.
- vLLM's CUTLASS W4A8 kernel shows that rounding "weight × scale" into FP8 is an accepted approach on Hopper (for INT4, group 128).

---

**Next:** [08 · The vanilla NVFP4 checkpoint](08-The-Vanilla-NVFP4-Checkpoint.md)
