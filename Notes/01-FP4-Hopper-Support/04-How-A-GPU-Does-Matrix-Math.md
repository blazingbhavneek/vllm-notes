# 04 · How a GPU does matrix math

**What you'll learn:** the parts of a GPU that matter for a matrix multiply; why the same kernel is limited by memory in one case and by math in another; which tensor-core instructions each NVIDIA generation has (and why Hopper has no FP4); and how a kernel unpacks 4-bit weights "in registers". This is the background for notes 07, 13, 17 and 18.

---

## 1. The parts of a GPU, from slow and big to fast and small

| Part | Size (H200) | Speed | What lives there |
|---|---|---|---|
| **HBM** ("global memory", "VRAM") | 141 GB | 4.8 TB/s | Weights, KV cache, activations |
| **L2 cache** | 50 MB | faster | Recently used data |
| **Shared memory** (per SM) | up to 228 KB | very fast | Tiles a thread block is working on |
| **Registers** (per thread) | up to 255 × 32-bit | fastest | Numbers being computed right now |

- An **SM** (streaming multiprocessor) is one of the GPU's ~132 small processors (H100/H200 SXM).
- Each SM has **CUDA cores** (ordinary math, like FP32 add and multiply) and **tensor cores** (special units that multiply small matrices very fast).
- A **thread block** (also called a CTA) is a group of threads that runs on one SM and shares its shared memory.
- A **warp** is 32 threads that execute together. On Hopper, a **warpgroup** is 4 warps (128 threads).

## 2. How a matrix-multiply kernel (GEMM) works

For `y[M, N] = x[M, K] · W[N, K]ᵀ`:

1. The output is cut into **tiles**, for example 128 × 128. Each thread block owns one tile.
2. The block walks along K in steps (for example 64 at a time). In each step it:
   - copies a slice of `x` and a slice of `W` from HBM into shared memory;
   - feeds them to the tensor cores, which add `x_slice · W_sliceᵀ` into **accumulators** (FP32 registers).
3. At the end, the **epilogue** applies any final scales, converts to BF16, and writes the tile to HBM.

Good kernels overlap step 2's copy for slice *k+1* with the math for slice *k* (a **pipeline**). On Hopper, the **TMA** unit does these copies in the background.

## 3. Two costs: moving bytes and doing math

A kernel's time is roughly the larger of:

```
time_memory = bytes moved / memory bandwidth
time_math   = math operations / peak math rate
```

A GEMM does `2 × M × N × K` operations (one multiply and one add per term).

For a W4A16 GEMM, the bytes are mostly the weights: `N × K / 2` bytes for FP4 (plus a little for scales and activations). So:

```
operations per byte  ≈  (2 × M × N × K) / (N × K / 2)  =  4 × M
```

On an H200:

| Math type | Peak (dense, 📄 spec sheet) | Operations per byte where math becomes the limit | M (tokens) where that happens for FP4 weights |
|---|---:|---:|---:|
| BF16 tensor cores | ≈ 989 TFLOPS | 989 / 4.8 ≈ 206 | M ≈ 52 |
| FP8 tensor cores | ≈ 1,979 TFLOPS | ≈ 412 | M ≈ 103 |

How to read this:

- **M below ~50: memory-bound.** The kernel spends its time reading weights. Fewer weight bytes = faster. FP4 weights are 4× smaller than BF16, so W4A16 is ~4× faster than BF16 here. Faster math (FP8) does **not** help.
- **M above ~100: compute-bound.** The kernel is waiting on math. FP8 math is 2× faster than BF16 here. This is where a W4A8 kernel could win.

## 4. Decode and prefill

- **Decode** produces one new token per request per step. M = number of requests in the batch (times 1 + the number of speculative tokens, if MTP is on). Usually M is 1 to 64. **Memory-bound.**
- **Prefill** processes the prompt. vLLM cuts long prompts into chunks of `--max-num-batched-tokens` (for example 4096 or 8192). M is large. **Compute-bound** for dense layers.

### The MoE twist

In an MoE layer, each token goes to only `top_k` of `E` experts. Each expert's GEMM sees only its share of tokens:

```
tokens per expert ≈ M × top_k / E     (if routing is even)
```

| Model | E | top_k | Tokens per expert at M = 4096 | at M = 16384 |
|---|---:|---:|---:|---:|
| Qwen3.8-Flash-Next | 512 | 10 | 80 | 320 |
| GLM-5.3-Flash | 288 | 8 | 114 | 455 |

At a 4096-token prefill chunk, each expert GEMM sits *right at* the memory/compute crossover (~50 to 100). So a W4A8 kernel would help MoE prefill only modestly unless chunks are large. **Measure before building** (note 18, Phase D).

## 5. Tensor-core instructions, generation by generation

| GPU generation | Compute capability | Examples | Tensor-core instruction | Input types | FP4 math? | Hardware block scales? |
|---|---|---|---|---|---|---|
| Ampere | sm_80 / sm_86 | A100, RTX 3090 | `mma.sync` (one warp) | FP16, BF16, TF32, INT8, INT4 | no | no |
| Ada | sm_89 | L40S, RTX 4090 | `mma.sync` | + FP8 (E4M3, E5M2) | no | no |
| **Hopper** | **sm_90** | **H100, H200, H20** | **`wgmma`** (one warpgroup, asynchronous, reads operands from shared memory) + `mma.sync` | FP16, BF16, TF32, FP8, INT8 | **no** | **no** |
| Blackwell datacenter | sm_100 / sm_103 | B200, GB200, B300 | `tcgen05.mma` ("UMMA", uses on-chip tensor memory) | + FP6, **FP4** | **yes** | **yes** (MX and NVFP4 scale vectors) |
| Blackwell consumer / workstation | sm_120 / sm_121 | RTX 5090, **RTX 5060 Ti**, RTX PRO 6000, DGX Spark (GB10, sm_121) | block-scaled `mma.sync` ("MXMMA" in Humming's code) | + FP6, **FP4** | **yes** | **yes** |

Facts about Hopper's `wgmma` that matter later:

- For FP8 inputs, one instruction covers **K = 32**. For BF16 it covers K = 16.
- It has **no scale inputs**. Every element in the K = 32 chunk is multiplied as-is and summed in the accumulator. If different parts of the chunk need different scales, the kernel must fix that *before* the multiply (by scaling the operand) or *after* it (by "promotion", section 7).
- It reads at least one operand from shared memory, and the other operand can come from registers. That's how a kernel can feed **freshly unpacked** weights from registers.

Blackwell's FP4 instructions take a scale for every 16 or 32 elements and apply it in hardware. That's the one thing Hopper truly lacks (note 13).

## 6. Unpacking 4-bit weights inside a kernel (W4A16)

This is the trick that lets Hopper run FP4 checkpoints at all:

```
 HBM: packed FP4 weights (4 bits each) + E4M3 scales
   │   (copy a tile: 4× fewer bytes than BF16)
   ▼
 shared memory: packed tile
   │   (each thread loads 32 bits = 8 FP4 values into a register)
   ▼
 registers: 8 FP4 values ──bit tricks──► 8 BF16 values ──× scale──► 8 scaled BF16 values
   │
   ▼
 tensor cores: BF16 × BF16 → FP32 accumulators
   │
   ▼
 epilogue: × global scale → BF16 output → HBM
```

The unpacked values exist only in registers, for a few cycles. Memory holds only the 4-bit form. Note 07 shows the exact bit tricks Marlin uses.

For **W4A8** the picture is the same, except the unpack target is FP8 instead of BF16, and the activations are also FP8. Unpacking FP4 to FP8 is exact; applying the E4M3 scale in FP8 is not (note 17).

## 7. "Promotion": applying a scale that changes along K

Sometimes a scale can't be folded into the operand. For example, DeepGEMM's FP8 kernels on Hopper have one activation scale per token per 128 columns and one weight scale per 128 × 128 block. They work like this:

1. Run 4 `wgmma` instructions (4 × 32 = 128 values of K) into a **temporary** accumulator.
2. Multiply that temporary result by `scale_activation × scale_weight` for this K-block, using CUDA cores, and add it to the **main** FP32 accumulator.
3. Repeat for the next 128 columns.

Step 2 is called **promotion**. It costs CUDA-core work for every K-block. Once per 128 columns is affordable. Once per 32 columns would cost about as much time as the tensor-core math itself, so it would cancel FP8's 2× advantage. Once per 16 isn't possible, because a single FP8 instruction already covers 32. Note 17 builds on this.

## 8. The GPUs in this project

| GPU | Arch | Memory | Bandwidth (📄 approx.) | Notes |
|---|---|---|---:|---|
| H200 SXM (at work) | sm_90 | 141 GB HBM3e (≈ 131 GiB) | 4.8 TB/s | Target of this project. No FP4 math. |
| RTX 5060 Ti (at home) | sm_120 | 16 GB GDDR7 | 448 GB/s | Has FP4 math. Too small for these models. Good for unit tests and small checkpoints. |
| RTX PRO 6000 Blackwell | sm_120 | 96 GB | ≈ 1.8 TB/s | What FP4-Local was validated on (1 GPU for Qwen, 2 for GLM). |
| DGX Spark (GB10) | sm_121 | 128 GB unified | ≈ 273 GB/s | Also validated. |
| B200 | sm_100 | 192 GB | ≈ 8 TB/s | Not used by b12x (b12x targets sm_120 / sm_121). |

## Key takeaways

- Small M (decode) is limited by memory. 4-bit weights win there, whatever the math type.
- Large M (prefill) is limited by math. FP8 math is 2× BF16 on Hopper, but MoE experts see only a slice of M.
- Hopper has FP8 tensor cores, but no FP4 math and no hardware block scales. One FP8 `wgmma` covers K = 32.
- Unpacking inside registers keeps memory small. The unpack target (BF16 or FP8) changes speed and exactness, not memory.

---

**Next:** [05 · Model: Qwen3.8-Flash-Next](05-Model-Qwen3.8-Flash-Next.md)
