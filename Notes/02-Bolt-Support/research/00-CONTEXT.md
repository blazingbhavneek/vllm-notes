# Shared context for the "how does everyone else run IQ2_XXS" research lanes

You are one of several parallel research lanes. Read this whole file first, then
do your own lane's brief. You are **read-only with respect to every repository you
are not told to write to**: do not edit source, do not run git commands, do not
commit. Your only write target is the one report file named in your brief.

## Who is asking, and why

We ship a Triton block-quantization path (IQ2_XXS + Q8_0) for
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` in a vLLM fork:

- repo `cwd` for code: `/home/seigyo/vllmServer/vllm` (editable install)
- kernel file: `vllm/model_executor/layers/quantization/modelopt_block_quant_triton.py`
  (`_decode_tile`, `_gemv_kernel`, `_gemm_kernel`, `_moe_gemv_kernel`,
  `_splitk_reduce_kernel`, `repack_blocks`, `dequantize_blocks`, `iq2_lut`)
- MoE expert wrapper: `vllm/model_executor/layers/fused_moe/experts/modelopt_block_triton_moe.py`
- tests: `tests/kernels/quantization/test_modelopt_block_triton.py`
  (`ref_decode` is a bit-exact GGML IQ2_XXS/Q8_0 reference decoder)
- GPU: **H200 (SM90)**, heavily shared with other tenants, so only same-process
  interleaved A/B timings are trustworthy.
- model: 512 routed experts, top-9, `hidden_size=4096`, routed experts run on a
  **1024-wide projected space** (`up_proj [2048,1024]`, `down_proj [1024,2048]`),
  shared expert `up [5376,4096]` / `down [4096,5376]`. IQ2_XXS for the MoE and
  dense block weights, Q8_0 for lm_head/qkv/o/fc, NVFP4 for the Mamba layers.

## The weight format (identical to GGML IQ2_XXS)

66 bytes per 256 values: fp16 `d`, then per 32 values one `idx` byte and one
`aux` byte (4+4 bytes per 32 values). `idx` selects a row of a 256-entry
codebook of **8 magnitudes, each from {8, 25, 43}**; `aux` carries a 7-bit sign
mask (the 8th sign bit is a parity fold) and, in the top 4 bits of the 4-byte
word, a grid index. Value = `sign * mag * d * (0.5 + grid/16) * 0.25`, i.e.
**3 bits of magnitude+sign per value + a per-32 scale**, stored in 2.0625
bits/value. So an exact codebook-free "plane" rep costs 3.5 bits/value.

## What we already measured on this box (do not re-derive; do not repeat)

| measurement | value |
|---|---|
| dense IQ2 GEMV, shared_up 5376x4096 (5.7 MB), M=1, in-tree incl. split-K reduce | 22.5 us (252 GB/s of packed bytes) |
| same tile, hand-written probe, no reduce | 18.8 us |
| load-only floor for the same bytes (probe) | 6.3 us = 905 GB/s achieved read rate |
| probe: decode without the dot | 17.7 us |
| probe: LUT read as broadcast instead of gather | 25.8 us (gather is FREE) |
| codebook-free 3-bit plane rep, bit-exact, same tiling+reduce | 13.3 us (1.70x faster, 1.70x bytes) |
| fused MoE at (E=512, inter=2048), M=1 | 49.8 us ~= 195 GB/s |
| Q8_0 lm_head | ~2.1 TB/s (near memory-bound, no headroom) |

Already tried and rejected **after measurement**: BN 2/4/16/32, BK 128/512/1024,
num_warps 2/4, num_stages 1/2/3, split-K powers and odd divisors, sign by XOR on
the fp32 sign bit, dropping the parity chain, 4-D accumulate with a single final
reduce (register spills), a 1-D accumulator for the M=1 case (exactly neutral),
folding the per-32 scale out of the inner loop (2-3x slower in the real kernel),
swapping the NVFP4 backend, and fusing the split-K reduce into the GEMV (rejected
for CUDA-graph safety). Conclusion so far: **inside the codebook format the decode
arithmetic and tile layout are the wall**; only buying bytes (plane rep) moves it.

## Hard constraints on anything you propose

1. Must stay bit-exact on the IQ2 values (the checkpoint is fixed; accuracy bars
   are already tight).
2. Runs under CUDA graph capture: no JIT/compilation at replay time.
3. Triton is the implementation language here; a technique only usable in CUDA
   C++ is still interesting if you explain the equivalent Triton formulation.
4. Extra VRAM is not free: the model is sized for a 24 GB card, so a +70 % bytes
   idea is a last resort, and you must say what it costs.

## The question every lane must answer

Given how *others* implement IQ2_XXS (or an equivalent codebook quant) on GPU at
**decode batch size 1**, what decode structure gets within ~2x of the memory
floor **without** increasing the weight bytes? Concretely: how do they break the
dependent chain of `load code -> table lookup -> unpack 8 values -> apply sign ->
reshape into the dot-product layout`? Integer (dp4a / HMMA-int8 / half2 HMMA)
accumulation against a codebook, register-resident or shared-memory codebooks,
permute/`lop3`/`prmt`-based codebook decode into packed half2, or a smarter block
layout, are the hypotheses we care about — but report what the code actually does,
not what you expect.

## Report format (write to your one file)

- `## Verdict` - 3-6 sentences: is there something to copy, and what is it worth?
- `## Techniques found` - one bullet per technique: what it is; `file:line`
  reference; what it costs (bytes, registers, accuracy); and **maps to Triton as**
  (1-2 lines, concrete enough that we could prototype it in an hour).
- `## Dead ends confirmed` - things they tried that we should not bother with,
  with the evidence (comments, PR links, git log, benchmark numbers in comments).
- `## Sources` - repo, commit/tag, file:line list, PR/issue/discussion links.

Be specific and quantitative. "llama.cpp uses a table lookup" is useless;
"llama.cpp loads the 8 magnitudes from a `__constant__` 2 KB table with
`__ldg`, one warp per 32-value block, and accumulates in fp32, i.e. the same
thing we do, which is why it is also at 4x off the memory floor" is useful.
