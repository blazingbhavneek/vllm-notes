# 22 · Glossary

One line per term. The number in brackets is the note that explains it best.

---

## Numbers and formats

- **Bit / byte:** a bit is 0 or 1; a byte is 8 bits. [02]
- **Sign, exponent, mantissa:** the three parts of a floating-point number: + or −, the power of two, and the digits after the leading 1. [02]
- **ExMy:** a float format with x exponent bits and y mantissa bits (E4M3 = 4 exponent + 3 mantissa bits, plus a sign bit). [02]
- **Binade:** the range between two neighboring powers of two (for example 2 to 4). "Spans 1.5 binades" = the largest value is about 2^1.5 times the smallest. [02, 11]
- **Exactly representable:** a value the format can store with no rounding. [02, 17]
- **Subnormal:** a tiny value below the format's normal range, stored with fewer significant bits. [02]
- **FP32 / TF32:** 32-bit float / a 19-bit tensor-core format with FP32's range. [02, 04]
- **FP16:** 16-bit float, 5 exponent and 10 mantissa bits. [02]
- **BF16:** 16-bit float with FP32's 8 exponent bits and 7 mantissa bits; the models' original precision. [02]
- **FP8 E4M3:** 8-bit float, max 448; used for FP8 weights and activations and for NVFP4 block scales. [02]
- **FP8 E5M2:** 8-bit float with more range and less precision; mostly for gradients. [02]
- **UE8M0 / E8M0:** an 8-bit, exponent-only, unsigned number: a pure power of two. The MX formats' scale. [02]
- **E2M1 (FP4):** 4-bit float with values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives. [02]
- **S0E5M3:** Marlin's internal 8-bit scale format (no sign, 5 exponent, 3 mantissa bits), made from E4M3 at load. [07]
- **INT4 / INT8:** 4-bit / 8-bit integers. [03]

## Quantization

- **Quantization:** storing numbers with fewer bits, as small codes plus scales. [03]
- **Scale:** the number codes are multiplied by to get back real values. [03]
- **Group / block:** the run of consecutive weights that share one scale (16 for NVFP4, 32 for MX). [03]
- **Per-tensor / per-channel / per-group / per-block scale:** one scale for the whole tensor / per output row / per small run / per 2-D tile. [03]
- **Global scale (`weight_scale_2`):** NVFP4's FP32 per-tensor scale, on top of the E4M3 block scales. [03, 08]
- **`input_scale`:** a per-tensor activation scale stored in ModelOpt checkpoints, used by FP4-activation (W4A4) kernels. [08]
- **NVFP4:** NVIDIA's 4-bit format: E2M1 values, one E4M3 scale per 16, one FP32 global scale. [03]
- **MXFP4 / MXFP8:** OCP microscaling formats: FP4 or FP8 values with one UE8M0 scale per 32. [03]
- **FP8_PB_WO:** DeepSeek-style FP8 weights with one scale per 128 × 128 tile ("per-block, weight-only"). [03]
- **WxAy:** x-bit weights, y-bit activations: W4A16, W4A8, W4A4, W8A16, W8A8. [03]
- **Weight-only quantization:** only the weights are small; the math still runs in 16 bits after unpacking (W4A16). [03]
- **Activation quantization:** activations are also made small at runtime so low-precision tensor cores can be used (W4A8, W4A4). [03]
- **PTQ:** post-training quantization: round a finished model's weights, no retraining. [03]
- **QAT:** quantization-aware training: train while simulating the rounding. [03]
- **QAD:** quantization-aware distillation: keep training the 4-bit model (student) to match the full-precision model (teacher). [10]
- **GPTQ / AWQ:** popular PTQ methods for INT4 weights. [03]
- **GSQ:** Gumbel-Softmax Quantization; learns which grid value each weight rounds to. Output is a normal GGUF. [09]
- **RCO:** Riemannian Constrained Optimization; picks the format per tensor under a total size budget. [09]
- **Ignore list (`ignore`):** checkpoint config entries for layers left in BF16. [08]
- **`quantized_layers`:** ModelOpt mixed-precision config: per-layer recipe (format, group size, scale encoding). [08, 15]
- **Lossless:** decoding gives back exactly the original bytes. [11]

## CSF

- **CSF:** NVIDIA's lossless compression of NVFP4 block scales (per-row base byte + 4-bit codes + exceptions). [11]
- **`weight_scale_encoding`:** the config field that says which CSF variant a layer uses (`csf` or `csf_32x2_v1`). [11, 20]
- **Base:** one byte per scale row; codes are added to it. At most 240. [11]
- **Code:** a 4-bit value, `scale byte − base`, two per byte, low nibble first. [11]
- **Exception:** a scale outside `[base, base + 15]`, stored separately with its position. [11]
- **Nibble:** half a byte (4 bits). [02]
- **Slab CSF (`csf`):** variant with 2 tensors, `nvfp4_csf_fixed` (bases + codes, 16 rows per slab) and `nvfp4_csf_exceptions` (u32 position + value). [11]
- **`csf_32x2_v1`:** GPU-friendly variant with 5 tensors (`csf32_bases`, `codes`, `page_bases`, `offsets`, `exceptions`). [11]
- **Block / tile / page (csf32):** 32 × 2 scales / 4 × 2 blocks / 32 tiles. [11]
- **Scratch buffer:** temporary GPU memory where compressed scales are decoded just before a kernel uses them. [14, 18]
- **Routed-only / active-set / all-experts decode:** b12x's three decode modes (only routed experts, only experts in use, everything). [14]

## Hardware

- **SM:** streaming multiprocessor, one of the GPU's ~132 compute units. [04]
- **sm_XY:** the compute capability: sm_90 = Hopper, sm_100 = datacenter Blackwell, sm_120 / 121 = consumer Blackwell and DGX Spark. [04]
- **Hopper:** H100, H200, H20 (sm_90). FP8 tensor cores, **no** FP4 tensor cores. [04, 13]
- **Blackwell:** B200, B300, GB200 (sm_100) and RTX 50xx, RTX PRO 6000, DGX Spark / GB10 (sm_120 / 121). FP4 tensor cores. [04, 13]
- **H200:** Hopper with 141 GB of HBM3e (about 140 GB usable by vLLM). [19]
- **HBM / VRAM:** the GPU's main memory. [04]
- **Shared memory / registers / L2:** fast on-chip storage, from medium to fastest; L2 is a cache shared by all SMs. [04]
- **Warp / warpgroup:** 32 threads that run together / 4 warps (128 threads). [04]
- **Tensor core:** the hardware unit that multiplies small matrix tiles. [04]
- **`mma.sync`:** the classic warp-level tensor-core instruction. [04]
- **`wgmma`:** Hopper's warpgroup-level asynchronous tensor-core instruction, operands from shared memory. [04]
- **Block-scaled MMA:** Blackwell tensor-core instructions that apply per-block scales inside the hardware (needed for fast FP4). [04, 13]
- **TMA:** Hopper's copy engine that moves tiles from HBM to shared memory in the background. [04]
- **PCIe / NVLink:** CPU–GPU link / fast GPU–GPU link. [05, 14]
- **TFLOPS:** trillions of floating-point operations per second. [04]

## Kernels and math

- **Kernel:** a function that runs on the GPU. [04]
- **GEMM:** general matrix multiply, `C = A × B`. [04]
- **M, N, K:** GEMM sizes: M = tokens, N = output features, K = input features. [04]
- **Memory-bound / compute-bound:** time limited by bytes moved / by math. [04]
- **Arithmetic intensity:** FLOPs per byte moved; tells which bound applies. [04]
- **Dequantize:** turn codes back into real values (codes × scale). [03]
- **On-the-fly dequantization:** done inside the kernel, in registers, never written back to memory. [07]
- **Epilogue:** the end of a GEMM kernel, where results are scaled and written out. [04, 17]
- **Promotion:** periodically adding a low-precision partial sum into an FP32 accumulator (DeepGEMM-style), also the place to apply per-block scales. [04, 17]
- **Accumulator:** the higher-precision register that holds the running sum of products. [04]
- **LUT:** lookup table. [07]
- **Permutation / index map:** a precomputed table that says where each element goes in a kernel's memory layout. [18]
- **JIT:** just-in-time compilation: kernels are compiled at first use for the exact shape. [07]
- **CUDA graph:** a recorded sequence of GPU work replayed with little CPU overhead; needs fixed buffer addresses. [15]
- **Custom op:** a function registered with PyTorch so `torch.compile` and CUDA graphs treat it as one opaque step. [18]
- **Stream:** a GPU work queue; work on two streams can overlap. [14]

## Kernel libraries

- **Marlin:** IST-DASLab's 4-bit-weight × 16-bit-activation kernel family, extended in vLLM to NVFP4, MXFP4, FP8 and more. [07]
- **Humming:** jinzhen-lin's JIT GEMM library (`vllm-project/humming`); vLLM's first choice for NVFP4 and MXFP8 on sm_90. [07]
- **`prioritize_humming`:** vLLM helper that puts Humming in front of Marlin on sm_90 only. [07]
- **CUTLASS / CuTe / CuTe DSL:** NVIDIA's C++ GEMM template library / its layout algebra / its Python kernel language. [04, 14]
- **DeepGEMM:** DeepSeek's FP8 GEMM library, with per-128 promotion. [17]
- **FlashInfer:** a kernel library vLLM uses for attention, sampling and some FP4 MoE kernels. [08, 14]
- **TRT-LLM kernels:** TensorRT-LLM's kernels, some reused by vLLM through FlashInfer. [08]
- **b12x:** Local Inference Lab's CuTe DSL kernel library for sm_120 / 121, with on-the-fly CSF decode. [14]
- **llama.cpp / ggml:** CPU/GPU inference engine that defined GGUF and its kernels (MMVQ, MMQ). [09]
- **MMVQ / MMQ:** llama.cpp's one-token kernel / batched kernel for GGUF types. [09]

## Model parts

- **Transformer layer:** one block of attention (or linear attention) followed by an MLP or MoE. [05]
- **Embedding / LM head:** the table that maps token ids to vectors / the final layer that maps vectors to token scores. [05]
- **Hidden size:** the length of each token's vector. [05]
- **Attention / KV cache:** the mechanism that looks at past tokens / the stored keys and values of past tokens. [05]
- **Linear attention:** attention replaced by a fixed-size running state (no growing KV cache). [05]
- **Gated DeltaNet (GDN):** Qwen's linear attention. [05]
- **KDA:** Kimi Delta Attention, GLM's linear attention. [06]
- **QSA:** Qwen Sparse Attention, attention over a selected subset of past tokens. [05]
- **DSA:** DeepSeek Sparse Attention (MLA + an indexer that picks 2048 past tokens). [06]
- **MLA:** multi-head latent attention: keys and values stored compressed (`kv_a`), expanded by `kv_b_proj`. [06]
- **MLP / SwiGLU:** the per-token feed-forward block: `down(silu(gate(x)) · up(x))`. [05]
- **`gate_proj`, `up_proj`, `down_proj`:** the three matrices of an MLP / expert. Gate and up are often fused. [05]
- **`swiglu_limit`:** GLM's clamp on the SwiGLU input. [06]
- **MoE:** mixture of experts: many small MLPs, a few chosen per token. [05]
- **Router / top-k:** the small layer that scores experts / the number chosen per token (10 for Qwen, 8 for GLM). [05, 06]
- **`topk_ids`:** the chosen expert ids per token, the input to routed-only decode. [14]
- **Routed expert / shared expert:** experts picked by the router / an expert every token uses. [05]
- **Hyper-connections / mHC:** carrying 4 residual streams instead of 1 (mHC = GLM's manifold-constrained version). [05, 06]
- **PLE:** Qwen's 51-billion-entry n-gram embedding table in layer 2; a lookup, not a matrix multiply. [05]
- **MTP:** multi-token prediction layer, used as a speculative-decoding draft. [05]
- **Speculative decoding:** a cheap draft guesses tokens; the big model checks several at once. [05]
- **Vision encoder / `mmproj`:** the image and video part of the model / its GGUF file. [05, 09]
- **`native-QB`:** GLM FP4-Local's shard in `tensors/` with BF16 tensors such as `kv_b_proj`. [06]

## vLLM

- **Model loader:** code that finds shards, reads tensors and calls `load_weights` (`default_loader.py`). [15]
- **`_prepare_weights`:** loader function that lists the shard files to read. [15, 20]
- **Shard:** one `.safetensors` file of a checkpoint; the index's `weight_map` says which tensor is in which shard. [15]
- **safetensors:** a file format: a JSON header followed by raw tensor bytes. [15]
- **Quantization config / method:** reads the checkpoint config (`ModelOptNvFp4Config`, `ModelOptMixedPrecisionConfig`) / creates and runs the quantized layers. [15]
- **`process_weights_after_loading`:** the step that rearranges checkpoint tensors into a kernel's layout. [15]
- **Oracle / backend selection:** code that picks the MoE / linear kernel per GPU (`fused_moe/oracle/nvfp4.py`). [15]
- **`--moe-backend` / `--linear-backend`:** CLI flags that force a kernel family. [15]
- **TP / EP / DP:** tensor / expert / data parallelism, ways to split work over GPUs. [19]
- **Prefill / decode:** processing the prompt (many tokens at once) / generating one token per step. [04]
- **Prefix caching:** reusing the KV cache of a prompt prefix seen before. [18]
- **Eager mode (`--enforce-eager`):** run without CUDA graphs; slower, easier to debug. [21]
- **ModelOpt:** NVIDIA's quantization toolkit that produced these checkpoints. [08]

## Files and model families

- **GGUF:** llama.cpp's single-file format with its own block quant types. [09]
- **Q4_0, Q4_K, Q6_K, …:** GGUF "legacy" and "K-quant" block types. [09]
- **IQ2_XXS … IQ4_NL / IQ4_XS:** GGUF "i-quants", codebook-based types. [09]
- **Q2_0:** a newer simple 2-bit GGUF type without a codebook. [09]
- **FP4-Local:** NVIDIA's QAD NVFP4 checkpoints with CSF scales, made for sm_120 / 121. [10]
- **LIL:** Local Inference Lab, the community behind the original QAD checkpoint, b12x and the vLLM fork. [10, 14]
- **IST-DASLab:** the IST Austria lab behind Marlin, GPTQ, GSQ and RCO. [07, 09]

## Testing

- **GSM8K:** 1,319 grade-school math word problems; a quick accuracy check. [21]
- **AIME25 / GPQA:** harder reasoning benchmarks. [10, 12]
- **KL divergence (KLD):** how different two models' next-token distributions are; 0 = identical. [21]
- **NLL / perplexity:** average −log probability of the real next token / its exponential. [21]
- **ITL / TTFT:** inter-token latency / time to first token. [21]
- **nsys:** NVIDIA Nsight Systems, a timeline profiler. [21]
- **compute-sanitizer:** NVIDIA's tool that finds out-of-bounds memory accesses and races. [21]
- **RFC:** request for comments, a GitHub issue that proposes a design before code. [20]

---

**Next:** [23 · Sources, open questions and risks](23-Sources-Open-Questions-And-Risks.md)
