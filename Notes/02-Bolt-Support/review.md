# Review: SM90 fallback for ModelOpt IQ2_XXS / Q8_0 (opencode sessions 1 + 2)

Reviewed 2026-10-22.

- **Diff reviewed:** the draft, now commit `2064cc2efb` on `feature/wip-nvidia-bolt-low-sm-support` (fork `blazingbhavneek/vllm`, base `ef5bcaa387`).
- **Files:** `modelopt_block_quant.py` (+36/−9), new `modelopt_block_quant_sm90.py` (250 lines), new `modelopt_block_quant_hopper.py` (900 lines), new test file, new benchmark.
- **What to do about it:** `03-improvment-handoff.md`. It refers to the finding IDs below (B = blocker, K = kernel quality, S = smaller).

## Verdict

The math is correct: every kernel matches an independent GGML reference. But it isn't ready to upstream:

- It's slow for a fused kernel: 13–30% of memory bandwidth at decode.
- Two configs break on GPUs with ≤100 KB of shared memory.
- `torch.compile` silently freezes its batch-size dispatch.
- It reads the IQ2 grid from a private b12x module.
- It's about 3× bigger than it needs to be.

The fixes are clear and mostly involve deleting code.

## What was tested

Hardware and versions: home RTX 5060 Ti, sm_120, torch 2.14.0+cu130, Triton 3.8.0 (same versions as the work box).

The Triton module was loaded standalone, with no vLLM install. The oracle is `gguf-py` (llama.cpp's own Python port), not the agent's code or b12x. Scripts are in `kernel-tests/`; run with `uv run --no-project --python 3.13 --with torch==2.14.0 --with gguf --with numpy python <script>`.

| Check | Result |
|---|---|
| IQ2_XXS and Q8_0 decode in fp32, including zero, negative, subnormal and 60000 scales | **Bit-exact** vs gguf-py |
| Decode in bf16 | Equals round-to-nearest of the oracle |
| `decode_expert_gather` | Exact |
| `block_linear`: GEMV, split-K and plain GEMM; M = 1…1000; N includes 257; 3-D and non-contiguous x | Error ≤ 1.34× cuBLAS error on the same bf16 operands |
| `moe_forward` vs fp64 reference: E = 8/64/128, top-k 2/9, M = 1…200; padding blocks poisoned | Max relative error 2.4e-3 to 5.9e-3; padding never read |
| Same kernels with the **shipped** `num_stages=2` | **OutOfResources**: needs 104–160 KB shared memory, limit is 99 KB |
| `torch.compile` with vLLM's settings (one trace, guards dropped) | **M=1 runs the BM=128 GEMM, not the GEMV** |
| Cross-compiled for sm_90a with Triton's own ptxas | All IQ2 kernels spill registers (K2) |

The correctness rows used `num_stages=1`, because 2 doesn't fit on this GPU.

### With vLLM installed (`~/.venvs/vllmNotes`, editable, torch 2.14.0+cu132, b12x installed)

The agent's fallback was forced on in-process by overriding the gates; the repo itself was not modified. The end-to-end script is `kernel-tests/e2e_tiny_bolt.py`; the other checks were one-off scripts, since deleted.

| Check | Result |
|---|---|
| Existing tests: `tests/quantization/test_modelopt.py` (block, iq2, mixed-precision) | 62 passed. The b12x path is not broken |
| Existing tests: `test_b12x_linear.py` + `test_b12x_warmup.py` on real sm_120 | 31 passed |
| The agent's own test file | 1 passed, 9 skipped (SM90-only gate) |
| b12x IQ2_XXS grid vs gguf grid; Triton decode vs `b12x.testing` reference | Identical |
| Fallback linear under vLLM-style compile + CUDA graph replay (rows 1/4/17) | Correct, memory stable |
| Same, kernel used at M=1 | Eager: `_iq2_gemv_kernel`. Compiled: `_iq2_gemm_split_kernel`. **Frozen at trace time** |
| `iq2_xs` | Accepted by `__init__`, `NotImplementedError` at first forward |
| Embedding fallback, 8-token lookup | 256 MiB scratch for a 32768×4096 table, so about 1 GiB at Bolt size |
| `moe_forward` with real `moe_align_block_size`/`moe_sum`, CUDA graph captured, replayed with new activations **and new routing** | Correct (rel ≤ 5.1e-3), memory stable. **Graph-safe claim confirmed** |
| Tiny Nemotron-H end to end via `LLM()`. Model: mamba + attention + latent ReLU² MoE + shared experts; IQ2_XXS on attention and routed experts, Q8_0 elsewhere. Modes: eager, compiled, compiled + CUDA graphs | All 9 block layers on the fallback. **CUDA-graph mode crashed at capture** (B4). With the grid pre-created it captures fine. Token agreement: see the note below |
| Kernels seen in compiled decode | `_q8_gemv_kernel` from the eager layers, but **no `_iq2_gemv_kernel`**. The traced IQ2 attention layers ran the frozen `_iq2_gemm*` kernels. B2 confirmed inside real vLLM |

**Note on token agreement.** A later rerun with the cleaned-up script compared eager and compiled with a BF16 control (no quantization). Even the control has prompt 0 diverge from the first step, while prompts 1–2 match. That divergence comes from vLLM, not from the block-quant code. Compare quantized runs against the control's pattern, never expect 100% equality.

Still untested:

- Hopper performance and the wgmma code path (sm_90 only)
- The real Bolt checkpoint
- GSM8K / `vllm bench`

## Findings

### Blockers

**B1. The IQ2 grid comes from a private b12x module.**
- `hopper.py:71` and `sm90.py:67` import the table from `b12x._lib.quant.iq2_xs`.
- *Corrected later the same day:* this does **not** crash on a standard H200 install. `flashinfer-python==0.7.2`, a core vLLM CUDA dependency (`requirements/cuda.txt`), bundles a top-level `b12x` package.
- It's still a private module of an experimental package, and it can move or vanish in any FlashInfer bump.
- **Fix:** embed the 256-entry table (2 KB), with a checksum test.

**B2. `torch.compile` freezes the dispatch.**
- Dense `apply` is traced by Dynamo (session 2 hit this). vLLM traces once at a large M and drops all guards (`compilation/wrapper.py:143`).
- So `if M == 1` (`hopper.py:635`), the BM choice and `if tiles < 256` (`:585`) are decided at trace time. In practice that means BM=128, plain GEMM, for every batch size. The GEMV and split-K paths never run when serving.
- Measured here, M ≤ 64 is 1.2–3.5× slower than intended:
  - shared-up at M=1: 279 vs 96 µs
  - fc1_latent at M=1: 180 vs 51 µs
- The "~16 tok/s" was measured on this frozen config.
- vLLM's own `docs/design/debug_vllm_compile.md` says: "avoid branching on the number of tokens … wrap the branching logic into a custom operator".
- **Fix:** register `block_linear` as a custom op. This also removes the `prewarm_device_checks` workaround.

**B3. Shared memory won't fit outside H100/H200/A100.**
- IQ2 GEMM at `num_stages=2` needs 104 KB (BM=16) to 160 KB (BM=128), and the MoE kernels need 104 KB. sm_86, sm_89 and sm_120 allow 99 KB.
- So prefill, split-K and the whole MoE path fail to launch on A10/3090/L4/4090/5090/RTX PRO. Hopper (227 KB) is fine.
- **Fix:** a smaller K step per iteration, and stages picked from the device's shared-memory limit.

**B4. CUDA-graph capture crashes when IQ2 is on a traced layer.**
- `_iq2_grid` (`hopper.py:74`) creates the lookup table lazily with `.to(device)`. When the first call happens inside Dynamo, the CPU→GPU copy is baked into the compiled graph.
- Capture then fails with "Cannot copy between CPU and CUDA tensors during CUDA graph capture". Without graphs it runs, but does a blocking host copy per IQ2 layer per step.
- Bolt avoids this only because all its IQ2 layers run inside the MoE custom op, which isn't traced.
- **Fix:** build the table once at weight-processing time and keep it on the layer.

### Kernel quality

**K1. The decode does per-weight work that should be per-group.**
- `_iq2_decode_tile` (`hopper.py:84`) computes everything on a `[BN, 256]` tensor:
  - the 4 control-byte loads and the scale: once per weight, though they belong to a 32-weight record (32× redundant)
  - the sign parity and the grid-index load: once per weight, though they belong to an 8-weight group (8× redundant)
- At M=1 on 5060 Ti:
  - IQ2 shared-up reaches 59 GB/s (13% of 448). That's only 1.28× faster than bf16 cuBLAS, which reads 7.75× more bytes.
  - Q8_0 lm_head reaches 133 GB/s, *slower* than cuBLAS on dense bf16 (4.3 vs 2.6 ms).
- The per-group prototype (`kernel-tests/proto_kernels.py`) reaches 48 µs / 117 GB/s on IQ2 shared-up and 1.40 ms on lm_head.

**K2. Register spills on sm_90a** (ptxas 12.9, the one Triton uses for Hopper):

| Kernel | Registers | Spill stores |
|---|---|---|
| iq2 GEMM, BM128 | 255 | 3.5 KB |
| iq2 split-K, BM16 (the decode path) | 32 | 17.6 KB, plus 6.5 KB stack |
| MoE pass 1 | 32 | 18.8 KB |
| GEMV | 255 | 1.5 KB |
| q8 GEMM | 168 | 0 (clean) |

Same cause as K1: the decoded `[64, 256]` fp32 tile plus per-element temporaries.

**K3. The unaligned layout prevents wide loads.**
- 66- and 34-byte blocks are only 2-byte aligned, so every load is a byte load.
- b12x itself repacks at load (`api.pack_weight`).
- **Fix:** a one-time split into a scales array plus an aligned payload, at the same total size.

**K4. The MoE tile is fixed at BM=16** (`hopper.py:843`).
- Every 16-token tile re-decodes the whole expert slice.
- Prefill at M=2048 took 48 ms per MoE layer (4 TFLOP/s). Dense cuBLAS with the same FLOPs takes 4.1 ms.
- **Fix:** size BM from tokens per expert.

**K5. The embedding fallback decodes the whole table on every forward** (`sm90.py:170`).
- That's 1 GiB of scratch per step for 131072×4096.
- Bolt's embeddings are BF16, so Bolt doesn't hit it, but any Q8_0-embedding checkpoint does.
- **Fix:** gather the rows first, then decode only those.

### Smaller

- **S1. int32 offsets:** `offs_m[:, None] * N` (`hopper.py:246, 285`). It overflows once M·N ≥ 2³¹, for example lm_head with `prompt_logprobs` and `--max-num-batched-tokens` > 16384.
- **S2. Unsupported combinations fail late:**
  - `iq2_xs` raises `NotImplementedError` at the first forward.
  - SiLU-gated IQ2 MoE and Q8_0 MoE fall into a `torch.unique` path that syncs with the host and crashes CUDA-graph capture.
- **S3. Explicit backends are silently ignored:** `--moe-backend` / `--linear-backend` are skipped on fallback (`modelopt_block_quant.py:81, 289`).
- **S4. Dead code and conventions:**
  - `_GRID_CACHE_LOCK`, two unused `_codec_layout` values, and an unreachable bf16 branch.
  - Triton is imported with try/except instead of `vllm.triton_utils`.
  - Two duplicate device gates with module-level caches.
  - An undocumented env var, `VLLM_SM90_BLOCK_KERNEL`.

### Other notes

- **Tests:** the repo tests have no independent oracle. `test_block_linear_*` compares against the same decoder under test, and there's no `moe_forward` test.
- **Shared experts:** they are **not** dropped (`runner/moe_runner.py:_maybe_apply_shared_experts`).
- **Served name:** session 2's serve command reused `--served-model-name gemma-4-31B`, the protected port-8000 server's name.
- **Q8_0 via Marlin W8A16 g32:** possible as a follow-up, but its scales are bf16, so the fp16 Q8_0 scales would lose 3 mantissa bits. Not for v1.
