## Overview

> **WIP: not yet human-reviewed.** AI assistance (Claude 5.5 + GPT 6.1 Sol) was used to write this PR.
> **TODO:** human review of the tests, benchmarks, diff and maintainability before this is marked ready.

ModelOpt block checkpoints such as `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (IQ2_XXS experts, Q8_0 dense layers) only run on SM120/SM121, because b12x is their only kernel path. On every other GPU, loading fails with `Block-quantized dense weights require b12x on SM12x`. This PR adds a Triton path for NVIDIA SM75+. SM120/SM121 keep b12x by default.

Fixes #62609 

**Not a duplicate:** #59486  is the b12x support for SM120/SM121 that this PR falls back from, and no open PR adds a path for other GPUs. I claimed the work in #62609 on 2026-10-22.

**Related:** Bolt in FP16 also needs a separate NemotronH bugfix PR (link TBD). Without it, FP16 Bolt scores 3/250 on GSM8K with any backend. See Details.

## Claims

- Bolt loads and serves on SM80, SM86, SM89 and SM90. The weights stay packed at 21.7 GiB.
- BF16 GSM8K-250 scores 230–236 across four GPUs (run-to-run spread is about ±5).
- The kernels support FP16 activations:
  - Kernel tests pass in FP16 on every tested GPU.
  - Full-model FP16 GSM8K scores 234–235 with the separate FP16 fix, at BF16 speed.
- On Turing (SM75), FMA tiles are used instead of tensor-core `tl.dot`. They cross-compile without spills (checked with ptxas), but have not been run on Turing hardware.
- Nothing compiles during inference: a warmup compiles every reachable kernel config before CUDA graph capture.
- Default behavior on SM120/SM121 is unchanged: b12x is still selected there.

## Validation

### GPUs and tests

| GPU | SM | Memory | `test_modelopt_block_triton.py` | `test_modelopt.py -k block` | Full model |
|---|---|---|---|---|---|
| H200 NVL | 90 | HBM3e | Run on an earlier revision (see TODO) | Earlier revision | BF16 GSM8K + serving |
| A100-class (CMP 170HX, 74 SMs) | 80 | HBM, 63 GiB | 192 passed, 2 min 51 s | 55 passed | BF16 + FP16 GSM8K, serving |
| L40S | 89 | GDDR6, 44 GiB | 192 passed, 3 min 9 s | 55 passed | BF16 + FP16 GSM8K, serving |
| RTX 3090 | 86 | GDDR6X, 24 GiB | 192 passed, 2 min 4 s | 55 passed | BF16 GSM8K (eager) |
| RTX 5060 Ti | 120 | GDDR7, 16 GiB | 191 passed, 1 skipped (needs 12 GiB free) | 55 passed | — (16 GiB) |
| T4 | 75 | — | not run | not run | — |

**Commands:**

```bash
pytest tests/kernels/quantization/test_modelopt_block_triton.py
pytest tests/quantization/test_modelopt.py -k "block or iq2"
```

**What the tests check:**
- One IQ2_XXS block is built byte by byte in the test, with every expected weight written out, so the decode can be checked by hand. The embedded grid table is checked against gguf-py's by hash.
- The kernel tests compare against an exact ggml reference decode. `dequantize_blocks` is also bit-exact on every tensor of the real Bolt checkpoint.
- Dense and MoE outputs on Bolt's layer shapes stay within 1.5x of cuBLAS's error on the dequantized weights, also when two layers are stacked.
- Non-finite scales stay in their rows; batch-invariant mode gives bit-identical rows across batch sizes.
- The warmup check reported no inference-time JIT from these kernels in BF16 or FP16. The only kernel that compiled at inference was upstream Mamba's `_causal_conv1d_fwd_kernel`.

### GSM8K-250

**Setup:**
- The first 250 test questions, sent to `/v1/completions` with the prompt `Q: {question}\nA: Let's think step by step.\n`.
- Temperature 0, `max_tokens` 768, 32 requests at a time.
- The prediction is the last number in the output. 225 is the pass bar.
- BF16 varies between 230 and 235 on identical code.

| GPU | BF16 | FP16 with the FP16 fix | FP16 without it |
|---|---:|---:|---:|
| H200 NVL | 236 | — | — |
| A100-class | 230, 231 | 235 | 3 |
| L40S | 235, 230, 232 | 234 | 3 |
| RTX 3090 | 231 | — | — |

### Serving, BF16

**Setup:** `vllm bench serve` with random data, 128 tokens in and 256 out, `--ignore-eos`, temperature 0. Each run used 12 prompts at c=1 and 32 prompts otherwise. Cells show output tok/s, with median TPOT in parentheses.

| GPU | c=1 | c=8 | c=16 | c=32 |
|---|---:|---:|---:|---:|
| H200 NVL ¹ | 132.1 (7.1 ms) | 492.0 (15.1 ms) | — | 865.0 (34.1 ms) |
| A100-class | 80.3 (11.7 ms) | 218.6 (34.2 ms) | 311.4 (48.1 ms) | 402.6 (73.3 ms) |
| L40S ² | 87.5 (10.6 ms) | 346.9 (21.0 ms) | 486, 512 | 623, 615 |

**Footnotes:**
1. The H200 run used `--gpu-memory-utilization 0.55` with `modelopt_iq2_dense_cache` on.
2. L40S c=16 and c=32 are two runs each with this PR's MoE setting (GEMV up to 8 rows). c=1 and c=8 come from the run just before that change, where those batch sizes already took the same path.

**FP16 vs BF16 speed** (A100-class, with the FP16 fix), c=1/8/32:

| Dtype | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| FP16 | 81.1 | 219.4 | 395.0 |
| BF16 | 80.3 | 218.6 | 402.6 |

There is no "before" column, because the model does not load on these GPUs without this PR.

**Commands:**

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --trust-remote-code --language-model-only \
  --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"marlin"}}'

vllm bench serve --model nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --trust-remote-code \
  --dataset-name random --random-input-len 128 --random-output-len 256 \
  --num-prompts 32 --max-concurrency 32 --request-rate inf --ignore-eos --temperature 0 --seed 0

python benchmarks/kernels/benchmark_modelopt_block_triton.py
```

The NVFP4 Mamba projections were pinned to Marlin so every GPU used the same backend. The auto choice also works: it picked Humming on the H200 and served at the same speed.

<details>
<summary>Which revision each run used</summary>

Development history, unsquashed: `blazingbhavneek/vllm`, branch `feature/modelopt-block-triton`.

- **5060 Ti:** ran this exact commit.
- **A100-class:** ran `ff9dd45e8b`, which also contains the FP16 fix. HBM dispatch is identical to this PR.
- **L40S:**
  - Tests and GSM8K ran at `ff9dd45e8b` and `34fa2bdf52`.
  - The c=16/32 serving runs used `34fa2bdf52`.
- **RTX 3090:** ran `34fa2bdf52`.
- **Changes after `34fa2bdf52`:**
  - The GDDR IQ2_XXS dense cutoff moved from 128 to 65. A microbenchmark on the 5060 Ti, 3090 and L40S shows dequant 1.4–2.1x faster at M=96. This only affects M = 65–127.
  - A comment-only cleanup that also removes an always-on constexpr flag.
- **H200:** the round before the SM90 single-row Q8 GEMV was added (see TODO).

</details>

## Details

### Design

**Dispatch inside custom ops.** The choice that depends on M (rows, i.e. tokens in the batch) happens inside custom ops (`modelopt_block_linear` and the MoE experts), so torch.compile cannot freeze it. The IQ2 grid lookup table is built at load time, so CUDA graphs capture.

**Dense layers:**
- A GEMV for small M.
- A fused-decode `tl.dot` GEMM with deterministic split-K.
- From a per-codec M onward, dequantize then cuBLAS, as AWQ does. The dequantized copy is capped at 2^26 elements.

| Path | HBM | GDDR | Turing |
|---|---:|---:|---:|
| IQ2_XXS dense: dequant + cuBLAS from M = | 8 | 65 | 32 (roofline estimate) |
| Q8_0 dense: dequant + cuBLAS from M = | 64 | 256 | 32 (roofline estimate) |
| MoE GEMV up to M = | 8 | 8 | 8 |

- **Memory class:** a GPU counts as HBM when its memory bus is at least 1024 bits wide. The dequantized copy is a write plus a read of 2 bytes per weight, so the crossover follows memory bandwidth.
- **Why 65 for IQ2 on GDDR:** past one 64-row M tile the fused IQ2 GEMM doubles in cost. On the 5060 Ti it goes from 213 to 425 µs at M=65, against 254 µs for dequant.

**MoE:**
- A `TritonExperts` subclass. Up to 8 routed rows, one program per row: direct routing without `moe_align_block_size`, with ReLU² fused into the first GEMM using the same BF16 rounding as the unfused path. Above 8 rows, upstream `fused_moe_kernel`, which decodes the packed tiles in its K loop (`BLOCK_CODEC`).
- The cutoff of 8 came from serving, not microbenchmarks. On the A100-class GPU and the L40S, a kernel sweep with uniform routing keeps the GEMV ahead up to M=128, but serving disagrees:
  - On the A100-class GPU, a cutoff of 64 made median TPOT 15% worse at c=16 and 27% worse at c=32.
  - On the L40S, a cutoff of 8 gave 4–12% more tok/s at c=16/32.

**Other parts:**
- Q8_0 embedding lookup, through compressed-tensors' `_dequant_gather_kernel`.
- On Hopper, single-row Q8_0 on four exact shapes uses one wider-K GEMV kernel.
- Batch-invariant mode (`VLLM_BATCH_INVARIANT=1`): dense layers dequantize, or use one tile shape without split-K; MoE uses upstream's batch-invariant config.
- The kernels follow the activation dtype (BF16 or FP16). b12x is BF16-only, so `--dtype half` uses Triton on every GPU.
- Docs: new section "ModelOpt block codecs with Triton kernels (SM75+)" in `docs/features/quantization/modelopt.md`.

### Complexity

About 1,230 non-test lines under `vllm/`:

| File | Lines |
|---|---|
| Kernels (`modelopt_block_quant_triton.py`) | 721, of which 66 are the IQ2 grid table |
| Warmup | 188 |
| MoE experts class | 108 |
| `modelopt_block_quant.py` | +111/−28 (backend selection and wiring) |
| `fused_moe.py` | +89/−4 (`BLOCK_CODEC` decode in `fused_moe_kernel`) |
| `compressed_tensors_embedding.py` | +10/−3 (`UNPACKED_INT8`, int64 offset) |
| Warmup hook | +4 |

**Why it's worth it:**
- It makes an NVIDIA checkpoint family usable on every NVIDIA GPU that is not SM12x.
- Most of it lives in new files. Existing code changes: backend selection in `modelopt_block_quant.py`, one warmup hook, a `BLOCK_CODEC` branch in `fused_moe_kernel` (compiled out for every other caller), and a flag on the compressed-tensors gather kernel.
- The alternative is to dequantize to BF16 at load. That needs about 148 GB for Bolt's 74B parameters, against 21.7 GiB packed.

### FP16 dependency, fixed in a separate PR

- `MoERunner._maybe_apply_routed_scale_to_output` divides the shared-expert output by `routed_scaling_factor` (5 for Bolt) in FP16 and expects the model to compensate.
- `nemotron_h` does not compensate, so FP16 Bolt collapses to 3/250 on any backend.
- The fix and its regression test are kept out of this PR.

### Limitations and TODO

- [ ] **Human review:** tests, benchmarks, diff and maintainability.
- [ ] **H200 rerun at this commit:** the SM90 single-row Q8 GEMV was added after the last H200 serving and GSM8K run. If it doesn't hold up, drop that dispatch.
- [ ] **SM75 hardware run** (T4). Otherwise, narrow the claim to SM80+.
- [ ] **GPU sync check:** run `VLLM_GPU_SYNC_CHECK=error` on the affected configuration.
- **Q8 cutoff on large-L2 GDDR GPUs** (L40S: 96 MiB L2, 142 SMs). Classifying by bus width keeps them on the GDDR value, which is up to 1.44x off the best path at M=128 (`fc2_latent`). This is a follow-up.
- **`lm_head` never dequantizes:** at 2^29 elements it is above the 2^26 cap. It runs 1.6–3.3x slower than BF16 once 64–128 or more sequences are batched. Slicing the dequant would fix this; follow-up.
- **MoE prefill:** against BF16 `fused_experts` on dequantized weights (RTX 5060 Ti, Bolt expert shapes), the packed path runs at 0.81–0.84x speed for M = 64–256, 0.98–1.0x at M = 1024 and 0.63–0.65x at M = 4096. Up to M = 16 it is 1.3–3.2x faster. That is the cost of keeping weights packed. H200 numbers to follow.
- **Out of scope:** IQ2_XS still requires b12x. ROCm is not supported. Routed experts support tensor parallelism only, as with b12x.
- **CI:**
  - The kernel tests run in the existing L4 quantization kernel shards. They take 2–4.5 minutes on the tested GPUs, against a 30-minute timeout. That step now also triggers on `fused_moe.py` and the experts file.
  - No new CI step.
  - The full model does not fit CI's L4 (24 GB), so it is not exercised in CI. Full-model validation was done outside CI, as above.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

<!-- ===== Everything below is NOT part of the PR body. Post it as a PR comment after the next push. ===== -->

## Update comment (2026-11-09)

**Update:** rebased onto `main` (`8db73f26c1`) and still squashed into one commit. This revision reuses more existing vLLM code, adds no new CI step, and reworks the tests.

**Reuse instead of new kernels.** About 1,230 new non-test lines under `vllm/` now, down from about 1,340.
- **MoE prefill:**
  - The PR's own grouped GEMM (`_moe_gemm_kernel`) is gone. Upstream `fused_moe_kernel` gets a `BLOCK_CODEC` constexpr that decodes IQ2_XXS / Q8_0 tiles in its K loop. It defaults to -1, which compiles the branch out for every other caller.
  - Routing uses `_prepare_expert_assignment` and the output goes through `_finalize_expert_output`.
  - Decode (up to 8 rows) still uses the PR's `_moe_gemv_kernel`.
- **Q8_0 embedding:** the PR's own lookup kernel is gone. It reuses compressed-tensors' `_dequant_gather_kernel` with a new `UNPACKED_INT8` flag, and that kernel's output offset is now int64.
- **Dense:** the GEMV and GEMM kernels are merged into one `_linear_kernel`. Split-K sums use `_fixed_order_sum_kernel` from batch-invariant mode.
- **Removed** the opt-in `modelopt_iq2_dense_cache` option.

**Changes to shared code:**
- `fused_moe.py`: the `BLOCK_CODEC` branch, plus `do_not_specialize=["EM", "num_valid_tokens"]` on `fused_moe_kernel`, so a new batch size doesn't trigger a recompile. On upstream BF16 MoE (Mixtral, Qwen3-30B-A3B and DeepSeek-V2-Lite shapes, M = 1–4096) it is within 0.2% of the specialized kernel, with identical outputs. `tests/kernels/moe/test_moe.py` passes (675).
- `compressed_tensors_embedding.py`: `UNPACKED_INT8` and the int64 offset.

**Batch-invariant mode** is supported now:
- Dense layers dequantize and use `x @ W.T`, which vLLM already makes invariant. Above the dequant size cap, they use one tile shape with no split-K.
- MoE uses upstream's batch-invariant config.
- A test checks bit-identical rows across batch sizes.

**Tests, reworked** (`tests/kernels/quantization/test_modelopt_block_triton.py`):
- **Hand-checkable IQ2_XXS:** one block built byte by byte, with every expected weight written out in the test. It covers grid entries 0, 1 and 255, the sign parity bit, scales s = 0, 1, 2 and 15, and a negative `d`. It goes through the decode, a GEMV, and a 2-expert MoE.
- **Grid table:** checked against gguf-py's by sha256, so the check no longer needs `gguf` and never skips.
- **Bolt shapes:** qkv, fc2_latent and the shared expert at M = 1, 8, 64 and 512, plus one MoE layer (E = 240, top-k 9). All stay within 1.5x of cuBLAS's error on the dequantized weights, and so does the stacked shared expert (up → ReLU² → down).
- **Non-finite scales:** they stay in their own rows.
- **Dropped:** the tiny Nemotron-H engine test. It ran in no CI job, and with random weights one prompt's eager vs compiled top-5 differed at the first token on H200. `test_modelopt.py` is back to a 3-line diff.

**Benchmark:** timing is now a CUDA-graph replay with L2 flushed before each run, taking the best of 3 captures. The table adds `bytes_ratio` (the ideal speedup) and `%copy` (achieved bandwidth vs a device copy).

**CI and docs:**
- **No new CI step:**
  - Dropped the new L4 step and restored the `test_modelopt.py` line in the test-tethering allowlist.
  - The backend-selection test moved into the kernel test file, which the existing L4 quantization kernel shards already run.
  - That step now also triggers on `fused_moe.py` and the experts file.
  - Dropped the dense-cache tests.
- **Docs:** moved from `b12x.md` to a new section in `modelopt.md`, "ModelOpt block codecs with Triton kernels (SM75+)".

**Context:** b12x landed in #59604 (a republish of #59486), merged 2026-10-13. This PR runs the same checkpoints on non-SM12x GPUs.

**Tests at this revision (RTX 5060 Ti):**
- `test_modelopt_block_triton.py`: 221 passed.
- `test_modelopt.py -k "block or iq2"`: 44 passed.

**Not yet rerun at this revision:** full-model GSM8K and serving. The tables in the description come from the previous revision, which used the old MoE prefill kernel; the H200 run also had the dense cache on. I'll rerun on H200 and an Ada GPU and update the tables.

**Question for reviewers: should Q8_0 dense layers go through `choose_mp_linear_kernel`?** #65272 does this for INT8. For Bolt's Q8_0 shapes (qkv, o_proj, fc1/fc2_latent, lm_head) it picks Marlin (`uint8b128`, group 32, SM75+). I checked that path on a 5060 Ti, with BF16 activations and CUDA graphs:
- **Speed** (Triton time divided by Marlin time; below 1 means Triton is faster):
  - M ≤ 16: 0.91–1.04 (a tie).
  - M = 64–256: Marlin is 1.04–1.25x faster on qkv, o_proj and lm_head.
  - M = 512–1024: 0.83–0.99 (Triton slightly ahead) on qkv, o_proj and the latents; Marlin is 1.23x faster on lm_head.
- **Accuracy** (relative error against the exact product): with BF16, Marlin 2.8e-3 against 1.6–2.3e-3 for Triton, because Marlin rounds the FP16 Q8_0 scales to BF16. With FP16 both match.

If you prefer that route, I'll move Q8_0 dense and lm_head to Marlin and drop the matching Triton path. The Q8_0 embedding and expert paths would stay.
