# Round 8: multi-arch validation and PR evidence (2026-11-01)

This note collects every measured result for the two upstream PRs:

1. **FP16 bugfix PR**, from branch `fix/nemotron-h-fp16-routed-scale` (`c2e927bf4a`).
2. **Triton PR** (closes vllm#62609), from `feature/modelopt-block-triton` at head `e17acbd246`.
   When cutting this PR, leave out `ff9dd45e8b`, which is the FP16 fix cherry-picked.

Raw files are in [`kernel-tests/rewrite/results/round8_multi_arch/`](kernel-tests/rewrite/results/round8_multi_arch/).
- **CMP 170HX, L40S, RTX 3090:** results came back from rented vast.ai boxes. Each box ran its own agent from a local
  `handoff.md` test plan, which was never committed and was deleted after round 8.
- **5060 Ti:** run on the home box.
- **H200:** numbers come from rounds 5 and 6 ([06](06-h200-gpu1-optimization.md), [07](07-round6-serving-profile.md)).

## Commits

| Commit | Change | Runs on it |
|---|---|---|
| `f5e6e1bb0b` | Round 7: SM75 + FP16 support, GDDR-aware cutoffs | CMP 170HX, L40S (first runs) |
| `ff9dd45e8b` | FP16 fix, cherry-picked from `c2e927bf4a` | CMP 170HX, L40S (reruns) |
| `34fa2bdf52` | MoE GEMV up to 8 rows on every GPU; warmup import order; dense-cache test uses `check_logprobs_close` | L40S (confirmation), RTX 3090, 5060 Ti |
| `e17acbd246` | GDDR IQ2_XXS dequant from M=65; test removed from the tethering allowlist | 5060 Ti |

**Why older-commit results still describe head:**
- **HBM GPUs (CMP 170HX, H200):** dispatch has not changed since `f5e6e1bb0b`. MoE GEMV was already 8, IQ2 is 8 and Q8 is 64.
- **GDDR GPUs:** `34fa2bdf52` differs from head only in the IQ2 cutoff (128 → 65).
  - That affects only the shared-expert layers at M = 65–127, which means prefill chunks or batches of more than 64
    sequences.
  - Decode at `--max-num-seqs 32` never reaches it.
  - It is validated by microbenchmark on three GPUs and by the 5060 Ti tests, not by serving.
- **H200 is the exception.** Its last full-model run came before `f1ec2fb051` added the SM90 single-row Q8 dispatch,
  which is active at head (see Gaps).

## GPUs

| GPU | SM | Memory | SMs | L2 | Copy BW | VRAM | Full model? |
|---|---|---|---:|---:|---:|---:|---|
| H200 NVL | 90 | HBM3e | 132 | 50 MiB | contended box | 141 GB | yes |
| CMP 170HX (A100-class) | 80 | HBM, 4096-bit | 74 | 32 MiB | 1483 GiB/s | 63.4 GiB | yes |
| L40S | 89 | GDDR6, 384-bit | 142 | 96 MiB | 609 GiB/s | 44.4 GiB | yes |
| RTX 3090 | 86 | GDDR6X, 384-bit | 82 | 6 MiB | 786 GiB/s | 23.6 GiB | eager only (see 24 GB fit) |
| RTX 5060 Ti | 120 | GDDR7, 128-bit | 36 | 32 MiB | 385 GB/s | 16 GiB | no (tests, microbenchmarks) |
| T4 | 75 | — | — | — | — | — | not tested: cross-compile + ptxas only |

**Notes on the hardware:**
- **The CMP 170HX is not a stock A100.** The box reports itself as the GA100 mining part, with 74 SMs where an A100 has
  108. NVIDIA reportedly caps compute on CMP cards, so its timings understate a real A100.
- **Software on the rented boxes:**
  - PyTorch 2.14.0+cu132 and Triton 3.8.0.
  - Driver 615.71.09 on the CMP 170HX; 595.99.02 on the L40S and 3090.
- **5060 Ti:** SM120 runs b12x by default, so its tests exercise the Triton kernels directly.

## Correctness

| GPU | Commit | Kernel suite | `test_modelopt.py -k block` | Dense-cache parity | Warmup, no JIT at inference |
|---|---|---|---|---|---|
| CMP 170HX | `f5e6e1bb0b` | 192 passed, 2 min 51 s | 55 passed, 9 min 12 s | Exact compare flaked 1 of 3 | BF16 + FP16 OK |
| CMP 170HX | `ff9dd45e8b` | 192 passed | 55 passed | `check_logprobs_close` version 10/10 | — |
| L40S | `f5e6e1bb0b` | 192 passed, 259 s | 54 passed, 1 failed (the dense-cache flake) | — | — |
| L40S | `ff9dd45e8b` | 192 passed, 189 s | 55 passed, 423 s | 3/3 | BF16 + FP16 OK |
| L40S | `34fa2bdf52` | 192 passed | 55 passed | — | — |
| RTX 3090 | `34fa2bdf52` | 192 passed, 124 s | 55 passed | 3/3 | both scripts OK |
| RTX 5060 Ti | `34fa2bdf52` | 191 passed, 1 skipped | 55 passed | included in the 55 | — |
| RTX 5060 Ti | `e17acbd246` | 191 passed, 1 skipped | 55 passed, 264 s | included in the 55 | — |

**What the table shows:**
- **5060 Ti skip:** the int64-index test needs 12 GiB of free memory.
- **CI time:** the kernel suite runs in 2–4.5 minutes on these GPUs, well inside the 30-minute budget of the L4 quantization
  kernel shard.
- **Dense-cache flake:** the old test compared cached and uncached runs exactly, including logprob floats. The failing runs
  had the same tokens with different logprob values.
  - `34fa2bdf52` switched to `check_logprobs_close` plus an output-length check.
  - A numeric tolerance was tried first and failed 1 of 10 on a logprob difference of 0.137.
- **JIT at inference:** across every serving log, the only kernel compiled at inference time was upstream Mamba's
  `_causal_conv1d_fwd_kernel`. None of ours.
- **Real-tensor accuracy (round 3b, H200, [04](04-progress-log.md)):**
  - `dequantize_blocks` is bit-exact on every real checkpoint tensor.
  - Dense error/bar is 0.41–0.67; MoE error/bar is 0.12–0.45.

## Accuracy: GSM8K, first 250 test questions

**Setup:**
- HTTP requests at concurrency 32, greedy decoding, a 768-token limit.
- Plain prompt `Q: …\nA: Let's think step by step.\n`, with no chat template.
- Dataset sha256 `3730d312…`.
- The pass bar is 225/250.

**Noise:** BF16 scored 230, 235, 231 and 232 on code that is identical for BF16. Treat differences of about 5 questions as
noise.

| GPU | Commit | BF16 (truncated) | FP16 (truncated) |
|---|---|---:|---:|
| H200 NVL | round 5 server, before the SM90 Q8 dispatch | 236 (0) | — |
| CMP 170HX | `f5e6e1bb0b` | 230 (3); 229 (3) with MoE GEMV ≤ 64 | **3 (250)** |
| CMP 170HX | `ff9dd45e8b` | 231 (3) | **235 (0)** |
| L40S | `f5e6e1bb0b` | 235 (2) | **3 (250)** |
| L40S | `ff9dd45e8b` | 230 (3); 227 (3) with MoE GEMV ≤ 8 | **234 (1)** |
| L40S | `34fa2bdf52` | 232 | — |
| RTX 3090 | `34fa2bdf52` | 231 (2), eager, 0.98 memory utilization | — |

## Serving throughput

**Workload:** `vllm bench serve` with random prompts, 128 tokens in and 256 out, temperature 0, seed 0.
- Request counts: 12 at c=1, 32 at higher concurrency.
- Server settings: compiled with CUDA graphs (capture sizes 1–32), `--max-num-seqs 32`, `--max-model-len 4096`,
  `--max-num-batched-tokens 1024`, Marlin for NVFP4.
- H200 round 6 used the same workload.
- Each run is single, so expect about ±5% between runs.
- Cells show output tok/s, with median TPOT in ms in parentheses.

| GPU | Commit | c=1 | c=8 | c=16 | c=32 |
|---|---|---:|---:|---:|---:|
| H200 NVL | round 5 server | 132.1 (7.12) | 492.0 (15.05) | — | 865.0 (34.05) |
| CMP 170HX | `f5e6e1bb0b` | 80.5 (11.62) | 218.6 (34.15) | 311.4 (48.10) | 394.5 (74.92) |
| CMP 170HX | `ff9dd45e8b` | 80.3 (11.68) | 218.6 (34.18) | — | 402.6 (73.26) |
| L40S | `ff9dd45e8b`, MoE GEMV ≤ 64 | 87.5 (10.60) | 346.9 (20.99) | 459.6 (31.47) | 555.0 (52.75) |
| L40S | `34fa2bdf52`, MoE GEMV ≤ 8 | — | — | 486 / 512 (two runs) | 623 / 615 (two runs) |

**FP16 speed with the fix (`ff9dd45e8b`):**

| GPU | FP16 c=1 / c=8 / c=32 tok/s | BF16 c=1 / c=8 / c=32 tok/s |
|---|---|---|
| CMP 170HX | 81.1 / 219.4 / 395.0 | 80.3 / 218.6 / 402.6 |
| L40S | 91.9 / — / 548.7 | 87.5 / 346.9 / 555.0 |

FP16 runs at BF16 speed. Before the fix, FP16 at c=32 on the CMP 170HX was 25% slower (94 ms vs 75 ms TPOT), because
every answer looped to the token limit.

**Startup:**
- Loading the weights takes 21.7 GiB on every box.
- CMP 170HX: about 4 minutes in total, including 20 s of loading and 31 s of `torch.compile`; the KV cache holds
  273k tokens.
- L40S: 290–340 s.

## FP16 bugfix PR evidence

**Symptom:**
- In FP16, GSM8K drops to 3/250 on both SM80 and SM89, with every answer stopped at the length limit.
- The L40S dispatches dense and MoE layers differently from the CMP 170HX, yet scores the same, so our kernels and cutoffs
  are ruled out.
- On the CMP 170HX at `f5e6e1bb0b`, the 250 FP16 outputs break down like this:

  | Output pattern | Count |
  |---|---:|
  | `<think></think>` pairs from the first token (125 contain nothing else) | 154 |
  | Loops on the prompt scaffold | 48 |
  | Copies the question | 12 |
  | Starts coherently, then derails | 27 |
  | Repeats a restated sentence | 9 |

- The logs show no NaN, inf, overflow message or exception.

**Cause, in upstream:**
- `MoERunner._maybe_apply_routed_scale_to_output` in `fused_moe/runner/moe_runner.py` moved there in upstream #35949.
- In FP16 with shared experts, it divides the shared-expert output by `routed_scaling_factor` (5 for Bolt). It expects the
  decoder layer to multiply that back.
- deepseek_v2, AXK1 and openpangu compensate; `nemotron_h` does not. So Bolt's MoE output came out at 1/5 scale.

**Fix:**
- A new runner argument, `fp16_divide_shared_output`, defaults to True.
- NemotronH passes False through `runner_args`.

**Test:**
- `test_nemotron_h_moe_fp16_matches_bf16` in `tests/models/language/generation/test_hybrid.py`.
- It fails before the fix and passes after, on the 5060 Ti and on the CMP 170HX (1 passed).

**Before and after:**

| GPU | FP16 GSM8K, before → after | BF16 GSM8K, same run |
|---|---|---|
| CMP 170HX | 3 → **235** (0 truncated) | 231 |
| L40S | 3 → **234** (1 truncated) | 230 |

**Not covered:** glm4_moe, laguna and bailing_moe_linear take the same runner path without compensating. They are untested
and unchanged, so mention them in the PR as possibly affected.

## Dispatch cutoffs at head

| Path | HBM | GDDR | Turing |
|---|---:|---:|---:|
| IQ2_XXS dense: dequant + cuBLAS from M = | 8 | **65** | 32 (estimate) |
| Q8_0 dense: dequant + cuBLAS from M = | 64 | 256 | 32 (estimate) |
| MoE GEMV up to M = | 8 | **8** | 8 |

**HBM vs GDDR:** `_is_hbm` checks for a memory bus of at least 1024 bits. The Turing values are a T4 roofline estimate and
are marked `ponytail:` in the code.

### MoE GEMV up to 8 rows: settled by serving, not by the microbenchmark

**The microbenchmark and serving disagree:**
- The MoE microbenchmark routes tokens uniformly at random.
- It puts GEMV ahead of `tl.dot` through M=128 on every GPU: 1.37–2.07x on the CMP 170HX, 1.12x at M=128 on the L40S.
- Serving says otherwise:

| GPU | Compared | c=8 | c=16 | c=32 |
|---|---|---|---|---|
| H200 | round 5 | — | — | 8 chosen ([06](06-h200-gpu1-optimization.md)) |
| CMP 170HX | 8 vs 64, median TPOT | 34.15 vs 34.19 ms | 48.10 vs 55.44 ms (64 is 15% worse) | 74.92 vs 95.13 ms (64 is 27% worse) |
| L40S | 64 vs 8, tok/s | 346.9 vs 345.5 | 459.6 vs 479.1 (+4.2%) | 555.0 vs 592.2 (+6.7%) |
| L40S | `34fa2bdf52` (8) vs old 64, tok/s | — | 486 / 512 vs 460 | 623 / 615 vs 555 |

- **c=8 is a control:** both settings take GEMV at M ≤ 8, and they tie.
- **Likely reason (untested hypothesis):** real routing is skewed, so tokens share experts, which helps the batched GEMM.
- **TTFT:** at c=32 it was 1294 and 1147 ms with 8 vs 1189 ms with 64. The 16% regression seen in the first single run was
  noise.

### Dense crossovers by GPU (microbenchmark)

Each cell is the first sampled M where dequant + cuBLAS beats the fused kernel, for BF16 (FP16 within one grid step unless
noted).

| GPU | shared_up (IQ2) | shared_down (IQ2) | qkv_proj (Q8) | o_proj (Q8) | fc1_latent (Q8) | fc2_latent (Q8) |
|---|---:|---:|---:|---:|---:|---:|
| CMP 170HX | ≤ 8 | ≤ 8 | 96 | 96 | 48 | 48 |
| L40S | ≤ 8 | ties 8–64, clear at 96 | 96 (FP16: 64) | 192 | 96 | 96 (FP16: 48) |
| RTX 3090 | 48 | 96 | 256 | 512 | 128 | 96 |
| RTX 5060 Ti (`e17acbd246`) | 65 | 65 | none up to 512 (FP16: 512) | 512 (FP16: 384) | 80 | 64 |

**IQ2 on GDDR moved 128 → 65:**
- Past one 64-row M tile the fused IQ2 kernel's time doubles, while dequant + cuBLAS barely moves.
- On the 5060 Ti at head:

  | Layer | Fused, M=64 → M=65 | Dequant at M=65 |
  |---|---|---|
  | shared_up | 213 → 425 µs | 254 µs |
  | shared_down | 174 → 381 µs | 257 µs |

- On the 5060 Ti, the dispatched op now picks the best path for IQ2 at every M.
- At M=96, before the change, dequant was 1.5–1.7x faster on the 5060 Ti, 1.4–1.6x on the 3090 and 1.9–2.1x on the L40S.

**Q8 on GDDR stays at 256:**
- The 3090 and 5060 Ti agree on about 256.
- The L40S, with 96 MiB of L2 and 142 SMs, crosses earlier, around 96–192, more like the HBM GPUs.
- Splitting GPUs by bus width therefore puts the L40S in the wrong class for Q8. Its worst case at head is fc2_latent at
  M=128, 1.44x off the best path. That is left as a follow-up rather than adding a third GPU class.

**Worst rows at head on the 5060 Ti:** qkv_proj at M=256 is 1.24x off the best path (BF16), fc1_latent at M=192 1.18x,
fc2_latent at M=128 1.13x. No single Q8 cutoff fixes both the attention and the latent layers.

**HBM Q8 = 64:**
- On the CMP 170HX, cutoffs of 64 and 96 tie at worst case about 1.31x: the attention layers want 96, the latents 48.
- A cutoff of 128 would make it 2.27x.
- Only prefill-chunk tails reach M = 48–96, so per-layer cutoffs are an option, not a fix this PR needs.

## 24 GB fit (RTX 3090)

- **Weights:** they load in 21.7 GiB.
- **CUDA graphs at 0.97 utilization:** startup fails with `ValueError: No available memory for the cache blocks`.
- **Eager at 0.98 utilization**, with `--max-num-seqs 16 --max-num-batched-tokens 1024`:
  - It starts with 0.9–1.0 GiB of KV cache, about 7.2k–7.9k tokens.
  - Decode is unusable: 11.9 / 53.5 / 53.1 tok/s at c=1/8/16, with a mean TPOT of about 83 ms. The L40S does 10.6 ms.
  - The KV cache sits at 100% at c=16.
- **Still pending:** CUDA graphs at 0.98 with `--max-num-seqs 8 --max-num-batched-tokens 512` and capture sizes
  [1,2,4,8]. If it starts, the PR can call 24 GB cards usable; if not, it says 32 GB.
- **OOM warnings:** the loading log shows allocation OOM warnings even on the run that worked.

## Gaps before opening the Triton PR

- **H200 at head.** The last serving and GSM8K runs (236; 132/492/865 tok/s) used the round-5 server.
  - `f1ec2fb051` then added an SM90 single-row Q8 dispatch, which has never been served or scored.
  - Either rerun the H200 at head, or drop that dispatch together with `SINGLETON_GEMV`.
- **SM75.** There is no hardware run: run the kernel suite on a T4, or limit the PR claim to SM80+.
- **`VLLM_GPU_SYNC_CHECK=error`** has not been run. It can run locally on the tiny Nemotron-H e2e.
- **3090 graphs at 0.98:** the 24 GB wording waits on it (see 24 GB fit).
- **Untested GPUs:** B200 (SM100), a real A100, and the L4 (the CI card; same SM89 as the L40S, too small for the full
  model).

## Follow-ups, not in this PR

- **Q8 cutoff for L40S-class GPUs (about 128).** This needs a GPU-class rule beyond bus width.
- **`lm_head` never dequantizes.** It is 2^29 elements, above `DEQUANT_MAX_NUMEL` (2^26), so at M ≥ 64–128 it runs
  1.6–3.3x slower than BF16 on the CMP 170HX and L40S.
  - M here is the number of sequences, so only servers with many concurrent requests hit it.
  - Fix: dequantize in slices.
- **MoE `tl.dot` speed.** It runs 2.3–2.8x behind unquantized BF16 MoE from M=16 up, which mostly affects prefill. That
  is the cost of keeping the weights packed.
- **Triton 3.10** will turn FP fusion off by default (`TRITON_DEFAULT_FP_FUSION`), which may shift the exact-match tests.

## Box setup lessons (for future rentals)

- **The vast image runs its own vLLM service** on the GPU (Qwen3.5-9B, about 20 GiB). Stop it by PID first.
- **Look for the model before installing.** The A100 agent's `find` matched the cache's top-level `models--…` directory,
  missed the existing snapshot, and downloaded another 22.5 GiB. That filled the disk.
- **Install notes:**
  - Use `--no-build-isolation` when disk is short, because the isolated build copies torch (about 5 GB).
  - Add `setuptools_rust` and `tblib`.
  - Check the torch version after every install step: one step bumped it to 2.15.
- **The editable install moves torch to cu132.** That breaks the image's torchaudio (cu130); remove torchaudio or match it.
- **High-concurrency serving tests need `--max-num-seqs` raised.** The handoff's `serve()` caps it at 32, so the L40S
  "c=64/128" dense A/B never decoded more than 32 rows and is invalid.

## Raw files

All paths are under `kernel-tests/rewrite/results/round8_multi_arch/`.

| Directory | Contents |
|---|---|
| `cmp170hx_sm80_f5e6e1bb0b/` | Full first run: `SUMMARY.md`, tests, cutoff sweeps, serving JSON and logs, GSM8K JSON (BF16, FP16, MoE ≤ 64), install logs |
| `cmp170hx_sm80_ff9dd45e8b/` | FP16-fix rerun: `RESULTS.md`, and `local.diff` (the dense-cache test fix that became `34fa2bdf52`) |
| `l40s_sm89/` | `INITIAL_SUMMARY.md` (`f5e6e1bb0b`), `SUMMARY.md` and `updated/` (`ff9dd45e8b`): tests, cutoff sweeps, serving JSON, GSM8K summaries |
| `rtx3090_sm86_34fa2bdf52/REPORT.md` | The 3090 report; the box's raw logs were not downloaded |
| `rtx5060ti_sm120_e17acbd246/bench_cutoff_check.txt` | Dispatched op vs best path at head, M = 32–512, BF16 + FP16 |

**Not saved as files:**
- The L40S `34fa2bdf52` confirmation run arrived only as a screenshot. Its numbers are in the tables above.
- The 5060 Ti test counts at `e17acbd246` were taken from the terminal and not saved.
