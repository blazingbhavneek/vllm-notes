# 05 · Handoff: finish the Bolt Triton path on the H200

Written 2026-10-27, after round 1 on the home box (RTX 5060 Ti, sm_120). This is for the next agent. Read in this order:

1. This file.
2. `04-progress-log.md`: results, deviations, and the experiment log. Keep appending to it.
3. `03-improvment-handoff.md`: the original plan. §0.2 has the hard rules, §2.2 the H200 bars, §4–§9 the phases and gates. Its §4.2 start point is out of date; use §1 below.
4. `review.md`: the draft's findings (B1–B4, K1–K5, S1–S4).

## 1. State

- **Code:** the fork `blazingbhavneek/vllm`, branch `feature/modelopt-block-triton`, pushed and up to date.
  - `HEAD` `f513496e9b` = the rewrite.
  - `HEAD~1` `b477391ff3` = the old draft (`2064cc2efb`), rebased. Use it as the A/B baseline.
  - Base = `upstream/main` `274a5e19a1`.
- **Home box:** checked out directly in `/mnt/common/Code/vllmServer/vllm`, on its in-place build. There is no extra worktree.
- **Phases:** 1 and 2 done. 3 started (first tuning round on the 5060 Ti). 0, 4 and 5 not started. All three of those need the H200.
- **Passing on sm_120:**
  - 76 kernel tests (1 skip: `gguf` not installed).
  - 71 ModelOpt repo tests, including the tiny Nemotron-H e2e.
  - 51 b12x tests.
  - Ruff clean.
  - Every dispatched config fits sm_80–sm_120, with zero spills on sm_90.
  - CUDA graphs capture (B4 fixed).
  - The compiled 1-token decode picks the GEMV (B2 fixed).

**Open must-fix items, in priority order:**

| # | Item | Why it blocks | Where |
|---|---|---|---|
| 1 | `_gemm_kernel` JIT-compiles during inference | Fails §2.1 "no runtime JIT after Graph capturing finished" | §5.3 |
| 2 | Dequant+cuBLAS path breaks hard rule §0.2.2 | The plan's own rule; reviewers may object | §5.4 |
| 3 | Dense M = 8–64 slower than bf16 (IQ2 from M=8, Q8 at M=64, fc2_latent from M=16) | §2.2 merge bar "M = 2…64 faster than cuBLAS" | §5.5 |
| 4 | Nothing measured against the draft yet | Half of the §2.2 bars are relative to the draft | §5.1 |

## 2. Code map (branch `feature/modelopt-block-triton`)

`vllm/model_executor/layers/quantization/modelopt_block_quant_triton.py`:

| Line | Symbol | What it does |
|---|---|---|
| 14 | `_IQ2XXS_GRID` | ggml grid, 256 × 8 magnitudes, each one of {8, 25, 43} |
| 85 | `repack_blocks` | Load-time repack: 66 B / 34 B blocks become an aligned payload plus fp16 scales. The total size is the same (rule §0.2.2). |
| 103 | `iq2_lut` | One int32 tensor: [0:2048] fp32 magnitudes stored as bits, then [2048:2304] 2-bit code words (one 16-bit word per grid row) |
| 115 | `_decode_tile` | Decodes a [BN, BK] fp32 weight tile. `GATHER` picks between an LUT gather (fast in blocked layouts) and 2-bit codes (fast in `tl.dot` operand layouts) |
| 186 | `_gemm_kernel` | `tl.dot` GEMM with codes decode and split-K (fp32 partials) |
| 231 | `_gemv_kernel` | GEMV with LUT gather. `BM == 1` takes a 1-D branch (the 3-D form is about 35% slower); `BM` 2/4 decodes once for all rows |
| 282 | `_dequant_kernel` | Packed weights to bf16 [N, K], with LUT gather |
| 298 | `_moe_gemm_kernel` | Grouped MoE GEMM over `moe_align_block_size` output, with codes decode |
| 363 | `_q8_embedding_kernel` | Q8_0 row gather plus decode |
| 380 | `_linear_config` | **Dense tile rules** (table below) |
| 409 | `launch_linear` | Launches GEMV (`BM < 16`) or GEMM, then sums the split-K partials. The benchmark and tools call it with an explicit config |
| 466 | `DEQUANT_MIN_M`, `DEQUANT_MAX_NUMEL` | **Dequant+cuBLAS switch-over:** IQ2 64, Q8 256, and only when N·K ≤ 2^26 |
| 470 | `_modelopt_block_linear` | The body of the custom op `torch.ops.vllm.modelopt_block_linear`. All M-dependent dispatch happens here, outside the traced graph (rule §0.2.6) |
| 519 | `_modelopt_block_embedding` | The custom op `torch.ops.vllm.modelopt_block_embedding` |
| 552 | `moe_config` | **MoE tile rules** (table below) |
| 567 | `invoke_block_moe_kernel` | Launches one grouped GEMM |

Other files:

- `vllm/model_executor/layers/fused_moe/experts/modelopt_block_triton_moe.py`: `ModelOptBlockTritonExperts(TritonExperts)`, 96 lines. `apply` = align, GEMM1, activation, GEMM2, `moe_sum`.
- `vllm/model_executor/layers/quantization/modelopt_block_quant.py`:
  - `_use_triton` (line 46) picks b12x or Triton, or raises `ValueError` in `__init__`.
  - Triton branches: MoE `__init__` 101 and `process_weights_after_loading` 250 / 268; linear `__init__` 318, `process_weights_after_loading` 366, `apply` 479, `embedding` 501.
  - Rule §0.2.1: with `auto` and b12x available, it returns False before anything else.
- Tests:
  - `tests/kernels/quantization/test_modelopt_block_triton.py`: bit-exact decode, linear vs reference at many M, offset views, int64 offsets, compile-does-not-freeze-dispatch, CUDA-graph replay, MoE vs bf16 reference (≤ 2× its error), MoE graph with new routing, embedding.
  - `tests/quantization/test_modelopt.py`: `test_modelopt_block_quant_backend_selection` (line 1630) and `test_modelopt_block_triton_tiny_nemotron_h` (line 1670).
- `benchmarks/kernels/benchmark_modelopt_block_triton.py`: dense shapes (Appendix C), MoE 240x2048 / 432x2560. `--layer`, `--m`, `--sweep`.
- `docs/features/quantization/b12x.md`: the section "Other NVIDIA GPUs (Triton kernels)".

**Dense dispatch today** (`_modelopt_block_linear` → `_linear_config`):

| M | Path | Tile (BM/BN/BK, warps) |
|---|---|---|
| 1 | GEMV | IQ2 1/8/256 w1; Q8 1/32/256 w4 |
| 2–4, IQ2 only | multi-row GEMV | BM = next_pow2(M), 8/256 w1 |
| ≤ 16 | GEMM | 16/64/128 w4 |
| ≤ 64 | GEMM | 32 or 64 /64/128 w4 |
| > 64 | GEMM | 128/128/64 w8 if smem ≥ 160 KB (H200: yes), else 64/128/64 w8 |
| ≥ `DEQUANT_MIN_M` and N·K ≤ 2^26 | dequant then `x @ W.T` | `_dequant_kernel` IQ2 BN8 w1, Q8 BN32 w4 |

Split-K: double `split` while `split < 16`, `tiles*split < 2*num_sms` and `(K//bk) % (2*split) == 0`.

**MoE** (`moe_config`): `per_expert = M*top_k/E`. BM is the first of 16/32/64 that is ≥ per_expert, else 128. Then BN = 128 if BM == 128 else 64; BK = 128 if BM ≤ 32 else 64; warps 8 if BM == 128 else 4; stages 2.

## 3. Tuning rules

- Keep the configs as small rules like the ones above. No per-shape tables (plan §7).
- Every config you add must print `ok` in `kernel-tests/rewrite/dispatch_report.py`: it must fit shared memory on sm_80/86/89/90/100/120 with no sm_90 spills (rule §0.2.7). Edit the config lists in that script to match the code first.
- `GATHER` is not a launch argument. Each kernel passes it in its own `_decode_tile(...)` call: GEMV and dequant use gather, GEMM and MoE use codes. To try the other decode on the H200, flip it in that call and re-run the benchmark. The H200 has a different balance of compute to bandwidth, so re-check this once.
- Correctness after any change: `python -m pytest tests/kernels/quantization/test_modelopt_block_triton.py -q`.

## 4. Tools (this folder)

Run every Python tool from the vllm repo root, with the branch's venv active.

| Tool | Use |
|---|---|
| `benchmarks/kernels/benchmark_modelopt_block_triton.py` (in the repo) | Main table: ours vs bf16, ratio, GB/s, config. `--layer shared_up --m 8 16 --sweep` searches tiles (GEMV BM for M ≤ 8, GEMM BM 16–128, split, warps, stages; MoE BM/BN/BK/warps/stages) and prints the best |
| `kernel-tests/rewrite/fused_vs_cublas.py [M ...]` | For M > 64: fused GEMM (forced BM64 and BM128) vs dequant+cuBLAS vs cuBLAS. **This is the §5.4 decision tool** |
| `kernel-tests/rewrite/dispatch_report.py` | Appendix D cross-compile check of every dispatched config |
| `kernel-tests/rewrite/moe_tolerance.py` | MoE error / bf16-reference error over 20 seeds (5060 Ti: median 0.84–1.03, max 1.64; the test allows 2) |
| `kernel-tests/rewrite/run_e2e.sh <out_dir>` | Tiny e2e: Triton and bf16 control in eager / compiled / graphs, then `compare`. Its last line reports any JIT during inference |
| `kernel-tests/e2e_tiny_bolt.py` | The e2e driver. Its kernel-name filter now matches both the draft's and the rewrite's kernels |
| `kernel-tests/harness.py`, `bench.py`, `test_*.py`, `proto_kernels.py` | Draft-era tools, for the Phase 0 draft numbers (plan §4.7) |
| `kernel-tests/rewrite/results/*_5060ti.txt` | Round 1 raw output: benchmark, dispatch report, e2e compare, e2e JIT lines |

## 5. H200 plan, in order

Follow the work-box etiquette in plan §0.2.9 every time:
- Never touch port 8000 or `gemma-4-31B`.
- Check free GPU memory first (§4.1).
- Run A and B back to back.

### 5.0 Setup

```bash
cd <fork checkout at work>
git fetch origin && git switch -c feature/modelopt-block-triton --track origin/feature/modelopt-block-triton
git remote add upstream https://github.com/vllm-project/vllm.git 2>/dev/null; git fetch upstream
git rebase upstream/main          # conflict-free twice so far; skip if upstream hasn't moved
git worktree add ../vllm-draft HEAD~1    # the draft, for Phase 0
```

Then create the venv as in plan §4.3 (a new `.venv`; don't modify an existing one). Also install `gguf`, so `test_iq2_grid_matches_ggml` runs instead of being skipped.

Duplicate check (AGENTS.md, mandatory): run plan §4.2's `gh` commands and paste the output into the log. If an open PR already does this, stop and tell the user.

### 5.1 Phase 0: draft baselines (plan §4.4–§4.8)

- Serve Bolt from `../vllm-draft`.
- Record: GSM8K-250, `vllm bench serve` at c = 1/8/32/128, startup time, and the draft kernel times (`kernel-tests/bench.py`).
- These are the "draft" columns of §2.2. Gate 0.

### 5.2 Gates 1 and 2 on the rewrite

- Gate 1: the kernel tests (also on the RTX 4000 Ada if it's free), then `dispatch_report.py`.
- Gate 2:
  - The repo tests: the `-k "block or iq2 or mixed_precision or triton"` set, plus the b12x tests.
  - `run_e2e.sh`.
  - Bolt serve: two `Using Triton kernels` lines and no `sm90` in the log.
  - Quick GSM8K-250.

### 5.3 Fix the JIT during inference (must-fix 1)

**Finding.** In the home e2e, vLLM's JIT monitor warns `Triton kernel JIT compilation during inference: _gemm_kernel` in both compiled and graphs mode. The bf16 control doesn't. See `results/e2e_jit_5060ti.txt`.

Why it happens:
- Warmup only runs the capture sizes (1/2/4 there) and `max_num_batched_tokens` (256, which takes the dequant path).
- A prefill chunk of 5–63 tokens then hits a GEMM config (BM16/32/64 and a split) that was never compiled.
- The first FULL graph capture also took 43 s with Triton, against 1 s for bf16. That is first-time compile time, and it counts against the "+60 s over the draft" startup bar.

**Fix: add a warmup like the existing ones.**
- Pattern: `vllm/model_executor/warmup/mamba_triton_warmup.py` (53 lines), called from `kernel_warmup()` in `vllm/model_executor/warmup/kernel_warmup.py` (line ~199).
- `b12x_warmup.py` shows how to find the layers: walk `model.modules()` and look for something the quant method set.

Sketch:
1. For each layer with `quant_method.use_triton`, collect the distinct `(codec, N, K)`. Bolt has few.
2. For each one, take the set of distinct `_linear_config(M, ...)` results for M = 1 … `max_num_batched_tokens`. This is cheap Python.
3. Run `_modelopt_block_linear` once at one M per distinct config, plus once at M ≥ `DEQUANT_MIN_M` for the dequant kernel.
4. For MoE, run the experts once per distinct `moe_config` BM bucket. Also cover the embedding.
5. Log "Warmed N ModelOpt block Triton configs".

Then:
- Re-run `run_e2e.sh`. It must print `no JIT during inference`.
- Add an assertion to `test_modelopt_block_triton_tiny_nemotron_h` if that's cheap.
- Watch Triton specialization: `M` is `do_not_specialize`, but `N`, `K // split` and `x.stride(0)` are specialized on divisibility by 16. Warm with the real shapes.

This can also be done on the home 5060 Ti.

### 5.4 Settle the §0.2.2 conflict (must-fix 2)

```bash
python ../Notes/02-Bolt-Support/kernel-tests/rewrite/fused_vs_cublas.py 96 128 256 512 2048 8192
```

- **Fused BM128 ≤ 2.0 at M ≥ 2048 on every shape** (the merge bar):
  - Delete the dequant path: `DEQUANT_*`, the branch in `_modelopt_block_linear`, and the benchmark's dequant label.
  - Keep `dequantize_blocks`, which the tests and benchmark use as their reference.
  - Check `"wgmma" in ck.asm["ptx"]` for the BM128 config. If wgmma is missing, try `num_stages` 3 and BK 64/128 with `--sweep`.
- **Otherwise:**
  - Keep it.
  - Set `DEQUANT_MIN_M` to the crossover the tool shows.
  - In the PR, cite AWQ (`auto_awq.py`, `num_tokens >= 256`), the 128 MiB transient cap, and the numbers.
- Log the decision and the table in `04-progress-log.md`.

### 5.5 Phase 3 tuning (must-fix 3)

1. Run the full benchmark and paste it into the log next to the draft numbers.
2. For each row that misses §2.2, run `--layer <row> --m <M> --sweep`. On the 5060 Ti these missed:
   - IQ2 M = 8–64 (1.14–2.06×).
   - Q8 M = 64 (o_proj 1.54, fc1 1.15, fc2 1.84).
   - fc2_latent M = 16 (1.04).
3. Things to try, cheapest first:
   - Raise the IQ2 GEMV bound past M = 4 (BM 8/16 GEMV; the sweep covers BM ≤ 8).
   - Retune the M ≤ 64 GEMM rows.
   - Try GATHER vs codes in the GEMM (§3).
   - Move `DEQUANT_MIN_M`.
4. MoE: the merge bars are M ≤ 8 vs the draft (≥ 1.5×) and M = 4096 vs bf16 (≤ 2.5×; it was 1.8 at home). Sweep the MoE rows too.
5. After each change: kernel tests, `dispatch_report.py`, and a log entry in the plan §0.1 format.

### 5.6 Phase 4 (plan §8)

GSM8K in full, A/B against the draft; §8.2 layer parity on real weights; the JIT monitor (must be clean after §5.3); `vllm bench serve` c = 1/8/32/128 vs the draft (≥ 1.3× at c = 1); startup time vs the draft (≤ +60 s).

### 5.7 Phase 5

1. Squash `HEAD~1` and `HEAD` into one commit with `git commit -s`, plus the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
2. `pre-commit run --all-files` on the changed files.
3. Run `.agents/skills/pr-checklist/SKILL.md` and fix what it finds.
4. Ask the user before pushing or opening the PR.

PR body (AGENTS.md requires each part):
- Summary: fixes #62609; Triton path for sm80+; b12x unchanged on SM12x.
- Why it isn't a duplicate (the §5.0 `gh` output).
- Design: custom-op dispatch, packed weights, the two decodes, and the dequant path if kept, with its justification.
- Test commands and results.
- Benchmarks (§2.2 table, H200).
- GSM8K A/B and `bench serve`.
- A statement that AI assistance was used.
- End with "🤖 Generated with [Claude Code](https://claude.com/claude-code)".

## 6. Gotchas learned in round 1

- The tiny e2e LLM needs `enable_chunked_prefill=True` ("Chunked prefill is required for Mamba prefix caching").
- Token streams differ even for bf16 eager vs compiled, so compare with `close=True` and never require exact equality.
- Benchmark timing: `bench_gpu_time_with_cupti(fn, use_cuda_graph=True)`. CUPTI still flushes L2 per iteration in graph mode.
- The `int64` offset test needs M = 16400 on lm_head. It's slow but must run.
- The MoE test tolerance is relative to a bf16 reference, and every test is seeded. A fixed 1% flaked on ReLU².
- Lint: `uvx ruff@0.14.0 check` and `format`, lines ≤ 88. Long kernel launches use `# fmt: skip`. The grid table sits inside `# fmt: off/on`.
- Cross-compile: `triton.compile(ck.src, target=GPUTarget("cuda", cc, 32))`. Use `ptxas-blackwell` for sm_100.
- Home box only (ignore at work): caches on tmpfs, short `TMPDIR`, `~/.venvs/vllmNotes`, never `pkill -f`.

## 7. Tried and rejected (don't redo without new evidence)

| Idea | Result on the 5060 Ti |
|---|---|
| Codes decode everywhere | M=1 IQ2 GEMV 38 → 74 µs (SASS 848 → 1704 instructions) |
| LUT gather inside `tl.dot` kernels | Scalar gathers; MoE 1.6× bf16 (codes: 0.92×) |
| Grid words + 2^23 float trick | Worse on every kernel |
| int→float convert (I2F) decode | Worse than codes |
| 3-D multi-row GEMV used for M=1 too | 38 → 52 µs, hence the `BM == 1` branch |
| Fused GEMM only at M = 2048 | 2.3–2.9× cuBLAS, hence the dequant path (§5.4 re-checks this on the H200) |
| `moe_config` gate on smem ≥ 160 KB | Not needed; the largest tile is 57 KB with codes |
| Split-K only for M ≤ 64 | fc1_latent M=128: 82 µs vs 44 µs with split-K |
