# 20 · Implementation plan for the agent

**Who this is for:** the agent (and the human) who will write the code. It turns the design in note 18 into small tasks. Each task says: goal, what it depends on, files, exact steps, tests, and when it's done. Do them **in order**, one at a time.

Read notes 11 (CSF formats), 15 (loader walk-through), 16 (gaps) and 18 (design) before starting. Come back to the theory notes when a step points to them.

---

## Part 1 · Rules (from vLLM's `AGENTS.md`; breaking them can get the account banned)

1. **Check for duplicates before every PR:**
   ```bash
   gh issue view <issue_number> --repo vllm-project/vllm --comments
   gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
   gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
   ```
   If an open PR already does it, **don't open another**. Help that one instead (test it, comment with results).
2. **No pure agent PRs.** The human submitter must understand and defend every line, review every changed line, and run the tests.
3. **The PR description must state:** why it isn't a duplicate; the test commands run and their results; model eval results if output, accuracy or serving changes; and that AI assistance was used.
4. **Run the `pr-checklist` skill** (`.agents/skills/pr-checklist/SKILL.md`) before opening any PR, drafts included, and fix what it finds.
5. **No busywork PRs** (lone typo fixes and the like).
6. **Python only through `uv` and `.venv/bin/python`.** Never system `python3`, never bare `pip`.
7. **Style:** match the surrounding code; 88-character lines; Google-style docstrings (`Args:` / `Returns:` / `Raises:`); few comments.
8. **Tests:** extend existing test files and fixtures where possible; one behavior per test; kernel benchmarks go in `benchmarks/kernels/`, not `tests/`; run model evals for model-affecting changes and put them in the PR.
9. **Commits** end with a `Co-authored-by:` trailer for the agent and a `Signed-off-by:` line for the human.

Project rules on top:

- Don't commit checkpoint bytes (even small slices) into tests. The LIL license forbids re-uploads, and NVIDIA's license doesn't invite it either. Generate test data with the encoders in `scripts/` (note 11 shows they reproduce the real bytes).
- Prefer the NVIDIA repos in docs and examples.
- **Stop and ask the human** when: a task's acceptance check fails twice for the same reason; a change touches Humming's repo; a design choice isn't covered here; you'd need to change a public API, a CLI flag or an env var name; or an eval drops more than the noise band.

## Part 2 · One-time setup

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt && pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in
export HF_TOKEN=...            # the human's token; never print or commit it
```

Disk: Qwen FP4-Local is 95 GiB and GLM FP4-Local is 165 GiB. Download them once to a shared cache (`HF_HOME`).

Search keywords for duplicate checks in this project: `csf`, `csf_32x2`, `nvfp4_csf`, `weight_scale_encoding`, `FP4-Local`, `Qwen3.8-Flash-Next`, `GLM-5.3-Flash`, `nvfp4 hopper`, `humming nvfp4`, `marlin nvfp4 moe`, `subfolder safetensors`, `index weight_map`.

Known related work on 19 Oct 2026:

| PR / issue | What | Relation |
|---|---|---|
| [#60646](https://github.com/vllm-project/vllm/pull/60646) (draft, stecasta) | slab CSF decode at load + subfolder shards fallback | **T1 / T2 / T3 build on it** |
| [#38728](https://github.com/vllm-project/vllm/pull/38728) (open, Tib-Gridello) | convert NVFP4 weights to **FP8 copies** at load on Hopper; reports +62 % throughput on Mistral-Small-4 NVFP4, 2 × H200 | different trade-off (doubles expert memory); read it before Phase D |
| #49070, #48624, #48895 | Marlin NVFP4 MoE bugs on Hopper | watch during T5 / T6 |
| #57748 | GLM NVFP4 on 4 × H100 question | user interest |

## Part 3 · Tasks

### T0 · Baseline on H200 (Phase 0). No PR.

- **Goal:** reference numbers for the vanilla checkpoints.
- **Steps:**
  1. `vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 --max-model-len 32768 --kv-cache-dtype fp8` on 1 GPU. Note the `Using '…' NvFp4 MoE backend` log line.
  2. Run GSM8K (note 21, section 2) and save the score.
  3. Save reference logprobs for KLD (note 21, section 3).
  4. Repeat with `--moe-backend marlin --linear-backend marlin`.
  5. Repeat steps 1–4 for `nvidia/GLM-5.3-Flash-NVFP4` with `-tp 2`.
  6. `vllm bench serve` at 1 and 32 users (note 21, section 4).
- **Output:** a results table in a local file (for example `results/phase0.md`) with versions, commands and numbers.
- **Done when:** both models give sensible GSM8K with at least one backend. Any failure → file an issue (duplicate check first), then continue with the other backend.

### T1 · Review and test PR #60646. No new PR.

- **Goal:** the slab decoder is correct and loads Qwen FP4-Local.
- **Steps:**
  1. `gh pr checkout 60646` on a branch.
  2. Run its tests: `.venv/bin/python -m pytest tests/model_executor/model_loader/test_csf_scales.py -v`.
  3. Compare its decoder with `scripts/csf_slab_decode.py` on a real expert (outside the repo).
  4. `vllm serve nvidia/Qwen3.8-Flash-Next-FP4-Local --max-model-len 32768 --kv-cache-dtype fp8` on 1 H200. Expect `Decoded … CSF expert scale tensors at load`.
  5. GSM8K; KLD vs Phase 0.
- **Output:** a comment on #60646 with exact commands and results ("tested on H200, sm_90"). Mention the GLM gap (T2) in the same comment.
- **Done when:** the comment is posted (by the human).

### T2 · Load index-listed shards in subfolders, even when top-level shards exist

- **Depends on:** T1 (talk to the #60646 author first: suggest folding it into #60646, or a small follow-up PR).
- **Files:** `vllm/model_executor/model_loader/default_loader.py` (`_prepare_weights`, around lines 229–256); the test file from #60646.
- **Steps:**
  1. After the root glob, if an index file exists, add every file listed in its `weight_map` that exists on disk and isn't already in the list. For Hub models, make sure the download includes them (the allow patterns must cover `tensors/*.safetensors`, or download by the index's file list).
  2. Keep `filter_duplicate_safetensors_files` working: it must not drop index-listed subfolder files.
- **Tests:** extend the #60646 test: a fake snapshot with `model-00001.safetensors` at the top **and** `tensors/native-QB.safetensors`, both in the index → both returned. Plus: a file in a subfolder that's **not** in the index is **not** returned.
- **Done when:** the tests pass, and GLM FP4-Local's `native-QB` tensors load (check that `kv_b_proj` weights aren't all zeros after load).

### T3 · Decode `csf_32x2_v1` at load

- **Depends on:** T1 (same file, `vllm/model_executor/model_loader/csf_scales.py`).
- **Steps:**
  1. Add suffix constants for the five streams: `.csf32_bases`, `.csf32_codes`, `.csf32_page_bases`, `.csf32_offsets`, `.csf32_exceptions`.
  2. In `decode_csf_scale_streams`, collect the parts per base name; when all five are present, decode and yield `<base>` as `float8_e4m3fn`.
  3. Port `decode_vec` from `scripts/csf32_vectorized.py` to torch (CPU): `rows = bases.numel()`, `cols = 2 * codes.numel() // rows`. Cache the block order / code positions per `(rows, cols)` with `functools.lru_cache`.
  4. Validate inputs and raise `ValueError` with a clear message: dtypes (u8, u8, u32, u16, u16), `rows % 32 == 0`, even `cols`, `len(offsets) == blocks + pages`, `bases <= 240`, exception indices in range.
  5. Keep the "stream without a partner" error for incomplete groups.
- **Tests** (in the #60646 test file):
  - Port `encode` from `scripts/csf32_vectorized.py` into the test as the reference encoder.
  - Round trip for shapes `(2048, 256)`, `(4096, 128)`, `(96, 6)` and `(160, 10)` (partial tiles), with injected out-of-window bytes.
  - Five streams in → one `<name>` tensor out; a missing stream → error.
- **Real-data check (local only, not in CI):** load `nvidia/GLM-5.3-Flash-FP4-Local`, decode `layers.3.mlp.experts.0.gate_proj`, compare with the LIL `qad2-kda-w8a16` uncompressed scale (`scripts/csf32_decode.py` does the same). Must be `True`.
- **Done when:** tests pass, the real-data check is `True`, GLM FP4-Local loads with `-tp 2` on H200, and GSM8K is within noise of Phase 0 GLM vanilla.

### T4 · Validate `weight_scale_encoding`

- **Files:** `vllm/model_executor/layers/quantization/modelopt.py` (`ModelOptMixedPrecisionConfig.__init__`, around line 1572).
- **Steps:** for each `quantized_layers` recipe, read `weight_scale_encoding`. Allowed: missing, `"csf"`, `"csf_32x2_v1"`. Anything else → `ValueError("Unsupported NVFP4 weight_scale_encoding '<x>' for <prefix>; this vLLM version supports: csf, csf_32x2_v1")`.
- **Tests:** extend the existing ModelOpt config tests (search `tests/` for `ModelOptMixedPrecisionConfig`): an unknown encoding raises; the known ones construct fine.
- **Done when:** tests pass. Fold this into T3's PR if the reviewers prefer.

### T5 · End-to-end Phase A validation. No code, unless bugs.

- **Steps:** with T1–T4 applied:
  1. Qwen FP4-Local, TP 1; GLM FP4-Local, TP 2. Both backends (default and Marlin).
  2. GSM8K, KLD vs Phase 0, `vllm bench serve`.
  3. Peak memory vs note 19's table.
- **Output:** a results table; post it in the T3 PR description.
- **Done when:** GSM8K for FP4-Local ≥ the vanilla checkpoint's (within noise), and Humming vs Marlin KLD is near 0.

### T6 · Phase B correctness matrix

- **Steps:** run the matrix in note 18 (Phase B): MTP on/off, images on/off, prefix caching, CUDA graphs vs `--enforce-eager`, TP and EP for GLM. Watch the risk list in note 16, section 4.
- **For each failure:**
  1. Reduce it to the smallest repro (one layer shape, if possible).
  2. Duplicate check.
  3. A unit test that fails.
  4. Fix in the shared function (all callers), not in one caller.
  5. One PR per root cause.
- **Done when:** every cell passes or has an issue with a workaround documented.

### T7 · Add H200 eval configs for FP4-Local

- **Files:** `tests/evals/qwen4_exp/configs/` (copy `Qwen3.8-Flash-Next-FP8.yaml`, change `model_name` to `nvidia/Qwen3.8-Flash-Next-FP4-Local` and `server_args` to `--tensor-parallel-size 1 --kv-cache-dtype fp8`; add it to `models-h200.txt`). For GLM: `tests/evals/gsm8k/configs/` (model YAML plus a `models-h200-…txt` list, like the existing ones).
- **Thresholds:** set them from T5's measured scores minus the tolerance the neighboring configs use. Don't guess.
- **Done when:** `pytest -s -v tests/evals/qwen4_exp/test_accuracy.py --config-list-file=configs/models-h200.txt` passes locally on H200.

### T8 · Phase C investigation: the scale layouts. A write-up, then an RFC issue.

- **Steps:**
  1. Marlin: read `nvfp4_marlin_process_scales` and `prepare_nvfp4_moe_layer_for_marlin` (`marlin_utils_fp4.py:59-121`, `:329`). Write down the permutation as a function of shape, the byte transform, and every per-layer factor.
  2. Humming: read `convert_to_humming_moe_kernel_format` (vLLM, `quantization/utils/humming/`) and `humming/transform.py` (Humming repo). Same questions.
  3. Prototype (outside vLLM, in a scratch script): decoded checkpoint scales → index map + LUT → bytes equal to what Marlin's post-load step produces. Bit-exact or it's wrong.
  4. Write an RFC issue: the problem (3.3 / 8.1 GiB), the design (note 18, Phase C), the API proposal (flag name, where the decode op lives), and the measured decode cost. Ask the MoE / quantization maintainers before coding.
- **Done when:** the prototype is bit-exact for Marlin, and the RFC is posted (by the human).

### T9 · GPU CSF decoder kernel (Triton)

- **Depends on:** T8 and maintainer feedback.
- **Files:** a new op (for example `vllm/model_executor/layers/quantization/utils/csf_decode.py`) registered as a custom op; tests in `tests/kernels/quantization/`; a benchmark in `benchmarks/kernels/`.
- **Steps:**
  1. Kernel input: the CSF streams of one layer, stacked per expert (pre-split exceptions per tile at load, like b12x), `topk_ids` (or "all"), the index map and the LUT from T8.
  2. Kernel output: write the routed experts' scales into the scratch, in the kernel's layout.
  3. Modes: routed-only with duplicate skip (small batch), active-set (medium), all experts (long prefill). Note 14 has the thresholds b12x uses.
- **Tests:** for random `topk_ids`, the scratch slots of routed experts equal the load-time (Phase A) processed scales, bit for bit, for both CSF variants and both models' shapes.
- **Benchmark:** decode time per layer for batch 1, 32, 4096 and 16384 tokens; compare with the MoE GEMM time.
- **Done when:** tests pass, and routed-only decode costs < 5 % of the MoE layer's time at batch 1–32.

### T10 · Wire Phase C into the NVFP4 MoE path (Marlin first)

- **Steps:**
  1. Opt-in flag (name agreed in T8's RFC).
  2. At load: keep the CSF streams as layer buffers; allocate one shared scratch per scale shape; point the Marlin scale parameters at it.
  3. At forward: call the T9 op before the MoE kernel, on the same stream.
  4. Must work with CUDA graphs and `torch.compile` (custom op).
- **Tests:** GSM8K identical to T5 (it's bit-identical math, so compare the generated tokens with greedy sampling: they must match exactly); `torch.cuda.max_memory_allocated` lower by the expected amount; a "NaN-poisoned scratch" debug run gives the same output.
- **Done when:** exact token match, the memory saving shows, and decode latency is within 3 % of T5.

### T11 · Phase D0 benchmark gate. No PR, unless it's a useful benchmark script.

- **Steps:** in `benchmarks/kernels/`, time Humming MoE **MXFP4** W4A16 vs W4A8 (FP8 activations) at Qwen and GLM expert shapes, 80 / 160 / 320 / 455 tokens per expert, on H200. Estimate the end-to-end prefill gain. Read #38728's numbers for comparison (they materialize FP8 weights, so they're an upper bound).
- **Done when:** a table and a go / no-go call. Below about 15 % end-to-end prefill gain → **stop Phase D** and write it up.

### T12 · Phase D1–D3 (only if T11 says go)

- **First:** open an issue on `vllm-project/humming` describing the "fused E4M3 residual scale" mode (note 17, section 4; note 18, Phase D) and ask jinzhen-lin whether they'd accept it.
- Then D1 (Humming kernel), D2 (vLLM opt-in wiring, W4A16 for small batches and W4A8 above a threshold, same packed weights), D3 (KLD + GSM8K on both models; per-layer rounding report).
- **Done when:** a measured prefill speedup, eval results within an agreed band, and maintainer approval.

## Part 4 · Templates

### PR description

```markdown
## Purpose
<one paragraph: what and why; link the issue>

## Not a duplicate
<the gh searches you ran and what they found; how this differs from #60646 / #38728>

## Test plan
<exact commands>

## Test results
<pytest output summary; GSM8K / KLD table with versions, GPU, flags>

## AI assistance
This PR was written with AI assistance (Claude). I reviewed every changed line and ran the tests above.
```

Keep the PR description's attribution lines as the human's tooling requires.

### Commit message

```text
[Model Loader] Decode csf_32x2_v1 NVFP4 scales at load

<what and why, wrapped at 72>

Co-authored-by: Claude <noreply@anthropic.com>
Signed-off-by: <Human Name> <human@example.com>
```

## Part 5 · Order on one page

| # | Task | Phase | PR? | Needs |
|---|---|---|---|---|
| T0 | Baseline | 0 | no | 2 × H200 |
| T1 | Test #60646 | A | comment | 1 × H200 |
| T2 | Subfolder shards always | A | small PR (or into #60646) | |
| T3 | `csf_32x2_v1` decode | A | PR | T1 |
| T4 | Encoding validation | A | PR or part of T3 | |
| T5 | Phase A end-to-end | A | results | T1–T4 |
| T6 | Correctness matrix | B | bug PRs | T5 |
| T7 | Eval configs | B | PR | T5 |
| T8 | Scale layouts + RFC | C | RFC | T5 |
| T9 | GPU decoder | C | PR | T8 |
| T10 | Wire Phase C | C | PR | T9 |
| T11 | FP8 benchmark gate | D | maybe | — |
| T12 | W4A8 kernel + wiring | D | PRs (Humming + vLLM) | T11 says go |

---

**Next:** [21 · Testing and benchmarking](21-Testing-And-Benchmarking.md)
