# 1: Don't add a new L4 CI job; move the tests into the existing kernel job

## What happened

PR #65003 (`pr/modelopt-block-triton`, HEAD `f533470e67`) adds a new CI job, "(L4) ModelOpt block codecs", in `.buildkite/test_areas/quantization.yaml`. It runs two tests from `tests/quantization/test_modelopt.py`. The same change also removes that file's line from the exceptions list, `tools/pre_commit/test_tethering_allowlist.txt`.

The problem: a new job like this uses extra GPU capacity, so it needs maintainer approval and a justification. The file was also turned off on purpose by maintainers in commit `9d48c8ab95` (torchao deps not locked), so re-enabling it through a new job goes against that decision.

Better option: the two tests fit into `tests/kernels/quantization/test_modelopt_block_triton.py`. That file already runs on L4 in the existing `(L4) Quantization Kernels Shard %N` job in `.buildkite/test_areas/kernels.yaml` (6 shards). No new job and no exceptions-list change.

## Todo

- [x] Undo the new L4 job in `quantization.yaml` and the allowlist line removal, in a new commit. Both are still in HEAD.
- [x] Move `test_modelopt_block_quant_backend_selection` into `tests/kernels/quantization/test_modelopt_block_triton.py`. It doesn't start an engine. Move its helper `_iq2_xs_moe_config` to `tests/quantization/utils.py`, since other tests in `test_modelopt.py` use it too. Drop `@_requires_sm80`, because the kernel file's `pytestmark` already covers it.
- [x] Decide on `test_modelopt_block_triton_dense_cache_matches_uncached`. It starts a full vLLM engine, which adds time to the kernel shards. Check shard runtime before moving it, and ask maintainers if it's unclear.

## 2: Move the Triton-kernel doc section to the ModelOpt page

PR #65003 adds a section, "Other NVIDIA GPUs (Triton kernels)", at the end of `docs/features/quantization/b12x.md` (commit `f533470e67`). It covers ModelOpt IQ2_XXS and Q8_0 Triton kernels, which have nothing to do with b12x. The right home is `docs/features/quantization/modelopt.md`.

- [x] Remove the appended section from `b12x.md`, and add it to `modelopt.md` before "Benchmarking with dummy weights". Link to `b12x.md#modelopt-block-codecs` for the codec table. Make this change in a new commit, only after the maintainers have answered.
- [x] Reword the moved text so it works without the b12x page. It says "without b12x" and "IQ2_XS still requires b12x" but never explains b12x. Add a link to `b12x.md` and a short explanation.
- [x] Rename the heading. "Other NVIDIA GPUs (Triton kernels)" only makes sense next to b12x. Use something like "ModelOpt block codecs with Triton kernels (SM75+)".
- [x] Check the moved text for "Bolt's shared dense experts". Public docs shouldn't name Bolt unless it's a published model. If it isn't, rewrite that sentence.

## 3: Check what vLLM already has before keeping our own kernels

The PR adds Triton kernels for every quantized row of Bolt, but the only format that is new to vLLM is IQ2_XXS (experts). The Q8_0 rows (attention, latent, lm_head) might be coverable by existing kernels. Layer sequence and shapes come from Bolt's public `config.json` and Appendix C of `03-improvment-handoff.md`.

### Bolt layer map (88 layers, 0 to 87)

Mamba sits at every even index. An odd index holds MoE, except 8 attention layers at 7, 16, 25, 36, 47, 58, 69 and 78. Hidden size is 4096. Every block is: RMSNorm (BF16) → block → residual add.

| Block | What it does | Why it exists here |
|---|---|---|
| **Mamba (SSM)** | A state-space layer. It reads the sequence once, left to right, and keeps a fixed-size memory ("state") that it updates per token. | Cost per token is flat, however long the context is. State is small (fp32 cache). |
| **Attention** | Every token looks back at all earlier tokens. Here it is GQA: 32 query heads share 2 KV heads, head dim 128, no sliding window, up to 262144 tokens. | Great at exact recall, but its cost and KV cache grow with context. So the model uses it sparingly: only **8 of 88** layers. |
| **MoE** | A router picks 9 experts out of 240–496 per token. Only those 9 run (ReLU², non-gated), plus 1 shared expert that always runs. | Lots of capacity for little compute per token. |

| Stage | Count | Sub-part | Shape `[N, K]` | Format | In this PR? | Stays packed? |
|---|---|---|---|---|---|---|
| **Embedding** | 1 | token table, vocab 131072 × 4096 | `[131072, 4096]` | BF16 (checked in Bolt's `config.json`). The PR's Q8_0 lookup is for other checkpoints, e.g. Super 3.5 packed (#59093 body). | Q8_0 lookup only | Yes. It decodes only the looked-up rows. |
| **Mamba** | 40 | `in_proj` | 4096 → 2×8192 + extras | NVFP4 (Marlin) | No | n/a |
| | | `out_proj` | 8192 → 4096 | NVFP4 (Marlin) | No | n/a |
| | | SSM and conv (128 heads × 64, state 128, 8 groups, conv width 4) | | BF16 | No | n/a |
| **Attention** | 8 | `qkv_proj` (q 4096 + k 256 + v 256) | `[4608, 4096]` | Q8_0 | Yes | Packed. At big M a temporary bf16 copy of about 38 MB is made. |
| | | `o_proj` | `[4096, 4096]` | Q8_0 | Yes | Same as above |
| | | softmax and KV cache | | BF16 | No | n/a |
| **MoE** | 40 | router (sigmoid, top-9) | 4096 → E | BF16 (not in `quantized_layers`, checked) | No | n/a |
| | | `fc1_latent_proj` (down to latent) | `[1024, 4096]` | Q8_0 | Yes | Packed. At big M a temporary copy is made. |
| | | **routed experts** up: `[I, 1024]`, down: `[1024, I]`, × E | e.g. E240, I2048 | **IQ2_XXS** | Yes | **Always packed.** The MoE kernel has no dequant path. |
| | | `fc2_latent_proj` (up from latent) | `[4096, 1024]` | Q8_0 | Yes | Packed. At big M a temporary copy is made. |
| | | shared expert up | `[5376, 4096]` | IQ2_XXS | Yes | Packed. At big M a temporary copy is made. (The opt-in cache was removed 2026-11-09.) |
| | | shared expert down | `[4096, 5376]` | IQ2_XXS | Yes | Same as above |
| **Final** | 1 | final RMSNorm | | BF16 | No | n/a |
| | | `lm_head` | `[131072, 4096]` | Q8_0 | Yes | **Always packed.** At 537M elements it is above the 67M-element cap. |

### What the check found so far

- **The "attention" Q8_0 row is not an attention kernel.** Softmax and the KV cache are BF16 and untouched. The Q8_0 rows are plain linear layers (`q/k/v/o_proj`): int8 weights with one fp16 scale per 32 values. The PR's Q8_0 code is a dense GEMM/GEMV kernel, not attention.
- **Why Q8_0 is in this PR at all.** On `main`, `ModelOptBlockQuantLinearMethod` and `ModelOptBlockQuantMoEMethod` both refuse to load without b12x (SM120/121). They raise "Block-quantized dense weights require b12x on SM12x" and "Block-quantized routed experts require b12x on SM12x". So on every other GPU, no row of this checkpoint ran. The PR adds a non-b12x path for all rows, not only the experts.
- **IQ2_XXS has no existing kernel in this tree.** No `csrc/quantization/gguf` and no `gguf.py` exist. GGUF support moved to the external `vllm-gguf-plugin` (see `docs/features/quantization/gguf.md`), so I could not check whether its kernels handle Q8_0 or IQ2_XXS.
- **Possible reuse for Q8_0: Marlin.** `MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]` (`marlin_utils.py:37`). Group size 32 matches Q8_0's block size, so an 8-bit Marlin W8A16 path might work after repacking the int8 weights and scales. Not verified: 8-bit support in the Marlin path, the repack layout, and the GPU floor (T4/SM75 is probably out, and SM75 is a goal of this PR).

- [x] Check each reuse candidate for the Q8_0 rows: (a) Marlin W8A16 with group size 32, including repack from the ModelOpt int8 layout, accuracy, speed and the SM75 gap; (b) the kernels in `vllm-gguf-plugin` for Q8_0 and IQ2_XXS.
  - Done 2026-11-09 on the 5060 Ti (script: `q8_marlin_check.py`, in the session scratchpad).
  - (a) Marlin works:
    - `choose_mp_linear_kernel` picks Marlin (`uint8b128` = int8 + 128, group 32) for all 5 Q8_0 shapes, in BF16 and FP16, and with TP=2 splits.
    - Repacking is `pack_quantized_values_into_int32(q + 128)`. Scales must be cast to the activation dtype; unconverted FP16 scales give garbage with BF16.
    - SM75 is not a gap: Marlin's minimum is SM75.
    - Speed is a tie at decode, mixed at M = 64–1024.
    - Accuracy: in BF16, Marlin's error is 2.8e-3 vs our 1.6–2.3e-3, because it rounds FP16 scales to BF16 (up to 2⁻⁸). In FP16 the two match.
  - (b) The old in-tree GGUF kernels did have IQ2_XXS, but they quantize activations to Q8_1, so they're not reusable as-is.
- [ ] For each Q8_0 row (qkv, o_proj, fc1/fc2_latent, lm_head), decide: reuse the existing kernel, or keep our Triton one and say why. If reuse wins, remove that code from the PR so the diff only adds what is new (IQ2_XXS experts and shared experts).
  - Not decided. Asked the reviewers in the update comment (`Notes/02-Bolt-Support/10-pr-triton-description.md`). Precedent: #65272 sends INT8 through Marlin.
  - Decided 2026-11-09: keep the Q8_0 embedding and expert paths, since other checkpoints use them.
- [ ] Put the decision in the PR description: which rows reuse existing kernels, which are new, and why. Reviewers will ask why Q8_0 isn't reused.
  - Partly done: the update comment lists what's reused and asks the Marlin question. The body is still stale.

### Index of existing vLLM kernels to check, row by row

The PR changes no attention code (`git diff main...HEAD` is empty for `vllm/v1/attention`). The 8 attention layers use the normal `Attention(...)` in `NemotronHAttention` (`models/nemotron_h.py:426`). Bolt is plain GQA (32 query heads, 2 KV heads, head dim 128, no sliding window, no MLA latent KV). So FlashAttention, FlashInfer or Triton attention are picked automatically per GPU, and MLA (DeepSeek-style) kernels do not apply. Only the weights of `q/k/v/o_proj` are Q8_0, and that is a linear-layer question.

Where each family lives (all paths under `vllm/`):

| Family | Where | What to check for Bolt |
|---|---|---|
| Attention backends (picked per GPU automatically) | `v1/attention/backends/`: `flash_attn.py` (FA2/FA3), `flashinfer.py`, `triton_attn.py`, `flex_attention.py`, `rocm_*`, `cpu_attn.py`, `mla/` | Nothing to write. Check the startup log to see which backend the 8 attention layers pick on each target GPU. |
| Mamba / SSM backends | `v1/attention/backends/mamba2_attn.py`, `mamba_attn.py`, `short_conv_attn.py` | Not touched by this PR. |
| Weight-only dense linear, picked case by case | `model_executor/kernels/linear/mixed_precision/`: `marlin.py` (SM75+), `humming.py` (SM75+), `conch.py` (SM80+), `machete.py` and `cutlass.py` (SM90+), `exllama.py`, `triton_w4a16.py` (4-bit only), plus CPU, XPU and ROCm kernels. Selection: `choose_mp_linear_kernel` in `kernels/linear/__init__.py:890`, which walks `_POSSIBLE_KERNELS` and checks min GPU and `can_implement`. | Q8_0 rows: `qkv`, `o_proj`, `fc1/fc2_latent`, `lm_head`. Marlin supports `uint8b128` and group size 32. Verify with Marlin, Humming and Conch. |
| Scaled-MM linear (FP8, int8, block-scaled) | `model_executor/kernels/linear/scaled_mm/`: `cutlass.py`, `triton.py`, `deep_gemm.py`, `flashinfer.py`, `marlin.py`, `humming.py`, `b12x.py`, ... | Probably not relevant (activations are quantized there), but confirm. |
| Skinny / low-latency GEMM | `model_executor/kernels/linear/cute_dsl/`: `skinny_gemm.py`, `ll_bf16.py` | Possible helpers for M=1 decode. Check whether they take packed weights. |
| MoE experts | `model_executor/layers/fused_moe/experts/`: `triton_moe.py`, `marlin_moe.py`, `fused_humming_moe.py`, `cutlass_moe.py`, `deep_gemm_moe.py`, `flashinfer_*`, `trtllm_*`, ... The PR adds `modelopt_block_triton_moe.py`, a subclass of the Triton experts. | IQ2_XXS experts. None of these seems to read IQ2_XXS. Confirm, and check what the PR reuses from `triton_moe.py` (routing, workspaces) so it does not rewrite them. |
| Embedding | `model_executor/layers/vocab_parallel_embedding.py` and the b12x Q8_0 lookup | Q8_0 embedding is a gather, not a matmul. Marlin does not cover it. |

- [ ] Build the index for real: for each Bolt row in the layer table, and for each GPU generation we target (SM75, SM80/86/89, SM90, SM100, SM120), write down which existing kernel `choose_mp_linear_kernel` (dense) or the MoE experts list would pick, and whether it accepts Q8_0 / IQ2_XXS. Mark each cell "works", "works with repack" or "not possible".
  - Partly done:
    - Dense Q8_0 → Marlin "works with repack" on SM120 (measured), and on SM75–SM89 by the kernel's minimum capability (not run).
    - SM90 is still open: Machete/CutlassW4A8 come first there and should reject group 32. Check with `q8_marlin_check.py select` on the H200.
    - IQ2_XXS: no existing kernel reads it.
- [ ] Measure the "works with repack" candidates against our Triton kernels, on accuracy and speed. Known risk for Marlin: `review.md:153` says its scales are bf16, so fp16 Q8_0 scales lose 3 mantissa bits. The earlier Humming PR for Q8_0 (#59093) was closed, and the reason is not in the notes (look it up with `gh pr view 59093`).
  - 5060 Ti done; see the first checkbox. The scale risk is confirmed (up to 2⁻⁸ relative error).
  - #59093 was closed by its author on 2026-09-30 with no comment. A Humming fix, "preserve FP16 group scales with BF16 outputs", links to it.
  - H200 numbers: planned for the run on 2026-11-10.
- [ ] Shrink the PR to what is new. Wire the Q8_0 rows through `choose_mp_linear_kernel` where an existing kernel is good enough, keep our Triton kernel only as the fallback for GPUs and shapes the existing ones cannot do, and keep IQ2_XXS (experts and shared experts) as the new code.
  - Partly done (2026-11-09, uncommitted): MoE prefill now uses upstream `fused_moe_kernel`, the Q8_0 embedding uses the compressed-tensors gather kernel, and split-K uses `_fixed_order_sum_kernel`. Non-test lines went from about 1,340 to about 1,240.
  - Q8_0 dense → Marlin waits on the reviewers' answer.

## 4: Rework the tests so they prove the new IQ2_XXS kernels, not just "run"

The kernel tests are not minimal. `tests/kernels/quantization/test_modelopt_block_triton.py` has about 35 test functions (many parametrized): bit-exact decode, dense output within 1.5× the error of cuBLAS, MoE against an unfused reference, odd K, offset views, int64 offsets, `torch.compile`, CUDA graph replay. So the basics of correctness are there. What is missing is what you listed:

- **No hand-checkable case.** Every decode test compares the kernel with `ref_decode`, which is code that uses the same `_IQ2XXS_GRID` table. A human cannot look at it and say "yes, that is right". The only outside check of the table is the `gguf` comparison, which is skipped when `gguf` isn't installed (it is not in the test requirements).
- **No speed test.** The benchmark script prints times and GB/s but asserts nothing, and CI never runs it. It does not print how close the kernel is to the GPU's copy bandwidth.
- **No parity test against a bf16 model.** The two engine tests use random (`dummy`) weights and compare eager vs compiled and cached vs uncached. Nothing compares packed Triton output with a plain bf16 forward on dequantized weights. The real-tensor check and GSM8K run exist only in the notes folder, not in the repo.
- **Untested hardware branches.** All kernel tests skip below SM80. The Turing (SM75) tile settings and the Hopper-specific Q8_0 rule are never run. CI's L4 is SM89.
- I did not see tests for bad inputs (wrong K, non-finite scales, unsupported dtype).

### What the tests should look like

**A. One hardcoded IQ2_XXS case a human can verify by hand.**
An IQ2_XXS block is 66 bytes for 256 weights: a 2-byte fp16 `d`, then 8 groups of 8 bytes. Each group covers 32 weights: 4 grid indices (one byte each, 8 weights per index), then 4 bytes holding four 7-bit sign indices and one 4-bit scale `s`. The decode is `value = d · (0.5 + s) · 0.25 · grid_byte · sign`. Grid entry 0 is `0x0808080808080808`, which is eight 8s, so with index 0 the value is `d · (0.5 + s) · 2`.
Build one block by hand with `d = 1.0`, all grid indices 0:

| 32-weight group | `s` | sign index | Expected 32 values |
|---|---|---|---|
| 0 | 0 | 0 (no negatives) | all `+1.0` |
| 1 | 1 | 0 | all `+3.0` |
| 2 | 0 | 1 (parity sets bit 7) | repeating `[-1, +1, +1, +1, +1, +1, +1, -1]` |
| 3 | 2 | 1 | repeating `[-5, +5, +5, +5, +5, +5, +5, -5]` |

The test writes these raw bytes, calls the kernel's dequant, and compares to the table above, written as literal numbers in the test. Then run the dense GEMV with `x` set to ones and compare the row sum with a number worked out by hand (for example group 0 alone gives 32). Also add a small hand-built MoE case with 2 experts. This replaces the need for `gguf` as the only outside check of the table.

**B. Parity against bf16.** For a few layer shapes from the layer table (e.g. `shared_up`, one expert, `qkv_proj`), dequantize with `ref_decode`, run a plain bf16 forward (`x @ w.T`, `fused_experts`), and compare with the packed kernel at M = 1, 8, 64, 512. Keep the "within 1.5× cuBLAS error" bar. Add a short stacked run of 2–3 layers on real or realistically scaled weights, to see error pile up. The layer-by-layer harness idea is in the earlier discussion (dequantize one layer at a time so the full model never has to fit in 16-bit).

**C. Speed.**
- *What to expect.* IQ2_XXS is 66 bytes per 256 weights, 2.06 bits, so it moves about 7.8× fewer bytes than bf16. That is the ceiling at small M, where the layer is memory-bound. It is not what we get: your notes show the IQ2 kernel is **decode-bound**. On the H200 it reads about 250 GB/s of packed bytes, against 2.3 TB/s for a plain copy (`04-progress-log.md`, around lines 385 and 402). So the gain over bf16 cuBLAS is far below 7.8×.
- *Make the benchmark say so.* Print for each row: effective GB/s, percent of measured copy bandwidth, the ideal-speedup (bytes ratio) and the measured speedup, so the gap to 7.8× is visible and honest.
- *Keep CI cheap.* Do not assert timings in CI (flaky). Put the benchmark numbers in the PR description. If a guard is wanted, a coarse one: at M=1 the packed kernel must not be slower than bf16 cuBLAS on the same GPU class.

**D. Basics a kernel PR usually has.** Fixed seeds; both bf16 and fp16; M = 1 and non-multiples of the tile size; K at the tile edge; every codec; both activations; CUDA graph and `torch.compile`; non-finite and extreme scales; at least one test on SM75 and one on SM90 if hardware is available (otherwise say in the PR that they were run by hand and show the output).

Status, updated 2026-11-09 (night). Done on branch `pr/modelopt-block-triton-test`, verified on the RTX 5060 Ti; H200 and Ada runs on 2026-11-10 (`handoff.md`).

- [x] Add the hand-built IQ2_XXS dequant + GEMV test (A), with the expected numbers written out in the test and a short comment explaining the layout.
  - `iq2_block` / `HAND_BLOCK` / `HAND_VALUES` build one block from bytes, with every expected value written out. It covers grid entries 0, 1 and 255, the sign parity bit, s = 0, 1, 2 and 15, and a negative `d`.
  - `test_iq2_hand_built_block` checks `ref_decode`, the kernel's dequant and a GEMV with x = 1 (expected `[[309.75, -619.5]]`).
  - `test_moe_hand_built` is a 2-expert MoE on constant blocks, run through both the GEMV and the `tl.dot` path.
  - `test_iq2_grid_matches_ggml` now compares a sha256 of the embedded grid with gguf-py's (0.19.0) table, so it never skips.
- [x] Add the bf16 parity tests (B) next to the kernel tests, and move the real-tensor and GSM8K evidence into the PR description.
  - `test_linear_bolt_shapes`: qkv, fc2_latent, shared_up and shared_down at M = 1, 8, 64 and 512, within 1.5× of cuBLAS's error.
  - `test_moe_bolt_shape`: E = 240, inter = 2048, top_k = 9.
  - `test_shared_expert_error_does_not_pile_up`: shared up → relu² → down, against cuBLAS.
  - Real-tensor and GSM8K evidence is already in the PR body. The ratios script is `kernel-tests/round10/bolt_parity_ratios.py`.
- [x] Extend the benchmark output (C) with percent-of-copy-bandwidth and ideal vs measured speedup, and attach the table to the PR.
  - Columns: `speedup`, `bytes_ratio` (the ideal), GB/s and `%copy`.
  - Timing is now CUDA-graph replay with a cold L2. The old numbers were warm L2, because CUPTI is missing and flashinfer's fallback skips the flush.
  - Timing takes the best of 3 captures, because captures of the same small kernel sat 15–25% apart on the 5060 Ti.
  - 5060 Ti table: `kernel-tests/round10/benchmark_5060ti_cold_l2.txt`.
- [x] Add the missing basics (D), and add `gguf` to the test requirements or embed the expected grid in the test so the table check can never be skipped.
  - Added `test_linear_non_finite_scales_stay_in_their_rows` (both codecs, M = 1, 17 and 512) and `test_batch_invariant_rows`. The gguf check is fixed by the embedded hash.
  - Still open: the SM90 run (2026-11-10) and SM75 (no hardware).
- [x] Cut test volume where it duplicates (the PR's test diff should not dwarf the code, per AGENTS.md), and keep the engine tests only if they stay cheap (see todo 1).
  - The tiny engine test is dropped. It ran in no CI job, and its H200 failure was random-weight sensitivity: one prompt had a different top-5 at the first token. `test_modelopt.py` is back to a 3+/1− diff.
  - Tests are now 681 added lines, against 1,513 for the code, docs, benchmark and CI (1,256 without the benchmark).
- Also done (review pass):
  - Batch-invariant mode is supported: the dense path dequantizes, or uses one tile shape with no split-K; MoE uses upstream's invariant config.
  - The L4 quantization-kernels CI step now also triggers on `fused_moe.py` and the experts file.
  - Upstream `tests/kernels/moe/test_moe.py` passes: 675.
  - `do_not_specialize` on `fused_moe_kernel` costs nothing on upstream BF16 MoE (worst ratio 1.002, `round10/moe_specialize_ab.py`).
- Open for the H200 and Ada runs:
  - Single-row Q8_0: the Hopper wide-K rule skips qkv (4608, 4096), and on the 5060 Ti the same tile beat generic split-K by 8–17%. A/B in `round10/q8_hopper_rule_ab.py`; it may become a rule for every GPU.
  - Marlin on SM90 (`q8_marlin_check.py select`).
  - MoE prefill at large M: on the 5060 Ti, BM64/BN64/BK64 beat the current BM128/BN128 by 13% at M=4096 (0.74x vs 0.65x BF16) and 6% at M=1024. The current tile came from H200 profiling, so the H200 sweep decides (optional step in `handoff.md`).
  - GDDR Q8 dense cutoff: with cold L2 the 5060 Ti crosses at about 768, not 256. That matches round 8; the L40S crosses at 96–192. The Ada `cutoff_dense.txt` decides whether 256 stays.
  - `check_warmup_relu2.py` was updated to the current `_warm_moe` (warms through `method.apply`). It now checks `fused_moe_kernel` too and passes in BF16 and FP16; a negative control (warm only M=1) catches the JIT.
