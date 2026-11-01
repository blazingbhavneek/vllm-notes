# NVIDIA L40S (SM89) — 2026-11-01

- Driver 595.99.02; PyTorch 2.14.0+cu132; Triton 3.8.0. Initial vLLM commit `f5e6e1bb0b`; FP16 retry after fast-forward: `ff9dd45e8b`.
- 142 SMs, 44.4 GiB VRAM, 384-bit bus, 96 MiB L2, copy bandwidth 609 GiB/s, GDDR.
- Cutoffs in effect: IQ2_XXS 128, Q8_0 256; MoE GEMV through M=64.
- Checkpoint: `/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f`.
- Phases A–C and baseline BF16/FP16 were run on the initial commit. After the user requested a pull, only full-model FP16 accuracy and throughput were rerun on the updated commit. Kernel and cutoff results below are not a revalidation of the updated commit.

## Correctness

| Check | Result on initial commit |
|---|---|
| Kernel tests | 192 passed, 0 failed, 0 skipped in 259.16 s; slowest case 3.28 s, next 3.11 s and 2.99 s. Full slowest 15 in `kernel_tests.txt`. |
| `test_modelopt -k block` | 54 passed, 1 failed, 75 deselected in 757.36 s. Failure: known dense-cache equality test. |
| Dense-cache test ×3 | pass / pass / pass (after the failure in the block run). |
| Warmup JIT BF16 / FP16 | Both clean: 2 warmed, 0 compiled during inference for ReLU2 GEMV. |

## Cutoffs

- Dense fine sweep: IQ2 dequant wins on both shared layers from M=48 in both dtypes, though timing at lower M is mixed and noisy. Suggest testing 48 instead of 128. Worst `op/best` is 2.13 (shared_up BF16, M=96); shared_down reaches 1.85 at M=96. The FP16 shared_up M=96 ratio is 2.08.
- Q8 dequant wins on a majority of four layers at M=192 in both dtypes, and all four by M=256; at M=96–128 results vary by layer. Suggest testing 192 instead of 256. Worst `op/best` is 1.43 (fc2_latent FP16, M=128); BF16 fc2 reaches 1.42 at M=128. These are microbenchmark suggestions, not serving-validated cutoffs.
- MoE, E=512: GEMV wins through the largest sampled M=128 for both BF16 (1.17×) and FP16 (1.121×); the actual crossover is above the measured range. This is a hint rather than a serving recommendation. CUPTI was unavailable, so timings used CUDA events; cold-L2 benchmarking was disabled by the benchmark helper.
- Full tables: `cutoff_dense.txt`, `cutoff_dense_fine.txt`, `cutoff_moe_bf16.txt`, `cutoff_moe_fp16.txt`, and `benchmark.txt`.

## Full model

| Run | GSM8K-250 | Trunc | c=1 tok/s | c=8 tok/s | c=32 tok/s | median TPOT c=1 ms | load GiB | startup s | Inference JIT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF16, f5e6e1bb0b | 235 | 2 | 84.54 | 345.99 | 543.26 | 10.70 | 21.7 | 340 | `_causal_conv1d_fwd_kernel` |
| FP16, f5e6e1bb0b | 3 | 250 | 85.45 | not requested | interrupted | 10.60 | 21.7 | 280 | `_causal_conv1d_fwd_kernel` |
| FP16, ff9dd45e8b | 234 | 1 | 91.85 | not requested | 548.65 | 10.19 | 21.7 | 291 | `_causal_conv1d_fwd_kernel` |

- BF16 median TPOT c=8 / c=32: 20.97 / 52.97 ms. Updated FP16 median TPOT c=32: 53.11 ms. Startup is wall time from launching the server to its first successful models request, sampled in 10-second intervals.
- Logs identify Triton IQ2_XXS linear and MoE, Q8_0 linear, and FP4 Marlin. No branch kernel from the requested list was logged as compiling during inference; the causal convolution JIT is a different kernel. There is no Q8 MoE backend line in these model runs.
- Updated FP16 example answers: the eggs question concludes “9 * $2 = $18 per day”; the robe question concludes “2 + 1 = 3 bolts”; the house example predicts $70,000. Question 111 looped and was truncated, predicting 64 instead of 16. Initial FP16 question 0 repeated the question and “Let's think step by step” until the limit. Full responses are in the JSON files.
- Optional serving MoE cutoff experiment was not run. Initial FP16 c=32 was interrupted at the user's pause request; it was rerun after the update, not on the original commit.

## Problems

- Initial FP16 accuracy collapsed: 3/250, all 250 length-limited. The updated commit improved it to 234/250, 1 length-limited; that remaining truncation is a finding. The output failure produced no exception traceback; `REPRO.md` and both run logs/JSONs preserve the reproduction and evidence. No fix was attempted locally.
- The known dense-cache test failed in the block suite (`modelopt_tests.txt`): `assert cached == uncached` at `tests/quantization/test_modelopt.py:1929`, with mismatching log probabilities. The requested three repeats passed. No investigation was performed.
- Initial collection lacked `tblib`; installed `tblib==3.2.2`. The editable install upgraded torch from 2.13.0+cu130 to 2.14.0+cu132. Existing TorchAudio was built for CUDA 13.0, causing `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions`. `uv pip install torchaudio==2.14.0 --torch-backend=auto` found no such version. Temporarily hiding the optional TorchAudio package enabled these tests; it was restored afterward. The original supervised vLLM service was stopped for testing; a restart after restoration failed with the same mismatch and is currently exited. Full traceback: `original_service_restart.log`; install details: `install.txt`, `install_test_missing.txt`.
- Nonfatal imports warned that `nixl_ep_cu13.nixl_ep_cpp_torch214` was missing; full tracebacks appear in warmup, benchmark and server logs. Warmup checks still explicitly reported no JIT in their target path.
- Workspace is not volume-backed. Results are on instance storage and need copying elsewhere to survive recycle/destroy.

## Local diffs

- `local.diff` is empty. No local source edits or MoE cutoff edits were made. The branch was fast-forwarded to the requested updated origin commit.
