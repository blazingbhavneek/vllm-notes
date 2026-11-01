# NVIDIA L40S (SM89) — 2026-11-01

- Driver 595.99.02; PyTorch 2.14.0+cu132; Triton 3.8.0; vLLM commit `ff9dd45e8b`. Previous comparison commit `f5e6e1bb0b`.
- 142 SMs, 44.4 GiB VRAM, 384-bit bus, 96 MiB L2, copy bandwidth 609 GiB/s, GDDR.
- Cutoffs in effect: IQ2_XXS 128, Q8_0 256; MoE GEMV through M=64. The optional alternate serving run used MoE GEMV through M=8.
- Checkpoint: `/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f`.
- All primary results below are on `ff9dd45e8b`; raw files are in `retry-ff9dd45e8b/`. Initial results are retained in this directory and described in `INITIAL_SUMMARY.md`.

## Correctness

| Check | Result on ff9dd45e8b |
|---|---|
| Kernel tests | 192 passed, 0 failed, 0 skipped in 188.72 s. The slowest listed timings were teardowns of 1.15 s; full slowest 15 in `kernel_tests.txt`. |
| `test_modelopt -k block` | 55 passed, 0 failed, 75 deselected, 6 warnings in 422.86 s. |
| Dense-cache test ×3 | Pass / pass / pass; 29.54 / 26.64 / 27.63 s. |
| Warmup JIT BF16 / FP16 | Both clean: 2 warmed, 0 compiled during inference for ReLU2 GEMV. |

## Cutoffs

- Dense fine sweep: IQ2 dequant wins on both shared layers from M=48 in both dtypes. Suggest testing 48 instead of 128. Worst `op/best`: shared_up BF16 2.14 and FP16 2.12 at M=96; shared_down BF16 1.86 at M=96.
- Q8 dequant wins on 3/4 layers at M=96 in both dtypes, only 2/4 at M=128, and 4/4 at M=192 and above. Results vary by layer and are not monotonic. Suggest testing 192 instead of 256, while recognizing the earlier crossover on several layers. Worst `op/best`: fc2_latent BF16 1.44 and FP16 1.43 at M=128. These are microbenchmark suggestions, not serving-validated dense cutoffs.
- MoE at E=512: GEMV wins through the largest sampled M=128 for both BF16 (1.116×) and FP16 (1.118×); the actual crossover is above the measured range. CUPTI was unavailable, so timings used CUDA events; cold-L2 benchmarking was disabled by the helper.
- MoE serving comparison: changing the GDDR GEMV cutoff from 64 to 8 changed median TPOT from 20.99 to 20.98 ms at c=8, 31.47 to 30.26 ms at c=16 (3.8% lower), and 52.75 to 48.33 ms at c=32 (8.4% lower). Output throughput changed from 346.90 to 345.48, 459.58 to 479.06, and 555.04 to 592.22 tok/s respectively. These are single runs; the serving result differs from the standalone MoE microbenchmark, as the handoff warned can happen.
- Tables: `cutoff_dense.txt`, `cutoff_dense_fine.txt`, `cutoff_moe_bf16.txt`, `cutoff_moe_fp16.txt`, and `benchmark.txt` in the updated-run directory.

## Full model

| Run | GSM8K-250 | Trunc | c=1 tok/s | c=8 tok/s | c=32 tok/s | median TPOT c=1 ms | load GiB | startup s | Inference JIT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF16, cutoff 64 | 230 | 3 | 87.46 | 346.90 | 555.04 | 10.60 | 21.7 | 300 | `_causal_conv1d_fwd_kernel` |
| FP16, cutoff 64 | 234 | 1 | 91.85 | not requested | 548.65 | 10.19 | 21.7 | 291 | `_causal_conv1d_fwd_kernel` |
| BF16, cutoff 8 | 227 | 3 | not requested | 345.48 | 592.22 | not requested | 21.7 | 301 | `_causal_conv1d_fwd_kernel` |

- c=16 output throughput / median TPOT: BF16 cutoff 64: 459.58 tok/s / 31.47 ms; cutoff 8: 479.06 tok/s / 30.26 ms. Baseline BF16 median TPOT c=8 / c=32: 20.99 / 52.75 ms; FP16 c=32: 53.11 ms. Startup is wall time to the first successful models request, sampled at 10-second intervals.
- Logs identify Triton IQ2_XXS linear and MoE, Q8_0 linear, and FP4 Marlin. No branch kernel from the requested list was logged as compiling during inference; the causal convolution JIT is a different kernel. There is no Q8 MoE backend line in these model runs.
- Updated FP16 example answers: the eggs question concludes “9 * $2 = $18 per day”; the robe question concludes “2 + 1 = 3 bolts”; the house example predicts $70,000. FP16 question 111 looped and was truncated, predicting 64 instead of 16. Full answers and truncations are in the run JSON files.
- The initial FP16 run at `f5e6e1bb0b` scored 3/250 with 250 truncations; after the new routed-scaling commit, it scored 234/250 with one truncation. The earlier BF16 scored 235/250 with two truncations. Scores are individual runs, not a statistical comparison.

## Performance interpretation

The baseline BF16 rates 87 / 347 / 555 tok/s at c=1 / 8 / 32 compare with the handoff's H200 rates of 132 / 492 / 865. Higher concurrency substantially improves aggregate throughput. Packed-weight reads and limited parallelism are plausible contributors at c=1, but the requested measurements do not establish whether memory, SM execution, scheduling, or another model component is the dominant limit at any concurrency. The 609 GiB/s copy measurement is a reference, not inference bandwidth. The log also warns that an L40S-specific Mamba selective-state-update config was not found and a potentially suboptimal default is used; its impact is unmeasured. The optional cutoff test shows that dispatch affects serving latency but does not isolate a hardware bottleneck.

## Problems

- Initial FP16 accuracy collapsed: 3/250, all 250 length-limited. The updated commit improved it to 234/250, one length-limited. Updated BF16 also has three truncations, and the alternate cutoff BF16 has three; these remain findings. The output failure produced no exception traceback; `REPRO.md` and run logs/JSONs preserve evidence. No local correctness fix was attempted.
- At the initial commit the known dense-cache equality test failed once in the block suite; its three separate repeats passed. At the updated commit, the block suite and three repeats all passed. The earlier traceback is in the root `modelopt_tests.txt`.
- Initial collection lacked `tblib`; installed `tblib==3.2.2`. The editable install upgraded torch from 2.13.0+cu130 to 2.14.0+cu132. Existing TorchAudio was built for CUDA 13.0 and fails to import with the new torch; no `torchaudio==2.14.0` package was found by the attempted `uv pip install ... --torch-backend=auto`. Temporarily hiding optional TorchAudio enabled tests; it was restored afterward. The original supervised vLLM service was stopped for testing. A restart after restoration failed with the same CUDA mismatch and the service remains exited. Full traceback: `original_service_restart.log`; install details: `install.txt`, `install_test_missing.txt`.
- Nonfatal imports warned that `nixl_ep_cu13.nixl_ep_cpp_torch214` was missing; tracebacks appear in warmup, benchmark and server logs. Target warmup checks still reported no JIT. BF16 startup also logged a `torch._dynamo` recompilation-limit warning for `grouped_topk`.
- Workspace is not volume-backed. Copy results elsewhere to survive recycle/destroy.

## Local diffs

- `local.diff` is empty after restoration. `moe_alt.diff` records the temporary change to MoE GEMV cutoff on GDDR from 64 to 8. The source file was restored after measurement. No local kernel rewrites or correctness fixes were made.
