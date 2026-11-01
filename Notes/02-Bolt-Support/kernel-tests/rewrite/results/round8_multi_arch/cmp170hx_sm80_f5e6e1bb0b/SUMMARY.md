# NVIDIA CMP 170HX (SM80) — 2026-11-01

- Driver 615.71.09; PyTorch 2.14.0+cu132 / CUDA 13.2; Triton 3.8.0; vLLM `feature/modelopt-block-triton` commit `f5e6e1bb0b`.
- 74 SMs; 63.4 GiB VRAM; 4096-bit bus; 32 MiB L2; measured copy bandwidth 1483 GiB/s; HBM class.
- Cutoffs in effect: IQ2_XXS 8; Q8_0 64; routed MoE GEMV through M=8.
- Active `/venv/main` was used, as requested. The original Bolt snapshot in `/workspace/.hf_home` was used for inference. The original Qwen supervisor service was stopped during tests and restarted afterward; its `/v1/models` endpoint was verified after restart and returned `Qwen/Qwen3.5-9B`. The active environment remains changed by the requested editable install and dependency repairs.

## Correctness

| Check | Result |
|---|---|
| Kernel tests | 192 passed, 0 failed, 0 skipped |
| `test_modelopt -k block` | 55 passed, 0 failed, 75 deselected; 7 warnings |
| Dense-cache test ×3 | pass / pass / fail; full failure in `dense_cache_3.txt` |
| Warmup JIT, BF16 / FP16 | clean / clean: 0 inference-time compilations for the checked ReLU2 GEMV path |

The dense-cache failure is the known flaky test noted in the handoff. Its third run reported `AssertionError` at `tests/quantization/test_modelopt.py:1929` comparing cached and uncached outputs. The original full block suite included the same test and passed. No attempt was made to chase it.

## Cutoffs

- Dense IQ2_XXS: dequantization beat fused on both shared layers at M=8, the smallest M in the initial sweep; crossover is at or below 8, not resolved below that. Current cutoff 8 is supported.
- Dense Q8_0: at M=48 and 64 the four shapes split 2–2 between paths; at M=96 dequantization wins on all four for both dtypes. The finest measured majority crossover is 96. Candidate cutoff **96**, subject to serving validation: one global threshold trades attention projections against the latent projections. Worst fine-sweep `op/best` rows include FP16 qkv_proj M=64 **1.31**, FP16 o_proj M=64 **1.29**, BF16 fc1_latent M=48 **1.29**, and BF16 fc2_latent M=48 **1.26**. At M=48 the latent projections prefer dequantization, so 96 does not fix every row. Full measurements are in `cutoff_dense.txt` and `cutoff_dense_fine.txt`.
- Routed MoE, E=512: GEMV won through **M=128**, the largest tested, for both BF16 and FP16; this does not establish a crossover above 128. This microbenchmark disagreed with serving. Keep the current HBM cutoff of 8 based on the following single serving measurements:

| BF16 concurrency | Current cutoff 8: median TPOT | Alternate cutoff 64: median TPOT | Change |
|---|---:|---:|---:|
| 8 | 34.15 ms | 34.19 ms | +0.1% |
| 16 | 48.10 ms | 55.44 ms | +15.3% |
| 32 | 74.92 ms | 95.13 ms | +27.0% |

The concurrency-16 baseline was collected in a separate restart after the alternate run, using the same benchmark settings. The per-layer reference is in `benchmark.txt`. Benchmarks used CUDA-event fallback because CUPTI was unavailable; the tool also warned that cold-L2 benchmarking was disabled for these calls.

## Full model

| run | GSM8K-250 | trunc | c=1 tok/s | c=8 | c=32 | median TPOT c=1 | load GiB | startup s | inference JIT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF16, cutoff 8 | 230/250 | 3 | 80.50 | 218.61 | 394.53 | 11.62 ms | 21.7 | ~232 | no branch kernels; `_causal_conv1d_fwd_kernel` |
| FP16, cutoff 8 | 3/250 | 250 | 81.53 | not run | 320.76 | 11.55 ms | 21.7 | ~198 | no branch kernels; `_causal_conv1d_fwd_kernel` |
| BF16, cutoff 64 | 229/250 | 3 | not run | 218.40 | 316.25 | not run | 21.7 | ~214 | no branch kernels; `_causal_conv1d_fwd_kernel` |

Startup is elapsed wall time from server launch to API-ready log, approximate to seconds. Model-loading times within that were 20.42 s, 19.50 s, and 20.13 s respectively. BF16 c=16 measured 311.44 output tok/s; alternate c=16 measured 271.19 output tok/s. All throughput benchmark requests succeeded. FP16 throughput is not evidence of useful output quality.

All three runs logged Triton IQ2_XXS linear, Q8_0 linear, and IQ2_XXS MoE; NVFP4 used Marlin. The inference JIT monitor reported only the non-branch causal convolution kernel above. All three BF16 truncations occurred at GSM8K indices 153, 205, and 214; the reference had none. Examples from the saved JSON:

- BF16 index 0: “The remaining eggs are 16 - 7 = 9 eggs”; its final answer was 18, matching the gold answer.
- BF16 index 1: “In total, the robe takes 2 + 1 = 3 bolts of fiber. Answer: 3”, matching the gold answer.
- FP16 index 1: repeated `<think></think>` until `finish_reason=length`; gold answer 3, no extracted prediction. FP16 index 0 repeatedly copied the question, predicted 2, and truncated; gold answer 18. Some FP16 outputs included a correct number before looping, accounting for the three scored correct.

## Problems

- **FP16 correctness collapse:** 3/250, with all 250 outputs truncated and repeated text. This is a finding, not a passing result. Cause is not established; no NaN diagnosis was made. The captured minimal request and response are `fp16_repro_request.json`, `fp16_repro_response.json`, and `fp16_repro.md`; full outputs are `gsm8k_fp16.json`, `gsm8k_fp16.log`, and `serve_fp16.log`. The failure is output-level and raised no Python exception, so there is no failure traceback. No code fix was attempted.
- **BF16 truncations:** 3/250 in both default and alternate runs, compared with zero in the reference. Full answers are in `gsm8k_bf16.json` and `gsm8k_bf16_moe_alt.json`.
- **Environment and setup:** I missed the already-downloaded Bolt snapshot and created a duplicate, consuming disk space. The requested isolated editable install then failed with `No space left on device` while copying `torch/lib/libtorch_cuda.so`. A retry using `--no-build-isolation` first failed with `ModuleNotFoundError: No module named 'setuptools_rust'`; after installing that, another install ran out of space while creating a Transformers directory and left the environment partly upgraded. I verified duplicate files by SHA-256 against the original snapshot and removed only matching copies, reinstalled PyTorch, and completed the editable install with `--no-build-isolation`. `install.txt` contains the complete errors. The first cleanup command was rejected by automatic command review because it used `rm -rf`; the later verified file-by-file cleanup succeeded.
- The first pytest import lacked `tblib`; it was installed. The initial PyTorch upgrade also mismatched the existing CUDA 13.0 TorchAudio. TorchAudio was upgraded to a CUDA 13.2 build and PyTorch restored to 2.14.0; package actions are in `missing_modules.txt`. One partial kernel run with PyTorch 2.15 was intentionally interrupted and preserved in `kernel_tests_interrupted_torch215.txt`; the reported 192-pass run used PyTorch 2.14.
- An optional `nixl_ep` extension was present but not importable for PyTorch 2.14. This warning appeared during warmup and serving; the tested paths continued. The start of the traceback is:

  ```text
  Module nixl_ep was found but failed to import
  Traceback (most recent call last):
    File "/workspace/vllm/vllm/utils/import_utils.py", line 415, in _has_module
      importlib.import_module(module_name)
    File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
      return _bootstrap._gcd_import(name[level:], package, level)
    ...
    File "/venv/main/lib/python3.12/site-packages/nixl_ep_cu13/__init__.py", line 27, in <module>
      _nixl_ep_cpp = importlib.import_module(f".nixl_ep_cpp_torch{_torch_mm}", __package__)
  ModuleNotFoundError: No module named 'nixl_ep_cu13.nixl_ep_cpp_torch214'
  ```

  The full traceback is in `warmup_bf16.txt`, `warmup_fp16.txt`, and serving logs. Installed FlashInfer cubin/JIT-cache packages were version 0.6.18 while FlashInfer was 0.7.2.post1, so they were ignored; sampling kernels were compiled on demand.

## Local diffs

- `local.diff` is empty: no source changes remain.
- `moe_alt.diff` records the temporary HBM MoE threshold 8 → 64 and the opposite GDDR value used for the optional serving comparison; the source was restored afterward.
