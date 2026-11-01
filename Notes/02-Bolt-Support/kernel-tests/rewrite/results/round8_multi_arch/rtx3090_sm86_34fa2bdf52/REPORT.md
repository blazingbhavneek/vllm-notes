# NVIDIA GeForce RTX 3090 (SM86) — 2026-11-01

- Driver 595.99.02; PyTorch 2.14.0+cu132 (CUDA 13.2); Triton 3.8.0; vLLM editable install at `34fa2bdf52`.
- `git log -1 --oneline`: `34fa2bdf52 [Quantization] ModelOpt block Triton: MoE GEMV up to 8 rows on GDDR too`.
- 82 SMs; 23.6 GiB CUDA-visible VRAM; 384-bit bus; 6 MiB L2; copy bandwidth 786 GiB/s; GDDR.
- Active dense cutoffs: IQ2_XXS M=128, Q8_0 M=256; MoE GEMV through M=8.
- Model: `/workspace/bolt`, obtained with the requested `hf download --local-dir` command. GSM8K test data contained 1,319 rows.
- Before setup the preinstalled vLLM service and orphaned server/engine processes occupied about 20 GiB. They were stopped by PID; the GPU was otherwise clear for tests. Initial disk availability was 58 GiB (approximately 62 GB).

## Correctness

| Check | Result |
|---|---|
| Kernel tests | **192 passed, 0 failed, 0 skipped**, 124.18 s. Slowest: IQ2 bit-exact decode 2.26 s; IQ2 MoE reference, FP16 2.00 s and BF16 1.97 s. Full 15 slowest are in `kernel_tests.txt`. |
| GSM8K, BF16 | **231/250**, 2 length truncations; no empty answers. One truncated response repeats the pool problem setup. |

The revised plan omitted the ModelOpt block, cache, and warmup checks. They had already run before the plan changed: block 55 passed; cache reruns passed 3/3; both warmup scripts reported no inference JIT. Their logs remain in the archive for transparency. No FP16 full-model run was made.

## Dense cutoffs — requested BF16 grid

The table reports the first sampled M with `deq < fused` for each layer; this is not necessarily a monotonic crossover. Times are microseconds. See `cutoff_dense.txt` for every row.

| Layer (codec) | First M | Fused | Dequant + cuBLAS | Worst op/best (M) |
|---|---:|---:|---:|---:|
| qkv_proj (Q8_0) | 256 | 221.4 | 214.8 | 1.13 (384) |
| o_proj (Q8_0) | 512 | 379.4 | 339.4 | 1.13 (384) |
| fc1_latent (Q8_0) | 128 | 44.0 | 35.3 | 1.24 (128) |
| fc2_latent (Q8_0) | 96 | 37.0 | 35.1 | 1.26 (384) |
| shared_up (IQ2_XXS) | 48 | 153.0 | 135.6 | 1.44 (96) |
| shared_down (IQ2_XXS) | 96 | 261.0 | 166.0 | 1.57 (96) |

Suggested IQ2 cutoff: **M=96** instead of 128; both IQ2 layers favor dequantization there, with substantial dispatch gaps. Q8 results differ by layer and are nonmonotonic: keep **M=256** as a provisional shared cutoff, with a notable 1.24/1.21 gap on the latent layers at M=128 and 1.26 on fc2 at M=384. These are microbenchmark suggestions, not code changes. MoE calibration was removed from the revised plan; its earlier results are retained in the logs but are not part of the requested final calibration.

## Full model — BF16

The initial 0.97 graph-enabled run loaded weights but failed KV-cache planning. The successful retry used the handoff's base serve arguments plus `--gpu-memory-utilization 0.98 --max-num-seqs 16 --enforce-eager` and `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}'`, `--max-model-len 4096`, `--max-num-batched-tokens 1024`, and `--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"marlin"}}'`. Eager mode disables graph capture and JIT warmup. No CPU-offload flag was used.

The accuracy server reported: `Model loading took 21.7 GiB memory and 14.919671 seconds`; available KV cache **0.9 GiB / 7,168 tokens**, maximum 1.75 full-length requests; startup approximately **113 s** to application startup. The server was stopped to run the exact revised dense sweep, then restarted with identical flags for throughput: loading **21.7 GiB / 14.738152 s**, KV cache **1.01 GiB / 7,936 tokens**, startup approximately **56 s**. The varying KV allocation is reported separately rather than treated as the same run.

| Run | GSM8K-250 | Truncated | c=1 output tok/s | c=8 | c=16 | Mean TPOT c=1 / c=8 / c=16 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 eager | 231 | 2 | 11.92 | 53.50 | 53.14 | 82.92 / 84.05 / 84.36 |

All benchmark requests succeeded: 12 at c=1 and 32 at each higher concurrency. At c=16 the log showed KV cache at 100% and requests waiting; results reflect the memory-constrained eager configuration. The model log selected Triton for IQ2 linear, Q8 linear, and IQ2 MoE, and `MarlinNvFp4LinearKernel` for NVFP4. No `during inference:` JIT lines appeared in either serving log; because eager disables JIT warmup, this absence alone does not prove no kernels compiled during inference.

Examples from `gsm8k_bf16.json`:

- Index 0, correct: “The remaining eggs are 16 - 7 = 9 eggs”; answer 18.
- Index 1, correct: “The total number of bolts is 2 + 1 = 3.”
- Index 2, incorrect: “Josh made a loss of $10,000.” Gold answer is 70,000; the script extracted 10,000.
- Index 214 ended at the length limit while repeatedly restating the pool setup; index 205 also hit the limit. Both complete texts are in the JSON.

## Problems

- The original graph-enabled startup failed after model load with the following root-cause traceback excerpt (`serve_bf16.log` contains the full traceback):

  ```text
  File "vllm/v1/engine/core.py", line 339, in _initialize_kv_caches
      kv_cache_configs = get_kv_cache_configs(
  File "vllm/v1/core/kv_cache_planning.py", line 1550, in get_kv_cache_configs
      _check_enough_kv_cache_memory(
  File "vllm/v1/core/kv_cache_planning.py", line 57, in _check_enough_kv_cache_memory
      raise ValueError(
  ValueError: No available memory for the cache blocks.
  ```

  The specified 0.98 eager retry succeeded; reducing model length was unnecessary. The log also contains allocation OOM warnings during loading, including on the successful retry.
- Test collection initially lacked `tblib`; installed `tblib==3.2.2`. The editable install upgraded PyTorch to CUDA 13.2 while preinstalled TorchAudio targeted CUDA 13.0. The matching `torchaudio==2.14.0` was unavailable from the configured index, so the incompatible optional `torchaudio==2.11.0+cu130` was removed. See `missing_deps.txt` and `kernel_collection_torchaudio.txt`.
- The optional `nixl_ep` extension emitted `ModuleNotFoundError: No module named 'nixl_ep_cu13.nixl_ep_cpp_torch214'`; tests and serving continued. FlashInfer warned that its preinstalled cubin/JIT caches did not match the upgraded package; benchmark timing fell back from CUPTI to CUDA events. Full messages are in the logs.
- The first c=1 throughput attempt was interrupted after the revised plan arrived, to free the GPU for the exact dense grid; all three reported throughput results come from the restarted server. The accuracy result comes from the earlier server using the same flags.

## Local diffs and machine state

`local.diff` is empty; no source changes or git writes were made apart from the requested fetch and branch checkout. Phase E and FP16 serving were skipped. The test server is stopped; `nvidia-smi` showed no GPU processes and 1 MiB used. The instance remains running. The workspace is not a persistent host volume on this instance.
