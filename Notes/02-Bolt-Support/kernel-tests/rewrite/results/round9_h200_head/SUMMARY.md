# Round 9 — H200 at PR head

Raw logs and JSON are in `round9_h200_head/`. All GPU work used `CUDA_VISIBLE_DEVICES=1`.

## Tests

| Suite | Passed | Failed | Deselected | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Kernel tests | 192 | 0 | 0 | 96.76 |
| ModelOpt block tests | 54 | 1 | 75 | 167.86 |

The single block-test failure is `test_modelopt_block_triton_tiny_nemotron_h[bfloat16]`, the known tiny-Mamba eager-vs-compiled logprob mismatch (documented in the parent notes); it is deterministic and not a block-kernel mismatch.

## Warmup / inference-time JIT audit

- ModelOpt block Triton configs warmed at startup: **77**
- Triton kernels JIT-compiled during inference: `_causal_conv1d_fwd_kernel`

- `torch._dynamo` hit its 16-recompile limit for `grouped_topk` because the MoE layers have different `e_score_correction_bias` sizes (240/304/368/432/496). This is a known nonfatal model-side warning, also seen on the L40S; it is not block-kernel code.

## SM90 single-row Q8_0 dispatch A/B

GPU: NVIDIA H200 NVL SM9.0

| N | K | Singleton µs | Generic split-K µs | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4096 | 3.52 | 8.70 | 2.47× |
| 1024 | 4096 | 5.66 | 11.23 | 1.98× |
| 4096 | 4096 | 11.20 | 17.25 | 1.54× |
| 4096 | 1024 | 5.06 | 7.90 | 1.56× |
| 131072 | 4096 | 200.00 | 200.06 | 1.00× |

## Serving (`vllm bench serve`, random 128 in / 256 out, greedy)

| c | Run | Output tok/s | Median TPOT ms | P99 TPOT ms | Median TTFT ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 136.3 | 6.83 | 6.83 | 126.4 |
| 1 | 2 | 137.2 | 6.82 | 6.82 | 124.7 |
| 1 | median | 136.8 | | | |
| 8 | 1 | 473.3 | 15.18 | 15.51 | 316.0 |
| 8 | 2 | 488.4 | 15.18 | 15.55 | 315.3 |
| 8 | median | 480.9 | | | |
| 16 | 1 | 624.7 | 23.72 | 24.80 | 520.7 |
| 16 | 2 | 659.9 | 22.98 | 25.02 | 381.7 |
| 16 | median | 642.3 | | | |
| 32 | 1 | 888.0 | 33.41 | 34.88 | 675.9 |
| 32 | 2 | 895.0 | 32.87 | 34.20 | 742.3 |
| 32 | median | 891.5 | | | |

## GSM8K-250 (BF16, greedy, HTTP)

- Correct: **231/250**
- Truncated at 768 tokens: 2
- Elapsed: 75.7s
- Dataset sha256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`

## Kernel benchmark

Compared with the round-5 H200 run on the same box (`benchmark_h200_round5_reference.txt`); both used the same benchmark and default MoE configs.

- MoE at M ≤ 8: **2.99–3.78×** faster than round 5.
- MoE at M ≥ 256: **2.25–3.39×** faster.
- Dense at M = 1 (excluding `lm_head`): **1.26–2.12×** faster.
- Dense at M ≥ 256: **0.90–1.00×** (neutral; a few rows 3–7% slower).
- `lm_head` at M = 256: **1.00×** (unchanged; the round-5 M = 512 point is an outlier).
- Full per-shape table: `microbench_vs_round5.tsv`.

Raw benchmark: `benchmark_h200_head.txt`.

