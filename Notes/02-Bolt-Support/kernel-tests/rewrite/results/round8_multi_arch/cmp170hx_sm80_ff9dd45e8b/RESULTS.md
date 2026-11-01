# Bolt vLLM results — NVIDIA CMP 170HX (SM80)

Date: 2026-11-01. Branch: feature/modelopt-block-triton, remote commit ff9dd45e8bab5dc23847d3b86ed0f5eb1be8fdb6. GPU: 74 SMs, 63.4 GiB VRAM, HBM. Driver 615.71.09, PyTorch 2.14.0+cu132, Triton 3.8.0.

## Full-model results

One measurement per dtype and concurrency, using the full Bolt checkpoint. TPOT is median milliseconds per output token; throughput is output tokens per second.

| Dtype | GSM8K-250 | Truncated | c=1 tok/s (TPOT ms) | c=8 tok/s (TPOT ms) | c=32 tok/s (TPOT ms) | Load | Startup |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 235/250 | 0 | 81.14 (11.50) | 219.44 (34.05) | 394.96 (74.94) | 21.7 GiB | 196.09 s |
| BF16 | 231/250 | 3 | 80.31 (11.68) | 218.55 (34.18) | 402.64 (73.26) | 21.7 GiB | 202.08 s |

BF16 truncations were questions 205, 214 and 241. Both modes used Triton IQ2_XXS linear, Q8_0 linear and IQ2_XXS MoE, with Marlin for NVFP4. The only reported inference-time JIT was upstream _causal_conv1d_fwd_kernel; no branch kernel JIT was observed.

## Fix and checks

The pulled remote commit fixes NemotronH FP16 MoE scaling. On the earlier commit, the model did not compensate for division of the shared-expert output by five. FP16 scored 3/250 and truncated all 250 answers; the new remote commit scored 235/250 with no truncations. Earlier BF16 scored 230/250 with three truncations; new BF16 scored 231/250 with three. Previous BF16 median TPOT was 34.15 ms at c=8 and 74.92 ms at c=32; new results above show no observed slowdown in these single runs.

- New NemotronH FP16/BF16 regression: 1 passed.
- ModelOpt block Triton kernel suite: 192 passed.
- ModelOpt block tests: 55 passed, 75 deselected, 7 warnings.
- Dense-cache helper-based test: 10/10 passed. An earlier numeric-tolerance attempt passed 9/10 and failed once on a 0.137 log-probability difference. A tighter initial tolerance caused 54 passes and 1 failure in the block suite.

## Local change

local.diff contains a test-only dense-cache comparison change. It uses the existing check_logprobs_close helper and checks output lengths. The helper checks mutual top-k membership at the first divergent token; it does not compare numeric log-probability values and does not compare later divergent histories. No local model runtime, kernel, architecture-specific tuning, or performance improvement was added. The FP16 fix is the pulled remote commit.
