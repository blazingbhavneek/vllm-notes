# Round 6 — hosted serving, CUDA timeline and Q8 decode (2026-10-30)

## Starting point

Round 5's changes were present in the working tree: direct routing for tiny
MoE decode, BF16-preserving non-gated ReLU² fusion, singleton-expert GEMV,
and the opt-in IQ2 dense BF16 cache. The model was started on visible GPU 1
under Nsight Systems, with CUDA graphs at batch sizes 1/2/4/8/16/32, context
4096, max 32 sequences, 1024 batched tokens, and Marlin for NVFP4 linears.
The local OpenAI endpoint was `127.0.0.1:8001`, model name
`nemotron-bolt-h200`. GPU 0 and port 8000 were not used.

The round 5 working changes remain uncommitted. Round 6 adds one more
experimental, narrowly dispatched Hopper optimization in
`vllm/model_executor/layers/quantization/modelopt_block_quant_triton.py`:
single-row Q8_0 GEMV uses one wider-K kernel instead of split-K partials plus
a reduction kernel for four model projection shapes. Dispatch requires SM90,
Q8_0, M=1, and an exact `(N,K)` match: `(256,4096)`, `(1024,4096)`,
`(4096,4096)`, or `(4096,1024)`. All other GPUs, codecs, batches and shapes
keep the previous path. Warmup passes the device major version into the same
config helper.

## Serving baseline, before the round 6 Q8 change

Controlled `vllm bench serve` random prompts, greedy temperature 0, 128 input
tokens, 256 output tokens, seed 0. Results are from the hosted round 5 model
and were taken before adding the Q8 dispatch below.

| Concurrency | Requests | Output tok/s | Median TPOT | P99 TPOT | Median TTFT | P99 ITL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 132.1 | 7.12 ms | 7.15 ms | 120.6 ms | 7.47 ms |
| 8 | 32 | 492.0 | 15.05 ms | 15.38 ms | 319.4 ms | 16.64 ms |
| 32 | 32 | 865.0 | 34.05 ms | 35.34 ms | 758.4 ms | 42.44 ms |

These are serving baselines for the `vllm bench serve` workload, not directly
comparable to round 5's offline `LLM.generate` throughput. The earlier actual
HTTP GSM8K greedy run scored 236/250 (94.4%), with zero 768-token truncations;
it used the round 5 server, before the Q8 dispatch. Do not attribute that score
to the round 6 change. No post-change full-model accuracy or serving A/B was
run before stopping.

Artifacts: `kernel-tests/rewrite/results/round6_serving_profile/`
contains the benchmark JSON, 250 answers and grades, Nsight Systems reports
and SQLite exports. `greedy_baseline_c1.json`, `greedy_baseline_c8.json`, and
`greedy_baseline_c32.json` are the controlled throughput baselines.

## CUDA timeline and SM resource view

The profile is process-scoped CUPTI activity (`cuda,nvtx,osrt`, CUDA graph
nodes). The single-request trace contains 126 graph replays, 118 used for the
steady-state summary, and 130,254 kernel launches overall. A steady graph has
985 kernels. Median graph span is 6.966 ms; the union of model-kernel intervals
is 6.671 ms, leaving 0.296 ms inside the graph with no model kernel running.
Between graph replays, the median gap with no model kernel is 0.069 ms. The
CUDA graph launch API call is about 1.50 ms, but overlaps queued GPU work and
must not be added as serial GPU idle time. The decode trace therefore shows
little host starvation; it does not say that all SMs are occupied while a
kernel is running.

Per steady decode graph, dense `_gemv_kernel` launches 176 times and sums to
2.222 ms of kernel duration; `_moe_gemv_kernel` launches 80 times and sums to
2.008 ms; Marlin launches 80 times and sums to 1.635 ms; and
`_splitk_reduce_kernel` launches 176 times and sums to 0.815 ms. Kernel sums
can overlap across streams, so these durations are not serial shares of graph
latency. The 176 reduction launches and the Q8 projection microbench results
motivated the one-kernel Q8 trial.

The concurrency-32 profile has a 32.159 ms median graph span, 31.806 ms union
kernel time and 0.355 ms inside-graph kernel idle. Its `_moe_gemm_kernel`
duration sums to 21.541 ms per graph (80 calls, overlapping work included),
Marlin to 2.325 ms and selective scan to 3.683 ms. This workload is dominated
by the large batched MoE GEMM; a GEMV-only improvement will not explain or
remove that cost.

The CUDA driver API loaded cached SM90 Triton cubins and queried register,
shared-memory and active-block limits. Examples from compiled configurations:

| Kernel configuration | Registers/thread | Threads/CTA | Dynamic shared memory | Theoretical resident warps |
|---|---:|---:|---:|---:|
| grouped MoE tile | 147 | 128 | 24 KiB | 12/64 (18.75%) |
| wide grouped MoE prefill tile | 244 | 256 | 170 KiB | 8/64 (12.5%) |

These are theoretical resource ceilings calculated by CUDA occupancy APIs,
not measured achieved occupancy. Grid size further limits available parallel
work. The 170 KiB wide prefill variant is resource-heavy; the actual M=1
decode grid launches many smaller kernels, so its resource figure should not
be generalized to decode.

The standalone CUDA toolkit probe compiled with CUDA 13.1 and passed a checked
streaming add on the visible H200 NVL (132 SMs). It measured 0.126 ms per
launch and 3194.69 GB/s effective for a 384 MiB read/write operation, with a
theoretical 8 resident probe blocks per SM. This is a CUDA/toolkit and broad
memory-path sanity check, not a model kernel or achieved-SM-occupancy result.

Nsight Compute's warp-stall and achieved-occupancy counters remain unavailable:
both the ordinary process and container root receive `ERR_NVGPUCTRPERM` from
the host driver. Container sudo did not change that. No host driver setting or
kernel parameter was changed. CPU perf sampling is also unavailable in this
container (`perf_event_open` restricted). Nsight Systems kernel timelines and
CUDA driver resource queries are the available evidence. CUPTI profiles are
process-scoped; they do not account for other GPU processes, so global GPU
idle time cannot be inferred from these reports.

## Kernel trials

The Q8 sweep used the same packed inputs, checked each output against the
existing BF16 cuBLAS-relative reference, then timed CUDA graph replays. The
default Q8 config and selected one-kernel candidates were:

| N × K | Existing split path | One-kernel candidate | Change |
|---|---:|---:|---:|
| 256 × 4096 | 17.38 µs | 3.52 µs | 4.9× |
| 1024 × 4096 | 11.14 µs | 5.57 µs | 2.0× |
| 4096 × 4096 | 17.38 µs | 11.36 µs | 1.5× |
| 4096 × 1024 | 7.81 µs | 4.67 µs | 1.7× |

Candidate tiles use BN=4, four warps, one stage, split=1, and BK=K except
N=4096 uses BK=1024. The actual LM head `(131072,4096)` was swept too; the
candidate did not win there, so it is deliberately not dispatched. Repeated
before/after measurements under isolated GPU conditions are still needed to
confirm these numbers because other GPU processes were present during the
session.

After changing dispatch, the full focused kernel file passed: **126 passed
in 42.70 s**, including cuBLAS-relative checks for all four shapes with
ordinary and extreme packed values, under graph replay. This verifies local
kernel correctness. A different split/reduction order can still change final
bits; measure full-model greedy outputs and GSM8K after the dispatch before
claiming model-level accuracy parity.

A Hopper multi-CTA GEMV experiment did not produce a usable IQ2 improvement:
best measured IQ2 full-K case was 21.09 µs against 22.74 µs for the split path,
while Q8's best was 5.57 µs against 11.11 µs. Increasing CTA cluster size
usually regressed; Triton compilation failed for `num_ctas=8` in its CTA
planning pass. No cluster configuration was retained.

## Exact handoff for the next optimization pass

1. Confirm the serving process is stopped before starting; check port 8001 and
   use `CUDA_VISIBLE_DEVICES=1`. GPU 1 had other resident work during this
   session, so first capture several minutes of `nvidia-smi dmon` or equivalent
   and wait for a quiet interval before comparing latency.
2. Re-run the three controlled greedy serving baselines at concurrency 1/8/32
   on the current code, then run the same workload after the Q8 change. Compare
   output tok/s and TPOT/P99; use repeated alternating A/B runs. Do not compare
   these numbers directly to offline `LLM.generate` results.
3. Run the same HTTP GSM8K test (`round6_http_accuracy.py`, 250 questions,
   concurrency 32, greedy, max 768) on pre-change and current builds, and save
   all responses. Compare token IDs/logprobs on a fixed prompt subset as well;
   numeric GSM8K grade alone can hide output drift.
4. Inspect the newly compiled Q8 cubins for registers, spills and theoretical
   occupancy; inspect PTX/SASS to see whether BK=4096 is unrolling a costly
   reduction. Sweep BK=512/1024/2048/4096 with BN=4/8 and warps=4/8 on all four
   shapes, warm compiles first, repeat/interleave runs, and report variance.
   Keep dispatch limited to shapes where the end-to-end serving A/B wins.
5. If Q8 sticks, add a deterministic unit test for selected configs and
   dispatch fallback, keep SM90 guarding, then run focused kernel/public model
   tests and pre-commit. Do not widen numeric tolerances.
6. Next bottleneck after Q8: optimize `_moe_gemm_kernel` and selective scan
   for concurrency 8/32, measured with representative expert-count and token
   distributions. The c32 Nsight trace says large MoE GEMM dominates; address
   its actual CTA/resource limit, not a synthetic 512-expert case.
7. For measured warp stalls, achieved occupancy, instruction issue and
   eligible warps, the host administrator must enable GPU performance-counter
   access. The current container cannot do this itself. Until then, use the
   saved Nsight Systems profiles and CUDA occupancy API, and label theoretical
   residency as such.

Useful files:

- `kernel-tests/round6_bench_serve.sh` — controlled `vllm bench serve` command.
- `kernel-tests/round6_http_accuracy.py` — live HTTP GSM8K check.
- `kernel-tests/round6_trace_analyze.py` — stream/kernel and idle-interval
  analysis; unions overlapping kernel intervals.
- `kernel-tests/round6_timeline_html.py` — standalone interactive timeline.
- `kernel-tests/round6_static_occupancy.py` — CUDA driver queries on compiled
  Triton cubins; reports theoretical occupancy.
- `kernel-tests/round6_cluster_gemv.py`, `round6_q8_sweep.py` — rejected and
  retained kernel experiments.
- `kernel-tests/cuda_round6_probe.cu` — CUDA toolkit and streaming probe.
- `kernel-tests/rewrite/results/round6_serving_profile/serve.3.timeline.html`
  — inspect per-stream kernels and hover resource metadata.

The API server used for profiling was stopped at the end of the round; port
8001 is free. No commits or pushes were made. The vLLM production changes are
still in the shared working tree and uncommitted.

## Committed (2026-11-01)

The round 5 and round 6 working tree is now on `origin/feature/modelopt-block-triton`:

- `f1ec2fb051` — the H200 diff exactly as it was (rounds 5-6, including the Q8 dispatch).
- `d71cc09daf` — warmup fix. `MoEActivation` is a plain `Enum`, so
  `module.activation == "relu2_no_mul"` was always False. Warmup compiled an
  unused `APPLY_RELU2=False` GEMM1 variant, and the fused-ReLU² variant that
  Bolt uses compiled during graph capture, or on the first decode in eager mode.
  `kernel-tests/check_warmup_relu2.py` fails before the fix and passes after it.

On the 5060 Ti (SM120), `test_modelopt_block_triton.py` gave 125 passed and
1 skipped (the int64 test needs 12 GiB free), and `test_modelopt.py -k block`
gave 51 passed. The binary profiles (`*.json.gz`, `*.nsys-rep`, `*.sqlite`,
`serve.3.timeline.html`) were not transferred from the H200 box and are not
in this repo.
