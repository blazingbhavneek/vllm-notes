# H200 GPU 1 optimization results — 2026-10-30

## Result and scope

The final code adds direct routed-row decode, exact ReLU² fusion, adaptive
singleton-expert GEMV, and an opt-in dense IQ2 BF16 cache. The remaining tile,
warp, representation and global-cutoff candidates failed to improve the
chosen balance of workloads. This is a measured local plateau, not a claim
that all possible CUDA implementations have reached a hardware limit.

All GPU work ran with `CUDA_VISIBLE_DEVICES=1` on the H200 NVL. GPU 0 and the
protected port-8000 server were untouched. No clock/power settings changed.
GPU 1 initially had about 110 GiB free, with about 30 GiB owned by other work.

Baseline source: vLLM `779b13849a`, with original sorted/unfused experts and
checkpoint-sized dense execution. `bolt_baseline_experts.py` retains that
implementation; the benchmark worker explicitly disables the new singleton
path and restores the old BK256 sparse GEMM tile. The broken warmup-only
logical K/output dimensions are corrected in both arms.

## Real-model performance

Model: `/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`.
Venv: `/home/seigyo/vllmServer/.venv`. Both arms use compiled execution and
CUDA graphs, seed 0, max context 4096, max sequences 32, max batched tokens
1024, capture sizes 1/2/4/8/16/32, GPU memory utilization 0.55, and Marlin for
NVFP4 Mamba linears only. IQ2/Q8 backend selection remains automatic.

These are medians of three end-to-end `LLM.generate` runs, 128 output tokens
per request with `ignore_eos=True`, including input processing and scheduling.
Distinct prompts use the same deterministic GSM8K question sets in each arm.
The original implementation was measured both before and after the first
optimized run; the table uses the repeated control, which also has distinct
prompts. The first control's repeated throughput was 129.2/451.1/647.5/1102.7.

| Workload | Batch | Original tok/s | Optimized tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Repeated prompt | 1 | 129.0 | 133.3 | +3.3% |
| Repeated prompt | 8 | 450.4 | 493.6 | +9.6% |
| Repeated prompt | 16 | 648.6 | 725.3 | +11.8% |
| Repeated prompt | 32 | 1080.0 | 1291.9 | +19.6% |
| Distinct prompts | 1 | 120.9 | 124.5 | +3.0% |
| Distinct prompts | 8 | 426.0 | 469.2 | +10.1% |
| Distinct prompts | 16 | 323.5 | 463.7 | +43.4% |
| Distinct prompts | 32 | 451.1 | 568.8 | +26.1% |

The opt-in cache adds 3.28 GiB (80 shared dense matrices). It is reused only
where the old code already dequantized to BF16; small-token GEMV ignores it.
Q8, routed-expert storage, and automatic b12x selection are unchanged.
Startup: original control 93.3s, optimized
91.1s; compilation-cache state affects these numbers.

The actual checkpoint has 12 MoE layers with 240 experts, 6 with 304, 9 with
368, 10 with 432, and 3 with 496. Some earlier notes called it a 512-expert
model; 512-expert microbenchmarks are synthetic stress cases.

## Accuracy

First 250 GSM8K test questions, greedy raw Q/A prompts, max 768 generated
tokens; exact final-number numeric comparison, normalizing equivalent strings
such as `18` and `18.0`:

- Original: **228/250 (91.2%)**,
  2 responses hit the output limit.
- Optimized: **235/250 (94.0%)**,
  0 responses hit the output limit.

This is an implementation parity check, not NVIDIA's published evaluation or
an official few-shot lm-eval score. The small observed improvement is not a
claim of a general quality increase. Earlier progress messages used literal
numeric strings without canonicalization; both scoring definitions and all
responses are retained in the artifacts.

There is no activation requantization or lossy weight repack. The singleton
path rounds decoded weights to BF16 at the same point as the tensor-core path
and accumulates in FP32; its sum order can change the last bits. ReLU² fusion
first rounds the GEMV result to BF16, preserving the original intermediate.
The dense cache is exactly the old dequantized matrix. Cache parity tests use
exact equality, including generated tokens/logprobs through `LLM`.

## Verification and existing limitations

- **127 focused tests passed**: 118 kernel/graph/routing tests plus 9 public
  cache-parity/backend-selection tests. Existing accuracy thresholds retained.
- Real checkpoint layer checks passed. Direct routing/fusion matched the
  original MoE exactly at M=1/8/64 on the real-weight parity case.
- Pre-commit checks passed on all changed vLLM files, including mypy and CI
  test tethering. A new L4 CI step selects the cache-parity regression test.
- New sparse BK128 configuration: 24 KiB shared memory, 147 SM90 registers,
  no spills. Cross-compilation covers SM80/86/89/90/100/120.
- **Existing wide-prefill limitation retained:** IQ2 BM128/BN128/BK64/w4
  spills 60 bytes on SM90. The original kernel independently reproduces it.
  w4 measured about 5.1 ms per full MoE at M4096; w8 about 6.9 ms, w16 7.9 ms,
  and smaller/split output tiles were also slower. Preserving the fastest
  original launch follows the current speed objective. The dispatch report
  now exits nonzero for this known case instead of silently succeeding.
- **Existing test failure:** the older tiny model's eager-versus-compiled
  Mamba logprob comparison fails on the original expert implementation too.
  It remains in the suite, without skip/xfail or a widened threshold. The new
  cache-parity test compares identical execution modes on the same model and
  passes exactly. See `e2e-original-check.log` and `public-cache-parity.log`.
- The upstream causal-conv kernel can still JIT on a new prefill shape. The
  new block-kernel specializations are warmed before graph capture.

## Why these choices

Luna agents inspected llama.cpp, TensorFold and Strata; the parent did not
read clone sources. Their packed-register techniques motivated the exact
BF16x2/PRMT prototype. It improved standalone expert-shaped GEMVs, but only
about 6% for the whole M1 MoE and regressed M8, so it remains experimental.

The old 1.7x expanded-plane claim was invalid: a rank/broadcast bug addressed
rows on the K-group axis, and an encoding check never checked that GEMV's
outputs. Combined scales were also rounded to FP16. Corrected exact FP32
planes were slower on the important shared-expert decode shapes. The loader
experiment is archived as a patch, not retained in production.

Raising the global GEMV cutoff to 32 gave only about 724 tok/s on repeated
batch-32 requests versus the original control's 1103. Adaptive selection
instead uses GEMV only when the aligned expert tile has a single valid row;
reused expert weights stay on GEMM. Coarse sweeps covered BM16/32/64,
BN16/32/64, BK64/128/256, 2/4/8 warps, and 1/2 stages. BK128/BN32/w4 was
selected for general distinct-request workloads. BK256 was faster for the
intermediate repeated-batch-16 case, but lost on the main distinct batches.

## Reproduce and run

Ready-to-run launch recipe (port 8001, GPU 1 by default):

```bash
bash /home/seigyo/vllmServer/vllm-notes/Notes/02-Bolt-Support/kernel-tests/run_bolt_h200.sh
```

No server was left running. The optimization changes remain uncommitted.

Benchmark from `/home/seigyo/vllmServer/vllm`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-project /home/seigyo/vllmServer/.venv/bin/python \
  ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/bolt_saturation.py \
  --dense-cache --linear-backend marlin --eval-count 250 --diverse-bench \
  --batches 1,8,16,32 --out /tmp/bolt-optimized250.json
```

For the original control, add the notes `kernel-tests` directory to
`PYTHONPATH`, replace `--dense-cache` with `--baseline`, and keep other flags.
`collect_round5.py` archives outputs/logprobs as gzip, timings, paired grades,
the production diff, and metadata in
[`round5_h200_gpu1`](kernel-tests/rewrite/results/round5_h200_gpu1/summary.json).
See [`ROUND5-EXPERIMENTS.md`](kernel-tests/rewrite/ROUND5-EXPERIMENTS.md) for
microbenchmark commands and rejected variants.
