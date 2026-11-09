# Round 9 — H200 validation at PR #65003 head

Commit: `f533470e67383392d21dfdef52397c468e5569b4`
(`pr/modelopt-block-triton`, one commit on top of upstream `e0d00e4b8e`)

All GPU work ran with `CUDA_VISIBLE_DEVICES=1` on the H200 NVL. GPU 0 was
shared with another tenant and was not used or modified. Port 8000 was not
touched. The model is
`/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`.

## Environment

- Python 3.13.11, PyTorch 2.14.0+cu130, Triton 3.8.0, vLLM
  `0.33.0rc1.dev448+gf533470e6`
- GPU 1: H200 NVL, SM90, 132 SMs, 139.8 GiB, 6016-bit HBM, 60 MiB L2
- GPU 1 already had about 30 GiB owned by other work at the start. Utilization
  was 0% at launch; `nvidia-smi` before/after and the launch log are included.
- `.venv/bin` must be on `PATH`: FlashInfer's sampler JIT calls `ninja`.

## What was run

See `repro/commands_round9.sh` for the exact commands. In order:

1. `pytest tests/kernels/quantization/test_modelopt_block_triton.py -q`
2. `pytest tests/quantization/test_modelopt.py -q -k block`
3. SM90 single-row Q8_0 dispatch A/B (`q8_singleton_ab_head.py`)
4. `benchmarks/kernels/benchmark_modelopt_block_triton.py`
5. Full Bolt server: BF16, compiled, CUDA graphs 1/2/4/8/16/32, 4096 context,
   32 sequences, 1024 batched tokens, `gpu_memory_utilization=0.55`, IQ2 dense
   cache on, NVFP4 Mamba linears pinned to Marlin
6. `vllm bench serve` at c=1/8/16/32, two alternating passes
7. GSM8K-250 over HTTP, greedy, 32 requests at a time, max 768 tokens
8. Warmup / inference-time-JIT audit of the server log

## Results

Full tables in `SUMMARY.md`; raw logs/JSON sit next to this file.

- Kernel suite: **192 passed** in 96.76 s.
- ModelOpt block suite: **54 passed, 1 failed, 75 deselected** in 167.86 s.
- SM90 Q8_0 singleton dispatch: **1.54×–2.47×** over the generic split-K path on
  the four dispatched shapes; `lm_head` unchanged at 1.00×.
- Serving (random 128 in / 256 out, greedy), median of two passes:
  **136.8 tok/s** at c=1, **480.9** at c=8, **642.3** at c=16, **891.5** at c=32.
  TPOT medians 6.8 / 15.2 / 23.0 / 33.1 ms.
- GSM8K-250 over HTTP: **231/250 (92.4%)**, 2 truncated at 768 tokens, 75.7 s.
- Kernel microbenchmark vs the round-5 H200 reference on the same box (same
  benchmark, default MoE configs, copy bandwidth 2330 vs 2376 GB/s):
  - MoE at M ≤ 8: **2.99–3.78×** faster.
  - MoE at M ≥ 256: **2.25–3.39×** faster.
  - Dense at M = 1 (excluding `lm_head`): **1.26–2.12×** faster.
  - Dense at M ≥ 256: **0.90–1.00×** (neutral; a few rows 3–7% slower).
  - `lm_head`: **1.00×** at M = 256 (unchanged).
  - Per-shape table: `microbench_vs_round5.tsv`.
- Startup warmed **77** ModelOpt block Triton configs (the full model has more
  shapes than the tiny test's 39). No ModelOpt block kernel JIT-compiled during
  inference. One inference-time JIT warning for the Mamba
  `_causal_conv1d_fwd_kernel`, a pre-existing Mamba warmup gap unrelated to
  this PR; see `warmup_jit_audit.txt`.
- One nonfatal `torch._dynamo` 16-recompile-limit warning for `grouped_topk`,
  caused by the model's heterogeneous MoE sizes (240/304/368/432/496). It is
  known from the L40S round-8 run and is not block-kernel code.

These serving numbers are in line with the round-6 H200 runs (131–132 tok/s at
c=1, 492 at c=8, 865 at c=32) within run-to-run noise; c=8 is slightly below the
best round-6 greedy run and c=1/c=32 slightly above.

GSM8K is 231/250 here versus 236/250 on the earlier H200 run. Five questions is
~1.4 binomial standard deviations at n=250 (σ≈3.6), and this run had 2
length-truncated answers versus 0 before, so this is not evidence of an
accuracy regression; it is within sampling noise and the GPU-1 contention.


## Known non-kernel failure

`test_modelopt_block_triton_tiny_nemotron_h[bfloat16]` fails deterministically
on this H200 box: eager and compiled logprobs diverge at prompt 2. This is the
existing tiny-Mamba eager-vs-compiled mismatch. It is not a block-kernel
mismatch: the round-5 H200 log `e2e-original-check.log` (parent notes tree)
shows the baseline worker (the original expert implementation) failing the same
comparison on the same H200, and the notes record it in `04-progress-log.md` and
`09-round8-multi-arch-results.md`. The FP16 case and 54 other block tests pass.
The same single test fails identically when re-run alone.

## Caveats

- These are single-box results with another tenant on GPU 1. Serving runs are
  repeated (two passes) to make noise visible.
- Absolute throughput is not a claim about an idle H200; compare like-for-like
  with the earlier H200 runs that also had GPU-1 contention.
