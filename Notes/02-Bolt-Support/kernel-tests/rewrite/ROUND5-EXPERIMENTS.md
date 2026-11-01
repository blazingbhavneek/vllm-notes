# GPU 1 experiment reproducibility

Run scripts from `/home/seigyo/vllmServer/vllm` with
`CUDA_VISIBLE_DEVICES=1 uv run --no-project /home/seigyo/vllmServer/.venv/bin/python`.
All experiments use the existing venv. No clock or power settings were changed.

- `bolt_saturation.py` (one directory above): real-model benchmark and GSM8K.
  `--baseline` uses copies of HEAD `779b13849a`'s expert implementation and
  disables all new MoE dispatch; `--dense-cache` enables the production cache.
  The baseline's malformed warmup K/output dimensions are fixed in both arms;
  forward math is retained. Use `--linear-backend marlin` to pin NVFP4 dispatch.
  The full model has variable expert counts: 12 layers of 240, 6 of 304,
  9 of 368, 10 of 432 and 3 of 496. The 512-expert microbenchmarks are synthetic
  sparse-routing cases, not the model's exact per-layer expert counts.
- `singleton_moe_ab.py`, `singleton_moe_sweep*.py`: adaptive singleton prototype,
  BF16 weight rounding before FP32 reduction, diverse and repeated routes.
  Microbenchmark reference configurations follow their source revision; use the
  pinned full-model baseline worker for final before/after comparisons.
- `packed_register_ab.py`, `packed_moe_ab.py`: Hopper packed BF16x2/PRMT prototype.
  It retains BF16 activations. Dense expert-shaped wins did not transfer well
  to the full MoE, so it is not in production.
- `moe_threshold_ab.py`: demonstrates why an isolated random-routing benchmark
  alone cannot choose the serving cutoff. Raising the global cutoff to 32
  regressed real repeated-request throughput.
- `moe_prefill_warps.py`, `moe_prefill_sweep.py`, `moe_prefill_splitn.py`: attempted
  removal of the pre-existing wide IQ2 prefill register spill. All alternatives
  were slower; the fastest original four-warp configuration is retained.
- `plane3_ab.py`: corrected historical plane benchmark. Row indices now have the
  correct rank; exact group scales are FP32 and real checkpoint tensors feed
  both timed arms. The original 1.7x headline was invalid.
- `expanded_ab.py`: needs `expanded-iq2-experiment.patch` applied to a clean
  worktree at `779b13849a`, not the current working tree. This archived rejected
  implementation adds exact expanded-plane loading and tests. Its valid
  single-token shared-expert timing was slower than packed IQ2.

`dispatch_report.py` now exits nonzero when any configuration exceeds its
shared-memory limit or has an SM90 spill. It intentionally reports the known
baseline wide IQ2 spill instead of silently returning success. See the main
round-5 report for the measured decision to retain that configuration.

`collect_round5.py` (one directory above) archives the `/tmp/bolt-*` evidence,
compressed model outputs with logprobs, paired grades, and the production diff
in `results/round5_h200_gpu1/`.
