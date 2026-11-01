# Round 7: SM75 + FP16, GDDR-aware cutoffs (5060 Ti, 2026-11-01)

Commit `f5e6e1bb0b` on local `feature/modelopt-block-triton`. History is kept
as is, with no squash. A single-commit PR branch will be cut from it later.

## Branch state: rebased, NOT pushed yet

The fork's `main` was synced to upstream, 700 commits ahead. The branch was
rebased onto `e0d00e4b8e` (origin/main). The rebase and this commit are local
only, because the force-push needs the user to run it:

```bash
git -C vllm push --force-with-lease origin feature/modelopt-block-triton
```

| Before rebase (pushed) | After rebase (local) | What |
|---|---|---|
| draft | `2cdaee0717` | opencode draft (A/B baseline) |
| rewrite | `c6873ddd01` | rewrite |
| | `c0982f3464`, `9373559772` | rounds 3-6 |
| `f1ec2fb051` | (in the above) | H200 diff |
| `d71cc09daf` | `6fdfb2fa46` | warmup ReLU2 fix |
| — | `f5e6e1bb0b` | this round |

The local branch `backup/modelopt-block-triton-pre-rebase` = `d71cc09daf`.
The `_C` extension is stale after the rebase, because upstream changed 41 csrc
files. Rebuild with `VLLM_USE_PRECOMPILED=1` before serving.

## What changed

- **SM75 gate.** `_use_triton` and `ModelOptBlockTritonExperts` require SM75,
  not SM80. This matches `modelopt_mixed.get_min_capability() == 75`: the
  NVFP4 Mamba projections run on FP4 Marlin, which is SM75+.
- **FP16.** The kernels follow the activation dtype:
  - `tl.dot` casts W to `x.dtype`;
  - the singleton MoE path rounds W to `a`'s dtype;
  - embedding / `dequantize_blocks` take a `dtype`.

  b12x is BF16-only, so `_use_triton(..., dtype)` sends FP16 to Triton on
  every GPU, SM120 included. The dense cache stays BF16-only, and the MoE
  `in_dtype` check now accepts FP16. The linear method takes its dtype from
  `torch.get_default_dtype()`, which is the `params_dtype` that `LinearBase`
  passes, because the loader sets it to the model dtype.
- **Turing tiles.** On sm_75, Triton lowers `tl.dot` to FMA, not tensor cores:
  `probe_mma.py` shows 0 `mma.sync` and 2048 `fma.rn.f32`, versus 32 `mma.sync`
  on sm80. With the default tiles, FMA accumulators spill badly. For example,
  dispatch_report with `DTYPE=float16 CCS=75,...` showed Q8 gemm BM128 spilling
  10.5 KB and MoE BM128 spilling 25 KB. `sweep_sm75.py` found spill-free tiles
  (`results/round7_sm75_gddr/sweep_sm75.txt`):
  - `_linear_config` with `major < 8` and BM ≥ 16 uses BM ≤ 64, BN64, BK32,
    8 warps, 1 stage;
  - `moe_config` without SM80 uses the same.

  GEMV, dequant, moe_gemv and split-K were already spill-free on sm_75.
- **Bandwidth-aware cutoffs.** `dequant_min_m(codec, device)` uses the HBM
  values when the memory bus is ≥ 1024 bits wide, and the GDDR values
  otherwise. `moe_gemv_config` follows the same split.

  | Cutoff | HBM (H200) | GDDR (5060 Ti) | Turing |
  |---|---|---|---|
  | IQ2_XXS dequant+cuBLAS from | 8 | 128 | 32 (estimate) |
  | Q8_0 dequant+cuBLAS from | 64 | 256 | 32 (estimate) |
  | MoE GEMV up to | 8 | 64 | 64 |

  The Turing number is marked `ponytail:` in code: it is a T4 roofline guess,
  because the fused GEMM runs on FMA there while cuBLAS keeps tensor cores.

## Numbers (5060 Ti, `bench_cutoff_check.py`)

The dispatched op against the better of fused or dequant+cuBLAS is mostly
1.00–1.08x of the best, in both BF16 and FP16. The worst rows are:
- Q8 qkv_proj at M=256: 1.24x (fused 258 µs, dequant 319 µs);
- o_proj at 256: 1.11x;
- fc2_latent at 128: 1.13x.

A Q8 cutoff of 512 would fix qkv but cost the fc latents 1.2x at M=256, so it
stays at 256 instead of becoming shape-aware. Under the old H200 cutoffs the
same card was 2.7x off: qkv at M=64 took 200 µs on dequant against 73 µs fused.

## Tests (5060 Ti)

- `test_modelopt_block_triton.py`: 191 passed, 1 skipped. FP16 parametrization
  was added to the linear, MoE and embedding tests.
  - `test_moe_cuda_graph_new_routing` now uses M=65, which stays on `tl.dot`
    on all GPUs.
  - The singleton-graph test forces the `tl.dot` path.
  - The dense cache test uses `dequant_min_m`.
- `test_modelopt.py -k block`: 54 passed, 1 failed.
  - The failure is `test_modelopt_block_triton_dense_cache_matches_uncached`.
    It is a **flake that predates this round**: on HEAD `6fdfb2fa46` with the
    changes stashed it went f p f p p p. With the old cutoffs forced via an env
    switch it also flaked.
  - Backend selection gained a dtype column. The tiny Nemotron-H e2e runs in
    bfloat16 and half, and both pass.
- `check_warmup_relu2.py bfloat16|float16`: no JIT at inference, now up to
  M=64.
- pre-commit on the changed files: all hooks pass.

## Not verified

- **No SM75 hardware run.** Turing coverage is cross-compile + ptxas only.
  The kernel test files still skip below SM80, because CI has no T4.
- SM86/SM89 (GDDR6/6X) use the GDDR cutoffs, calibrated on SM120 only.
- The dense-cache parity flake is still open. The likely suspect is
  nondeterminism outside the block kernels (Mamba/attention), since
  cached and uncached differ only on the dequant path, and that path
  produces identical weights.

## Next

1. The user force-pushes the branch.
2. If a T4 is reachable (Colab or Kaggle), run the kernel tests with the SM80
   skip lifted for FP16, and measure the Turing dequant cutoff.
3. Fix or relax the dense-cache parity test before the PR.
4. Remaining PR items from round 6:
   - Hopper Q8 serving A/B and GSM8K;
   - drop `SINGLETON_GEMV`;
   - pr-checklist.
