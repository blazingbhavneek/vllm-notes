# SM90 kernel implementation from handoff doc — session summary

**Session ID:** ses_e9ec1cd4bffe12zAkSeXlcS0HD
**Span:** 10/22/2026, 1:33 AM – 5:29 AM (model: GLM-5.3-Flash Free)
**Summary of:** the opencode transcript of this session (summarized in place).
**Follows:** the first session (`opencode-kernel-1.md`), which built the correctness-first SM90 fallback and wrote the handoff.

## Goal

Implement the proper Hopper kernel that the handoff left open: a fused dequant→GEMM replacement for b12x on SM90. The packed IQ2_XXS and Q8_0 weights are never materialized as dense copies. Then bring up `Nemotron-3.5-Bolt-74B` serving without `--enforce-eager`, and hand the user a working serve command.

## Timeline

### 1. Read the handoff and map the wiring
- Confirmed the graph-safe MoE path needs: `activation == RELU2_NO_MUL`, `not apply_router_weight_on_input`, codec `iq2_xxs`, and `expert_map` None.
- Read the vLLM `AGENTS.md` contribution rules. Key points: check existing PRs before opening one, disclose AI assistance in PR descriptions, and put kernel perf work under `benchmarks/kernels/` rather than `tests/`.

### 2. Triton kernel design
- New file: `vllm/model_executor/layers/quantization/modelopt_block_quant_hopper.py` (~700 lines).
- Packed weights are decoded in registers, tile by tile, right before the tensor-core dot.
- Kernels (Triton 3.8):
  - `_iq2_decode_tile` / `_q8_decode_tile`: in-register decode helpers. Their fp32 math is exact, so results match the GGML oracles bit-exactly before bf16 rounding on store.
  - `_iq2_gemv_kernel` / `_q8_gemv_kernel`: M=1 weight-stationary GEMV, used when `M == 1 and N >= 4096`.
  - `_iq2_gemm_kernel` / `_q8_gemm_kernel`: tiled `tl.dot` GEMM, bf16 in, fp32 accumulate.
  - `_iq2_gemm_split_kernel` / `_q8_gemm_split_kernel` plus `_combine_kernel`: split-K for small-N, long-K shapes. The combine is a fixed-shape fp32 sum with no atomics, so it is CUDA-graph safe.
  - `_decode_blocks_kernel`: standalone decode, optionally with expert gather.
- Dispatch in `block_linear(x, packed, codec)`: GEMV for M=1 and N≥4096, split-K when the plain tile grid would have fewer than 256 programs, plain GEMM otherwise. Static heuristics, no runtime autotune.
- `modelopt_block_quant_sm90.py` now routes `dequantize_linear_weight`, `apply_linear_fallback` (bf16 activations only), `apply_embedding_fallback`, and the MoE decode through the fast kernels.
- Flag `VLLM_SM90_BLOCK_KERNEL`: `1` or unset = on for SM90, `0`/`off`/`false` = old reference path. Read once per process.

### 3. Graph-safe MoE path (`moe_forward`)
- Two packed-dequant GEMM passes driven by `moe_align_block_size`:
  - Pass 1 (up_proj) applies ReLU² in-kernel.
  - Pass 2 (down_proj) multiplies by routed weights in fp32.
- No `torch.unique`, no host syncs, and experts with zero tokens are never scheduled.
- Wired into `apply_moe_fallback`, guarded to `iq2_xxs` + `RELU2_NO_MUL` + no router-on-input + no EP map. Everything else falls back to the old chunk path.
- Validation:
  - Parity vs reference `fused_experts`: relative error ≤ 7e-3, bf16 accumulation order only.
  - Tiny-case diff was one bf16 ulp at very large magnitudes (squared activations), which is acceptable.
  - `VLLM_SM90_BLOCK_KERNEL=0` remains bit-exact (0.0).
  - Manual CUDA-graph capture and replay on new inputs matched eager exactly (`verify_sm90_moe_graphsafe.py`).

### 4. First serve attempt, then the dynamo fix
- Serving without `--enforce-eager` failed in `torch.compile`. Cause: `current_platform.get_device_capability()` is an `lru_cache`, and Dynamo raises `Unsupported` when the gate runs cold inside the compile region (`apply_linear_fallback` → `_hopper()`).
- Fix: both gates now early-return module-level cached booleans. They are warmed eagerly via `prewarm_device_checks()` in the quant-method `__init__` hooks, so they fold to constants in traced code.
- The env var `VLLM_BLOCK_FORMAT_FALLBACK` produced "Unknown vLLM environment variable" warnings. The later commands don't set it; the SM90 fallback is selected automatically.

### 5. Result: CUDA graphs captured
- `FULL_AND_PIECEWISE` capture: PIECEWISE 35/35 and FULL 35/35, no `cudaErrorStreamCaptureUnsupported`.
- Server live on port 8012 (API PID 138153, EngineCore 138766 at the time), `--gpu-memory-utilization 0.27`, `--max-num-seqs 256`.
- End-to-end streaming chat completions work. Reasoning arrives in `reasoning`, the answer in `content`.
- Engine-reported decode: **~16 tok/s**, measured under heavy contention from other GPU tenants (~139/143 GiB busy).
- Lint clean, ruff formatted, 10/10 repo tests pass, all verify scripts pass.
- Handoff updated: `HANDOFF-SM90-IMPLEMENTATION.md` §13.1–§13.7.

### 6. Handing over the serve command (several rounds with the user)
The user asked for a command to run themselves. Their requests changed each round:

1. Stop the server and give the command. Assistant stopped the server (port 8012 freed, the port-8000 gemma server untouched) and gave a `nohup` command.
2. The user rejected `nohup` and asked for tmux with: no `--max-model-len`, the reasoning parser kept, served name `gemma-4-31B`, and auto-selected cudagraph capture sizes.
3. The user asked for port 8080 and host `0.0.0.0`.
4. The user asked for a plain command with no tmux wrapper.

Launch failure on the user's run:
- The engine failed with mamba-cache sizing. With the default `max_model_len` of 262144 and util 0.27, the mamba cache has only 313 blocks, but auto `max_num_seqs` defaults to 1024.
- Proposed fix: add back `--max-num-seqs 256` (≤ 313). Keeps auto capture sizes (1..256). Use `--gpu-memory-utilization 0.28` if 512 seqs are needed.
- **Status at end of transcript:** the fix was proposed. A successful run was not confirmed.

## Final command (as last proposed)

```
CUDA_VISIBLE_DEVICES=0 /home/seigyo/vllmServer/.venv/bin/vllm serve /home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --host 0.0.0.0 --port 8080 --served-model-name gemma-4-31B --trust-remote-code --max-num-seqs 256 --gpu-memory-utilization 0.27 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser nemotron_v3 --disable-log-requests
```

Note: the served name `gemma-4-31B` is the same name the protected port-8000 server uses.

## Files touched

- New: `vllm/model_executor/layers/quantization/modelopt_block_quant_hopper.py` (~700 lines).
- Changed: `vllm/model_executor/layers/quantization/modelopt_block_quant_sm90.py` (routing through the fast kernels, `prewarm_device_checks`).
- Possibly changed: `modelopt_block_quant.py` (the transcript says the prewarm hooks run in the quant-method `__init__`s, but doesn't list the exact edit; check the handoff §13.
- Notes (outside the repo): `HANDOFF-SM90-IMPLEMENTATION.md` §13, and `scripts/verify_sm90_moe_graphsafe.py`.

## Open at end of session

- Confirm the final serve command starts on the user's machine with `--max-num-seqs 256`.
- Throughput under contention is ~16 tok/s. Kernel-level tuning was not pursued (static heuristics, no autotune).
- Upstream PR process (per `AGENTS.md`) not started: no PR opened, no `benchmarks/kernels/` numbers.
