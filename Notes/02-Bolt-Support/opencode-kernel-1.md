# Running Nemotron Bolt 74B with vLLM — session summary

**Session ID:** ses_e9f0199dcffe2ME5rZdgalUrtp
**Span:** 10/22/2026, 12:24 AM – 1:33 AM (model: DeepSeek V4.1 Flash Free)
**Summary of:** the opencode transcript of this session (summarized in place).

## Goal

Serve `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (IQ2_XXS MoE, ~22.4 GiB packed, ModelOpt mixed quant) on an H200 NVL (SM90) using the local vLLM fork at `/home/seigyo/vllmServer/vllm` (v0.32.1rc1.dev191+gc2861f7f1, commit `c2861f7f12`). Constraints from the handoff:

- Packed uint8 weights stay byte-identical at rest. Only transient, per-call decode into scratch is allowed.
- TP=1 on GPU 0 (`CUDA_VISIBLE_DEVICES=0`).
- Don't disturb other services (a gemma-4-31B server on port 8000 is protected).

Done means: serves, streams tokens, tool calling and reasoning parsing work. Accuracy checks were deferred.

## Key findings

- **The blocker:** b12x (the IQ2/block-quant kernel stack) is SM120/121-only. Startup failed with `ValueError: Block-quantized dense weights require b12x on SM12x` from `ModelOptBlockQuantLinearMethod.__init__` in `modelopt_block_quant.py` (~line 189).
- **Quant layout** (`dump_quant_map.py`):
  - IQ2_XXS, 16.3 GiB: routed and shared experts.
  - W4A16_NVFP4, 2.3 GiB: mamba `in_proj`/`out_proj`. Already runs via Marlin on SM90, no changes needed.
  - Q8_0, 1.1 GiB: `lm_head`, attention q/k/v/o, latent `fc1`/`fc2`.
- **Model shape:** 88 layers, hidden 4096, MoE latent 1024, topk 9, non-gated ReLU² MoE, `routed_scaling_factor` 5.0, per-layer expert counts vary.
- **Block formats:**
  - IQ2_XXS: 66 bytes per 256 weights (fp16 base, 8 records of 4 grid bytes plus a 32-bit control word).
  - Q8_0: 34 bytes per 32 weights (fp16 scale plus int8 payload).
  - A vectorized PyTorch decoder matched the b12x oracle exactly (max diff 0.0) on random blocks and on real checkpoint tensors.

## Timeline

### 1. Recon and plan
- Read the handoff, confirmed the repo was clean at `c2861f7f12`, surveyed the model dir, config, `hf_quant_config.json` and GPU state.
- Confirmed W4A16_NVFP4 on Marlin should work unmodified.
- Chose an approach: an SM90 fallback that decodes packed blocks to bf16 per call, then uses existing kernels (vLLM Triton `fused_experts` for MoE, `F.linear` for dense). No b12x on SM90.

### 2. SM90 fallback implementation
- New module: `vllm/model_executor/layers/quantization/modelopt_block_quant_sm90.py` (decode helpers plus dense and MoE apply).
- Patched `modelopt_block_quant.py`:
  - SM90 allowed in both quant-method constructors, with an `sm90_fallback` flag.
  - b12x packing and plans skipped on the fallback path.
  - `apply` dispatches to the fallback.
- Dummy-weight serve (`--load-format dummy`) reached "Application startup complete".

### 3. Real weights, first generation
- Real-weight serve came up on port 8011.
- First token took ~19 s. A 32-token request timed out.
- Root cause: the first version decoded **all** experts every token, which is about 0.1 tok/s.
- Fix: decode only the experts actually routed (`torch.unique(topk_ids)`), in chunks. Startup went from 66 s to 15 s. Throughput became ~0.9 tok/s.
- Output was correct: "The capital of France is Paris." Chunked MoE matched full decode with max diff 0.0.

### 4. "Why is it so slow / why eager?"
- Not HF kernels. The bottleneck is PyTorch unpacking IQ2 into bf16 scratch: roughly 13–16 GiB of bf16 materialized per token, about 12–15 GB/s effective.
- `--enforce-eager` was a bring-up choice. CUDA graphs were blocked by `torch.unique` and the Python chunk loop.
- `torch.compile` would hit graph breaks on the dynamic indexing.
- The real fix is a fused dequant→GEMM kernel, which is what b12x provides on SM120.
- The other GPU tenant explains only a fraction of the gap.

### 5. Peripherals wrap-up
- Reasoning parser: `--reasoning-parser nemotron_v3`. In this build, reasoning comes back in `message.reasoning` (not `reasoning_content`).
- Tool parser: `--tool-call-parser qwen3_xml`. The model emits Qwen3-style `<tool_call><function=...>` XML, and no nemotron tool parser is registered.
- Verified: non-streaming reasoning/content split, streaming `delta.reasoning`, and a `get_weather` tool call returning `tool_calls` with `finish_reason` `tool_calls`.
- Ruff (0.14.6 via `uvx`) check and format clean, `git diff --check` clean. **No commit made.**
- Final server left on `127.0.0.1:8012`, eager, parsers enabled (PID 70681).

### 6. Handoff written
- New file: `/home/seigyo/vllm-notes/Notes/02-Bolt-IQ2-Hopper/HANDOFF-SM90-IMPLEMENTATION.md`.
- After a follow-up question, added a self-contained §0.1 "Exact paths" table.
- Updated the status line in the older `HANDOFF.md`.
- Notes scripts: `verify_sm90_block_decode.py`, `verify_sm90_moe_fallback.py`, `bench_sm90_decode.py`, `serve_sm90.sh`, `smoke_sm90.sh`. The smoke wrapper itself was not run end to end; its underlying commands were run manually.

## Files touched

- Changed: `vllm/model_executor/layers/quantization/modelopt_block_quant.py` (+31 / −9).
- New: `vllm/model_executor/layers/quantization/modelopt_block_quant_sm90.py`.
- Notes (outside the repo): the handoff docs and `scripts/` listed above.

## Open at end of session

- The fast Hopper IQ2/Q8 kernel itself (left for the next agent, see the handoff).
- Removing the `torch.unique` / `--enforce-eager` restriction.
- Optional: perplexity spot-check and full tok/s benchmark.

## Errors hit and fixed

1. Dummy serve: b12x SM12x gate (the reason for the whole fallback).
2. Decode prototype bugs: sign LUT indexed 128 instead of 256, a tensor dimension mismatch, and non-finite random fp16 bases. Fixed, and the final decode matched the oracle.
3. Prototype `sed` edits left Python comparison syntax errors. Fixed.
4. Serve launched with `nohup` in the shell was killed when the shell call ended. Fixed by relaunching with `setsid` and redirecting all fds.
