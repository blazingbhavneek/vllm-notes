#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec uv run --no-project /home/seigyo/vllmServer/.venv/bin/vllm serve \
  /home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB \
  --trust-remote-code --language-model-only \
  --served-model-name nemotron-bolt-h200 --port 8001 \
  --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.55 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --kernel-config '{"modelopt_iq2_dense_cache":true,"linear_backend_per_quant":{"nvfp4_w4a16":"marlin"}}' \
  "$@"
