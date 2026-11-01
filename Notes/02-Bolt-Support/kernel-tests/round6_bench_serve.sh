#!/usr/bin/env bash
set -euo pipefail
label=${1:?result label}; concurrency=${2:?concurrency}; prompts=${3:-32}
CUDA_VISIBLE_DEVICES=1 uv run --no-project /home/seigyo/vllmServer/.venv/bin/vllm bench serve \
 --host 127.0.0.1 --port 8001 --model nemotron-bolt-h200 \
 --tokenizer /home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB \
 --trust-remote-code --backend openai --dataset-name random \
 --random-input-len 128 --random-output-len 256 --num-prompts "$prompts" \
 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --temperature 0 --seed 0 \
 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
 --save-result --result-dir /home/seigyo/vllmServer/vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/results/round6_serving_profile \
 --result-filename "$label.json"
