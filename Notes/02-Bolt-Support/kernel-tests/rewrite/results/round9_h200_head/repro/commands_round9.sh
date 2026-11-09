#!/usr/bin/env bash
# Reproduce the round-9 H200 validation for PR #65003 at f533470e67.
# Every GPU command pins CUDA_VISIBLE_DEVICES=1. GPU 0 is shared/untouched.
set -euo pipefail

REPO=/home/seigyo/vllmServer/vllm
VPY=/home/seigyo/vllmServer/.venv/bin/python
VLLM=/home/seigyo/vllmServer/.venv/bin/vllm
OUT=/home/seigyo/vllmServer/vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/results/round9_h200_head
MODEL=/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB

export CUDA_VISIBLE_DEVICES=1
# FlashInfer builds its sampling module with ninja; the venv ships it here.
export PATH="/home/seigyo/vllmServer/.venv/bin:$PATH"
export PYTHONPATH="$REPO"
mkdir -p "$OUT"
cd "$REPO"

# 1. Kernel tests.
"$VPY" -m pytest tests/kernels/quantization/test_modelopt_block_triton.py -q \
  | tee "$OUT/pytest_kernel_tests.log"

# 2. ModelOpt block tests.
"$VPY" -m pytest tests/quantization/test_modelopt.py -q -k block \
  | tee "$OUT/pytest_modelopt_block.log"

# 3. SM90 single-row Q8_0 dispatch A/B.
"$VPY" ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/q8_singleton_ab_head.py \
  --out "$OUT/q8_singleton_ab.json" | tee "$OUT/q8_singleton_ab.log"

# 4. Kernel microbenchmark.
"$VPY" benchmarks/kernels/benchmark_modelopt_block_triton.py \
  > "$OUT/benchmark_h200_head.txt" 2>&1

# 5. Full model server (BF16, compiled, CUDA graphs, Marlin for NVFP4).
nohup "$VLLM" serve \
  "$MODEL" --trust-remote-code --language-model-only \
  --served-model-name nemotron-bolt-h200 --port 8001 \
  --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.55 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --kernel-config '{"modelopt_iq2_dense_cache":true,"linear_backend_per_quant":{"nvfp4_w4a16":"marlin"}}' \
  > "$OUT/serve_launch.log" 2>&1 &
echo $! > "$OUT/server.pid"

# Wait for the endpoint.
for _ in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null; then break; fi
  sleep 5
done
curl -sf http://127.0.0.1:8001/health >/dev/null

# 6. Two alternating serving passes at c=1/8/16/32.
for pass in 1 2; do
  for c in 1 8 16 32; do
    prompts=32; [ "$c" = 1 ] && prompts=12
    "$VLLM" bench serve \
      --host 127.0.0.1 --port 8001 --model nemotron-bolt-h200 \
      --tokenizer "$MODEL" --trust-remote-code --backend openai --dataset-name random \
      --random-input-len 128 --random-output-len 256 --num-prompts "$prompts" \
      --max-concurrency "$c" --request-rate inf --ignore-eos --temperature 0 --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
      --save-result --result-dir "$OUT" --result-filename "serve_c${c}_pass${pass}.json" \
      > "$OUT/serve_c${c}_pass${pass}.log" 2>&1
  done
done

# 7. GSM8K-250 through the live server.
"$VPY" ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/round6_http_accuracy.py \
  --base-url http://127.0.0.1:8001 --count 250 --concurrency 32 \
  --out "$OUT/gsm8k_250.json" | tee "$OUT/gsm8k_250.log"

# 8. Stop the server.
kill "$(cat "$OUT/server.pid")" || true
sleep 5

# 9. Summarize.
"$VPY" ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/summarize_round9.py "$OUT"
