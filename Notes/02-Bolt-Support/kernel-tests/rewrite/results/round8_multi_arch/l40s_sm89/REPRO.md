# FP16 accuracy repro

The original run at f5e6e1bb0b produced 3/250 with 250 truncations. The retry at ff9dd45e8b produced 234/250 with 1 truncation. See the corresponding `serve_fp16.log`, `gsm8k_fp16.log`, and `gsm8k_fp16.json` files. This was an output correctness failure, not an exception, so there is no failure traceback.

With the optional incompatible torchaudio package hidden from imports, the test used:

```sh
export MODEL=/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f
export TMPDIR=/tmp/b
vllm serve "$MODEL" --trust-remote-code --language-model-only --served-model-name nemotron-bolt-h200 --port 8001 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90 --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}' --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"marlin"}}' --dtype half
# Once ready, in another shell:
python /workspace/vllm-notes/Notes/02-Bolt-Support/kernel-tests/round6_http_accuracy.py --base-url http://127.0.0.1:8001 --count 250 --concurrency 32 --out /tmp/bolt-fp16-repro.json
```
