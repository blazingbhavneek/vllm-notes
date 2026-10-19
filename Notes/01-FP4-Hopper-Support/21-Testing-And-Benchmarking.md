# 21 · Testing and benchmarking

**What you'll learn:** every command needed to check correctness, accuracy, speed and memory for this project: unit tests, GSM8K, KL divergence, throughput benchmarks, profilers, memory checks, and how to reproduce every measurement in these notes from `scripts/`.

Run all vLLM commands from the vLLM checkout with its `.venv` active (note 20, Part 2).

---

## 1. Unit tests

```bash
# CSF decoders (after #60646 / T3)
.venv/bin/python -m pytest tests/model_executor/model_loader/test_csf_scales.py -v
# Loader file discovery
.venv/bin/python -m pytest tests/model_executor/model_loader/test_filter_duplicate_safetensors.py -v
# NVFP4 kernel selection and MoE kernels (run on the H200)
.venv/bin/python -m pytest tests/kernels/quantization/test_nvfp4_kernel_selection.py -v
.venv/bin/python -m pytest tests/kernels/moe/test_nvfp4_moe.py -v
.venv/bin/python -m pytest tests/kernels/quantization/test_marlin_gemm.py -v -k fp4
```

Rule of thumb: a loader or decoder change must pass on any machine (CPU tests). A kernel change must pass on sm_90 **and** at least one other architecture (your RTX 5060 Ti, sm_120, works for small shapes).

## 2. Accuracy: GSM8K

### vLLM's own GSM8K script (fast, used in CI)

```bash
# terminal 1
vllm serve nvidia/Qwen3.8-Flash-Next-FP4-Local --max-model-len 32768 \
    --kv-cache-dtype fp8 --reasoning-parser qwen3 --port 8000
# terminal 2 (chat mode, thinking off, for a quick, stable score)
.venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --port 8000 \
    --use-chat-completions --max-tokens 1024 \
    --chat-template-kwargs '{"enable_thinking": false}' --save-results
```

GLM: add `-tp 2 --reasoning-parser glm45`.

### The Qwen4Exp suite (GSM8K + AIME25 with EvalScope, what CI runs on H200)

`tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8.yaml` already runs on H200 (TP 4, GSM8K threshold 0.98, AIME25 0.90). Copy it for FP4-Local (task T7):

```bash
pytest -s -v tests/evals/qwen4_exp/test_accuracy.py \
    --config-list-file=configs/models-h200.txt
```

### Humming's GSM8K configs (good small Hopper smoke tests)

`tests/evals/gsm8k/configs/humming/` has configs such as `Qwen3-30B-A3B-NVFP4-humming.yaml` and `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4-humming.yaml`, plus H100 shard lists. Run them first on the H200: if a small NVFP4 MoE fails with Humming on sm_90, the big ones will too.

```bash
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/humming/config-h100-shard-0.txt
```

### How to judge a score

- Run each configuration **twice** (different seeds, if sampling) to see the noise.
- GSM8K has 1,319 questions: the standard error at ~95 % accuracy is about 0.6 points. A drop of more than about 1.5 points is a red flag.
- Compare FP4-Local against: the vanilla NVFP4 checkpoint on the same H200 (Phase 0), and the other backend (Humming vs Marlin).

## 3. Accuracy: KL divergence (more sensitive than GSM8K)

`scripts/kld_topk.py` (compare part self-tested; the dump part needs vLLM and a GPU and hasn't been run yet ⚠️):

```bash
# Prompts: plain text, paragraphs separated by blank lines (for example the
# wikitext-2 test split, about 200 paragraphs; same file for every run).
.venv/bin/python kld_topk.py dump nvidia/Qwen3.8-Flash-Next-NVFP4 prompts.txt ref.jsonl
.venv/bin/python kld_topk.py dump nvidia/Qwen3.8-Flash-Next-FP4-Local prompts.txt test.jsonl
.venv/bin/python kld_topk.py compare ref.jsonl test.jsonl
# GLM: add  --llm-kwargs '{"tensor_parallel_size": 2}'
```

It prints:

- `kld_nats_per_token`: mean KL(ref ‖ test) over the reference's top 20 tokens plus a "rest" bucket. 0 = identical.
- `top1_agreement`: how often both runs pick the same most likely token.
- `nll_ref`, `nll_test`: average negative log-likelihood of the real next token (lower is better; it's log-perplexity).

What to expect:

| Comparison | Expected KLD |
|---|---|
| Same checkpoint, Humming vs Marlin (W4A16 both) | ≈ 0 (≤ 0.001; only accumulation-order noise) |
| Phase A vs Phase C (bit-identical scales) | ≈ 0, same as above |
| FP4-Local vs vanilla NVFP4 | clearly > 0 (different weights) |
| W4A16 vs W4A8 (Phase D) | small; compare with what an FP8 KV-cache change gives on the same prompts |

For a reference point, LIL measured BF16 vs NVFP4 GLM at about 0.16 nats/token with natural routing, and 0.067 with the BF16 routes forced (note 10).

## 4. Speed: `vllm bench serve`

```bash
# decode-heavy: 1 user, then 32 users
vllm bench serve --backend vllm --model nvidia/Qwen3.8-Flash-Next-FP4-Local \
    --dataset-name random --random-input-len 2048 --random-output-len 512 \
    --num-prompts 64 --max-concurrency 1
vllm bench serve ... --num-prompts 256 --max-concurrency 32
# prefill-heavy (for Phase D): long inputs, short outputs
vllm bench serve ... --random-input-len 16384 --random-output-len 16 \
    --num-prompts 64 --max-concurrency 8
```

Record output tokens/s, mean and P99 inter-token latency (ITL) and time to first token (TTFT). Always write down: vLLM commit, GPU, all server flags, and the backend log lines.

Kernel-level benchmarks go in `benchmarks/kernels/` (see `benchmark_moe.py` and `benchmark_marlin.py` for the style).

## 5. Profiling

```bash
# PyTorch profiler, through the server
vllm serve <model> --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
vllm bench serve ... --profile --num-prompts 4

# Nsight Systems
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end repeat \
    vllm serve <model> --profiler-config.profiler cuda
vllm bench serve ... --profile --num-prompts 2
nsys stats report1.nsys-rep           # per-kernel time table
```

Look for: which MoE kernel runs (names contain `humming` or `marlin`); how much time the MoE layers take vs attention; in Phase C, the CSF decode kernel's share.

## 6. Memory sanitizer (for any new GPU kernel, such as T9's decoder)

```bash
compute-sanitizer --tool memcheck .venv/bin/python -m pytest \
    tests/kernels/quantization/test_csf_decode.py -x
compute-sanitizer --tool racecheck ...   # shared-memory races
```

## 7. Memory checks

- The startup log prints the model weight memory ("Model loading took … GiB") and "GPU KV cache size: … tokens". Compare with note 19.
- In a test: `torch.cuda.reset_peak_memory_stats()` … `torch.cuda.max_memory_allocated()`.
- `nvidia-smi --query-gpu=memory.used --format=csv -l 1` while the server runs.

## 8. Reproduce the checks in these notes

Everything below runs on a laptop (no GPU), downloads only metadata and a few single tensors (about 6 MB in total), and needs Python 3 with numpy (`uv venv .venv && uv pip install numpy`).

```bash
cd Notes/01-FP4-Hopper-Support/scripts
D=/tmp/fp4data && mkdir -p $D        # a fresh, empty folder for downloaded data

# 1) FP4 x E4M3 exactness (no download)
python fp4_scale_exactness.py

# 2) Qwen slab CSF decode + byte-exact re-encode
Q=nvidia/Qwen3.8-Flash-Next-FP4-Local
P=model.language_model.layers.0.mlp.experts.0.down_proj
S=tensors/model-00001-of-00021.safetensors
python st_tensor.py $Q main $S $P.weight $D/qloc_down.weight.bin
python st_tensor.py $Q main $S $P.weight_scale.nvfp4_csf_fixed $D/qloc_down.fixed.bin
python st_tensor.py $Q main $S $P.weight_scale.nvfp4_csf_exceptions $D/qloc_down.exc.bin
python st_tensor.py $Q main $S $P.weight_scale_2 $D/qloc_down.ws2.bin
V=nvidia/Qwen3.8-Flash-Next-NVFP4; VS=model-00001-of-00010.safetensors
python st_tensor.py $V main $VS $P.weight $D/qptq_down.weight.bin
python st_tensor.py $V main $VS $P.weight_scale $D/qptq_down.scale.bin
python st_tensor.py $V main $VS $P.weight_scale_2 $D/qptq_down.ws2.bin
python csf_slab_decode.py $D

# 3) W4A8 error study (+ BF16 comparison: first expert of the fused BF16 tensor)
python st_slice.py Qwen/Qwen3.8-Flash-Next model-00003-of-00131.safetensors \
    model.language_model.layers.0.mlp.experts.down_proj $((2560*640*2)) $D/bf16_down_e0.bin
python w4a8_error_study.py $D

# 4) GLM csf_32x2_v1 decode vs LIL's uncompressed scale, plus the vectorized decoder
G=nvidia/GLM-5.3-Flash-FP4-Local
GP=model.language_model.layers.3.mlp.experts.0.gate_proj
GS=model-00006-of-00040.safetensors
for s in bases codes page_bases offsets exceptions; do
  python st_tensor.py $G main $GS $GP.weight_scale.csf32_$s $D/glm_gate.csf32_$s.bin
done
python st_tensor.py local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD qad2-kda-w8a16 \
    $GS $GP.weight_scale $D/glm_gate.raw_scale.bin
python csf32_decode.py $D
python csf32_vectorized.py $D

# 5) KLD helper self-test (no download)
python kld_topk.py selftest
```

Expected output (19 Oct 2026):

| Script | Key lines |
|---|---|
| `fp4_scale_exactness.py` | 41 of 56 exact (73.2 %); worst 4.76 %; RMS 1.75 % |
| `csf_slab_decode.py` | 226 exceptions (0.221 %); CSF 53.4 % of raw; re-encoding reproduces NVIDIA's streams: True True |
| `w4a8_error_study.py` | per-row exponent: weight 2.11 %, 22.3 % inexact, GEMM 2.13 %; FP8 act only 2.16 %; both 3.04 %; QAD vs BF16 9.56 %, PTQ 8.13 % |
| `csf32_decode.py` | `decoded == uncompressed QAD scale: True`; CSF 54.1 % |
| `csf32_vectorized.py` | all round trips OK; vectorized decode == uncompressed: True; test encoder reproduces NVIDIA's streams: all True |
| `kld_topk.py selftest` | `selftest OK` |

Note: the LIL file is under the LIL license (no re-uploads). Downloading it for a local check is fine; never commit it anywhere.

---

**Next:** [22 · Glossary](22-Glossary.md)
