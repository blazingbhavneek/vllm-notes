# Handoff: test the Bolt Triton branch of vLLM on this GPU

You are on a machine with one NVIDIA GPU. Two repos are cloned side by side
in the same directory:

- `vllm/`: the vLLM fork. Its `origin` is `github.com/blazingbhavneek/vllm`.
- `vllm-notes/`: notes and test scripts. This file is `vllm-notes/handoff.md`.

Your job:
1. Switch `vllm` to the branch under test.
2. Install it editable.
3. Run the checks below.
4. Write a report.

You are testing, not developing. Follow the phases in order and don't explore
beyond them.

## 1. What is under test

- **Branch.** `feature/modelopt-block-triton` on `origin`, commit `f5e6e1bb0b`
  or newer.
- **What it adds.** Triton kernels for ModelOpt block-quantized weights, so
  `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` runs on NVIDIA GPUs from SM75
  up.
  - Before this, only SM120/121 could run it, via the external `b12x`
    package (BF16 only).
  - The weights stay packed in memory.
  - The code is in `modelopt_block_quant_triton.py`,
    `fused_moe/experts/modelopt_block_triton_moe.py` and
    `warmup/modelopt_block_triton_warmup.py`.
- **The model** is public and ungated. Its weights are:
  - IQ2_XXS (2-bit, 256 weights / 66 B): MoE experts;
  - Q8_0 (32 / 34 B): attention, latent fc1/fc2, lm_head, embedding;
  - NVFP4 W4A16: Mamba projections, run by FP4 Marlin.
- **Dispatch.**
  - **Dense layers.** A fused Triton GEMV or GEMM is used at small token
    counts M. From a cutoff M upward, it switches to dequantize then cuBLAS.
    The cutoff comes from `dequant_min_m(codec, device)`, which depends on
    whether the GPU counts as HBM (memory bus ≥ 1024 bit) or GDDR:

    | | HBM | GDDR | Turing |
    |---|---|---|---|
    | IQ2_XXS | 8 | 128 | 32 (a guess) |
    | Q8_0 | 64 | 256 | 32 (a guess) |

  - **Routed MoE.** A per-row GEMV is used up to M = 8 (HBM) or 64 (GDDR),
    and `tl.dot` tiles above that.
  - **Turing.** `tl.dot` runs on FMA, not tensor cores, so Turing uses
    smaller tiles.
- **Verified so far.**
  - H200 (SM90, BF16), full model: GSM8K-250 236/250; 132 / 492 / 865 output
    tok/s at concurrency 1 / 8 / 32; no JIT during inference.
  - RTX 5060 Ti (SM120, BF16 + FP16): kernels only.
  - SM75 has only been cross-compiled.
- **More background** is in
  `vllm-notes/Notes/02-Bolt-Support/08-round7-sm75-fp16-gddr.md`.

## 2. Rules

- **No git writes.** Never commit, push, rebase or open PRs. Local edits are
  allowed only where this file says so. Save them with
  `git diff > $R/local.diff` before finishing.
- **Don't tune or rewrite kernels.** Your output is measurements and
  suggested numbers.
- **Correctness bugs.** Save a minimal repro and the traceback. Attempt a fix
  only if it is obvious and takes under 30 minutes, and keep it as a diff.
- **Report exactly what happened.** A failure with its log is a result. Never
  round a failure into a pass.
- **Don't get stuck.** If an environment problem blocks a phase for more than
  20 minutes, write down what failed and go to the next phase.
- **Servers.**
  - Stop them by PID (`kill $(cat $R/serve.pid)`). Never use
    `pkill -f "vllm serve"`: it also matches your own shell's command line
    and kills it.
  - Run one server at a time. Check that `nvidia-smi` shows no process before
    starting the next.
- **Scope.** Run only the tests named here, not upstream-wide suites.

## 3. Setup

```bash
cd <the directory that holds vllm/ and vllm-notes/>
export WORK=$PWD
cd vllm
git fetch origin
git checkout -B feature/modelopt-block-triton origin/feature/modelopt-block-triton
git log -1 --oneline                # f5e6e1bb0b or newer
# Use the active Python env if there is one; otherwise:
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest pytest-asyncio gguf datasets
```

- If `tests/conftest.py` fails to import, install the missing modules one at
  a time and list them in the report.
- If the precompiled wheel doesn't match the driver (vLLM main defaults to
  CUDA 13.0 wheels, which need driver 580+), try
  `VLLM_MAIN_CUDA_VERSION=12.9 ... --torch-backend=cu129`, and note what
  happened.

**The full model needs ≥ 32 GB of GPU memory** (22.4 GiB of weights plus
graphs and cache). If this GPU has that, start the 24 GB download now; it
finishes while phases A–C run:

```bash
export HF_HOME=$WORK/hf HF_XET_HIGH_PERFORMANCE=1
nohup hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB \
  --local-dir $WORK/bolt > $WORK/dl.log 2>&1 &
```

Run this in every shell. `<slug>` is the GPU and arch, e.g. `a100-80g-sm80`
or `t4-sm75`.

```bash
cd $WORK/vllm && source .venv/bin/activate   # or your env
export HF_HOME=$WORK/hf MODEL=$WORK/bolt
export TMPDIR=/tmp/b; mkdir -p $TMPDIR       # keep short: zmq socket paths cap at 107 chars
export N=$WORK/vllm-notes/Notes/02-Bolt-Support/kernel-tests
export R=$WORK/results/<slug>; mkdir -p $R
```

**On SM75 (Turing), read section 9 before going on.**

## 4. Phase A: fingerprint (~2 min)

```bash
python - <<'EOF' | tee $R/env.txt
import subprocess, time, torch, triton
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t
p = torch.cuda.get_device_properties(0)
print(p.name, f"sm{p.major}{p.minor}", p.multi_processor_count, "SMs",
      round(p.total_memory / 2**30, 1), "GiB, bus", p.memory_bus_width,
      "bit, L2", p.L2_cache_size // 2**20, "MiB")
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("class", "HBM" if t._is_hbm(0) else "GDDR",
      "dequant_min_m", {c: t.dequant_min_m(c, 0) for c in ("iq2_xxs", "q8_0")})
a = torch.empty(2**30, dtype=torch.uint8, device="cuda"); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize(); s = time.perf_counter()
for _ in range(20): b.copy_(a)
torch.cuda.synchronize()
print("copy bandwidth", round(2 * 20 / (time.perf_counter() - s)), "GiB/s")
print(subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                      "--format=csv"], capture_output=True, text=True).stdout)
EOF
git log -1 --oneline >> $R/env.txt
```

## 5. Phase B: correctness tests (~20–40 min)

```bash
pytest -q -rs tests/kernels/quantization/test_modelopt_block_triton.py 2>&1 | tee $R/kernel_tests.txt
pytest -q -rs tests/quantization/test_modelopt.py -k block 2>&1 | tee $R/modelopt_tests.txt
for i in 1 2 3; do
  pytest -q tests/quantization/test_modelopt.py -k dense_cache_matches_uncached 2>&1 | tail -1
done | tee $R/dense_cache_flake.txt
python $N/check_warmup_relu2.py bfloat16 2>&1 | tee $R/warmup_bf16.txt
python $N/check_warmup_relu2.py float16 2>&1 | tee $R/warmup_fp16.txt
```

**What to expect** (reference: RTX 5060 Ti):
- **Kernel tests:** 191 passed, 1 skipped. The int64 test skips without
  12 GiB free.
- **`-k block`:** 54 passed, plus the dense-cache test, which flakes. It
  failed in 2 of 6 runs there, and that predates this branch. Record its
  3-run pattern; don't chase it.
- **`check_warmup_relu2.py`:** must report no JIT.
- **Anything else that fails** is a real finding. Save the full output.

## 6. Phase C: cutoff calibration (~15–30 min)

```bash
python $N/bench_cutoff_check.py 2>&1 | tee $R/cutoff_dense.txt
MS=1,8,16,32,64,128 python $N/rewrite/moe_threshold_ab.py 2>&1 | tee $R/cutoff_moe_bf16.txt
MS=1,8,16,32,64,128 DTYPE=float16 python $N/rewrite/moe_threshold_ab.py 2>&1 | tee $R/cutoff_moe_fp16.txt
python benchmarks/kernels/benchmark_modelopt_block_triton.py 2>&1 | tee $R/benchmark.txt
```

**Dense layers (`cutoff_dense.txt`).**
- The header prints the cutoffs in effect.
- Each row gives four columns:
  - `op`: the dispatched op;
  - `fused`: the Triton kernel;
  - `deq`: dequant + cuBLAS;
  - `op/best`: how far dispatch is from the faster of the two.
- The Q8_0 layers are qkv_proj, o_proj, fc1_latent and fc2_latent. The
  IQ2_XXS layers are shared_up and shared_down.
- For each codec, find the smallest M from which `deq` beats `fused` on most
  of its layers.
- If that differs from the current cutoff, or `op/best > 1.10` on two or more
  rows, rerun with finer steps around it and report a suggested cutoff:
  `MS=16,24,32,48,64,96,128,192,256,384 python $N/bench_cutoff_check.py`.
- Don't edit the code.

**MoE (`cutoff_moe_*.txt`).**
- `speedup > 1` means the per-row GEMV beats `tl.dot` at that M. The header
  prints the current "gemv up to M".
- The default `E=512` is Bolt's real expert count. On OOM, rerun with
  `E=256`, then `E=128`, and record which E worked. Fewer experts shift the
  crossover lower.
- Report the largest M where the GEMV still wins.
- On H200 this microbenchmark disagreed with serving, so treat it as a hint.
  Phase E decides.

`benchmark.txt` is a per-layer reference table. Just keep it.

## 7. Phase D: full model (≥ 32 GB GPUs only, ~45–90 min)

GSM8K test set, once:

```bash
python - <<'EOF'
import json
from datasets import load_dataset
with open("/tmp/bolt-gsm8k-test.jsonl", "w") as f:
    for r in load_dataset("openai/gsm8k", "main", split="test"):
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
EOF
```

Helpers. The accuracy script hardcodes the served name `nemotron-bolt-h200`,
so keep it.

```bash
serve() {  # serve <tag> [extra vllm args...]
  tag=$1; shift
  vllm serve $MODEL --trust-remote-code --language-model-only \
    --served-model-name nemotron-bolt-h200 --port 8001 \
    --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.90 \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
    --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"marlin"}}' \
    "$@" > $R/serve_$tag.log 2>&1 &
  echo $! > $R/serve.pid
  until curl -sf localhost:8001/v1/models >/dev/null; do
    sleep 10; kill -0 $(cat $R/serve.pid) || { echo "server died"; return 1; }
  done
}
measure() {  # measure <tag> <concurrency...>
  tag=$1; shift
  python $N/round6_http_accuracy.py --base-url http://127.0.0.1:8001 --count 250 \
    --concurrency 32 --out $R/gsm8k_$tag.json | tail -1
  for c in "$@"; do
    vllm bench serve --host 127.0.0.1 --port 8001 --model nemotron-bolt-h200 \
      --tokenizer $MODEL --trust-remote-code --backend openai --dataset-name random \
      --random-input-len 128 --random-output-len 256 --num-prompts $([ $c = 1 ] && echo 12 || echo 32) \
      --max-concurrency $c --request-rate inf --ignore-eos --temperature 0 --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir $R --result-filename serve_${tag}_c$c.json
  done
  grep -E "Using (Triton|b12x)|Model loading took|Graph capturing finished|during inference:" $R/serve_$tag.log
  kill $(cat $R/serve.pid); sleep 15
}
```

Runs (for SM75, see section 9):

```bash
# SM120/121 only, before the first serve: makes the default BF16 run use b12x.
uv pip install b12x==1.5.1

serve bf16 && measure bf16 1 8 32
serve fp16 --dtype half && measure fp16 1 32
# SM120/121 only: also force Triton in BF16.
serve bf16_triton --linear-backend triton --moe-backend triton && measure bf16_triton 1 8 32
```

**If the server won't start:**
- If it rejects `linear_backend_per_quant`, drop that `--kernel-config` and
  record which NVFP4 backend the log shows.
- If a 32 GB GPU OOMs at startup, add `--max-num-seqs 16` and note it.

**What to check per run:**
- **Backend lines.** Look for `Using Triton kernels for ModelOpt ...` for
  each codec and layer kind. On an SM120 auto run, look for `b12x` lines
  instead.
- **Startup.** Record "Model loading took X GiB" and the startup time.
- **GSM8K-250.** H200 scored 236/250 with zero truncations.
  - A score below 225, or any NaN, empty or looping text, is a finding.
  - Quote a few answers from the JSON in the report.
- **Throughput.** Record output tok/s and median TPOT from each
  `serve_*_c*.json`. For scale, H200 managed 132 / 492 / 865 tok/s at
  c = 1 / 8 / 32.
- **JIT.** Lines containing `during inference:` must be zero for this
  branch's kernels: `_gemv_kernel`, `_gemm_kernel`, `_moe_*`, `_splitk_*`,
  `_dequant*`, `_embedding*`. If other kernels JIT, record their names; they
  aren't ours.

## 8. Phase E (optional, only with time left): MoE cutoff in serving

This needs phase D done. Edit the MoE cutoff locally. It is in
`moe_gemv_config` in
`vllm/model_executor/layers/quantization/modelopt_block_quant_triton.py`:

```python
max_rows = 8 if _is_hbm(...) else 64
```

Set `max_rows` to the other class's value: 64 on HBM, 8 on GDDR. Then:

```bash
serve bf16_moe_alt && measure bf16_moe_alt 8 16 32
git diff > $R/moe_alt.diff && git checkout -- vllm/model_executor/layers/quantization/modelopt_block_quant_triton.py
```

Compare TPOT at c = 8 / 16 / 32 against the `bf16` run.

## 9. SM75 (Turing) specifics

Turing has no BF16, so vLLM runs it in FP16. The test gates still say SM80.

1. **Lift the gates locally.** Each file has exactly one:
   ```bash
   sed -i 's/has_device_capability(80)/has_device_capability(75)/' \
     tests/kernels/quantization/test_modelopt_block_triton.py tests/quantization/test_modelopt.py
   ```
2. **Phase B as written.**
   - BF16 cases (`dtype0`, `bfloat16`, and unparametrized tests that build
     BF16 inputs) may fail. BF16 isn't supported on Turing, so list those
     failures but don't count them.
   - Every FP16 case (`dtype1`, `half`) must pass.
   - Run only `check_warmup_relu2.py float16`.
3. **Phase C in FP16 with fine steps.** This is the main result on Turing:
   ```bash
   DTYPES=float16 MS=8,16,24,32,48,64,96,128,256 python $N/bench_cutoff_check.py 2>&1 | tee $R/cutoff_dense.txt
   MS=1,8,16,32,64,128 DTYPE=float16 E=128 python $N/rewrite/moe_threshold_ab.py 2>&1 | tee $R/cutoff_moe_fp16.txt
   ```
   - The current Turing cutoff of 32 for both codecs is a guess. Report the
     measured crossover for each codec.
   - With ≥ 48 GB, use `E=512`.
   - Skip `benchmark_modelopt_block_triton.py` if it fails on BF16.
4. **Phase D, only with ≥ 32 GB.** Run
   `serve fp16 --dtype half && measure fp16 1 8 32`.
   - Nemotron was trained in BF16 and may overflow in FP16. NaN, garbage or a
     GSM8K collapse is a finding.
   - If vLLM itself can't start on SM75 (no attention backend, Mamba kernel
     errors), save the traceback and stop.

## 10. Report

Write `$R/SUMMARY.md`:

```markdown
# <GPU name> (SM<cc>) — <date>
- driver ...; torch ...; triton ...; vllm commit ...
- SMs ..., VRAM ..., bus ... bit, L2 ... MiB, copy bandwidth ... GiB/s, class HBM|GDDR
- cutoffs in effect: iq2_xxs ..., q8_0 ..., MoE GEMV up to ...

## Correctness
| Check | Result |
|---|---|
| kernel tests | N passed / N failed / N skipped |
| test_modelopt -k block | ... |
| dense-cache test x3 | p/f pattern |
| warmup JIT bf16 / fp16 | clean / names |

## Cutoffs
- Dense: worst op/best rows; measured crossover per codec; suggested cutoff (or "current is fine")
- MoE: largest M where GEMV wins (E used), bf16 and fp16

## Full model
| run | GSM8K-250 | trunc | c=1 tok/s | c=8 | c=32 | median TPOT c=1 | load GiB | startup s | JIT |
|---|---|---|---|---|---|---|---|---|---|

## Problems
- what failed, first ~30 lines of the traceback, what you tried

## Local diffs
- local.diff / moe_alt.diff: what they change
```

Then:

```bash
git diff > $R/local.diff
tar czf $WORK/bolt-$(basename $R).tgz -C $(dirname $R) $(basename $R)
```

End with the tarball path and the full `SUMMARY.md` in your final message.
