# Handoff: test Bolt's Triton path on one rented GPU (round 8, 2026-11-01)

You are an agent on a rented single-GPU box (vast.ai). Your job is to run the
checks below on this GPU and leave one results tarball behind. You are not
here to develop or optimize. The rental is billed by the hour, so follow the
phases in order and don't explore beyond them.

Section 1 is for the human picking GPUs. Agents start at section 2.

---

## 1. Which GPUs to rent (for the human)

The full model needs **≥ 32 GB**: 22.4 GiB of weights plus 4–8 GiB for graphs,
activations and the mamba/KV cache. "Kernel-only" cards run phases A–C (tests
and microbenchmarks) but skip phase D (the full model).

| Arch | Full model (≥ 32 GB) | Kernel-only stand-ins | Class the code sees | What this box answers |
|---|---|---|---|---|
| **SM75** Turing | Quadro RTX 8000 48 GB | T4 16 GB, Titan RTX 24 GB, Quadro RTX 6000 24 GB, RTX 2080 Ti 11 GB | GDDR | First real Turing run (FP16 only). Replaces the guessed Turing cutoff of 32 with a measured one. |
| **SM80** | A100 40/80 GB (SXM or PCIe), A800 | A30 24 GB | HBM | Do the HBM cutoffs (tuned at H200's 4.8 TB/s) hold at 1.5–2 TB/s? |
| **SM86** | RTX A6000 48 GB, A40 48 GB | RTX 3090 / 3090 Ti 24 GB, RTX A5000 24 GB, A10 24 GB, RTX A4000 16 GB | GDDR | GDDR cutoffs on Ampere (small 6 MB L2). |
| **SM89** | L40S 48 GB, L40 48 GB, RTX 6000 Ada 48 GB, RTX 5000 Ada 32 GB (tight) | RTX 4090 24 GB, L4 24 GB, RTX 4500 Ada 24 GB, RTX 4000 Ada 20 GB | GDDR | GDDR cutoffs on Ada (48–96 MB L2). |
| **SM90** | H100 PCIe / SXM / NVL, GH200 | — | HBM | Optional: you already have H200 results. H100 PCIe (2 TB/s) is a cheaper HBM point. |
| **SM100** | B200 180 GB, GB200 | — | HBM | b12x doesn't run here, so Triton is the path that would actually be used. Never run yet. |
| **SM103** | B300 288 GB | — | HBM | Optional. Nothing has been tried on 10.3, not even a cross-compile. |
| **SM120** | RTX 5090 32 GB (tight), RTX PRO 6000 96 GB, RTX PRO 5000 48 GB, RTX PRO 4500 32 GB (tight) | RTX 5080 / 5070 Ti 16 GB (or the home 5060 Ti) | GDDR | Full model on b12x (BF16) vs Triton BF16 vs Triton FP16. |

**Can't test:**
- SM70 V100: below the model's SM75 floor.
- SM87 / SM110 (Jetson) and SM121 (DGX Spark): not on vast.

**Cards that test the cutoff rule itself, not just their arch.** The code
classes a GPU as HBM when `memory_bus_width >= 1024`. These cards break the
assumption that a wide bus means fast memory:
- **A30.** HBM bus, but only 933 GB/s, which is GDDR-class speed.
- **RTX 5090 / RTX PRO 6000.** GDDR7 at 1.8 TB/s, faster than an A100 40 GB
  (HBM, 1.55 TB/s), yet classed GDDR.
- **L4.** 300 GB/s, the slow end of GDDR.

If any of these disagree with its class, the rule should probably switch to
measured bandwidth.

**Cheapest set that covers every arch:**
- T4 (kernel-only)
- A100 40 GB
- RTX 3090 (kernel-only) or A40
- L40S
- RTX 5090
- B200, for one short session

**Box filters on vast:**
- Max CUDA ≥ 13.0 (driver 580+), because vLLM main defaults to cu130 wheels.
- ≥ 100 GB disk.
- Fast download, since the checkpoint is 24 GB.
- One GPU.

---

## 2. Background (for the agent)

- **Model.** `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` is public and
  ungated. It is a ModelOpt mixed-precision checkpoint with three weight types:
  - IQ2_XXS (2-bit blocks, 256 weights / 66 B): the routed and shared MoE
    experts.
  - Q8_0 (32 / 34 B): attention q/k/v/o, the latent fc1/fc2, lm_head and the
    embedding.
  - NVFP4 W4A16: the Mamba in/out projections, run by FP4 Marlin (SM75+).
- **Code.** The fork branch `feature/modelopt-block-triton` at
  `https://github.com/blazingbhavneek/vllm`, commit `f5e6e1bb0b` or newer.
  - On SM120/121 with BF16, the packed weights run on the external `b12x`
    package.
  - Everywhere else from SM75 up (or with FP16, or when forced), they run on
    Triton kernels: `modelopt_block_quant_triton.py`, the experts in
    `fused_moe/experts/modelopt_block_triton_moe.py`, and the warmup in
    `warmup/modelopt_block_triton_warmup.py`. Weights stay packed in memory.
- **What gets dispatched.**
  - **Dense layers.** A fused Triton GEMV or GEMM at small token counts M.
    From a cutoff M upward, it switches to dequantize-to-BF16/FP16 then
    cuBLAS. The cutoff comes from `dequant_min_m(codec, device)`:

    | | HBM | GDDR | Turing |
    |---|---|---|---|
    | IQ2_XXS | 8 | 128 | 32 (guess) |
    | Q8_0 | 64 | 256 | 32 (guess) |

  - **Routed MoE.** A per-row GEMV up to M = 8 (HBM) or 64 (GDDR), and
    grouped `tl.dot` tiles above that (`moe_gemv_config`).
  - **Turing.** `tl.dot` becomes FMA, not tensor cores, so it uses smaller
    tiles.
- **Verified so far.**
  - H200 (SM90, BF16), with the full model: GSM8K-250 236/250, 132 / 492 /
    865 output tok/s at concurrency 1 / 8 / 32, no JIT during inference.
  - RTX 5060 Ti (SM120, BF16 + FP16), kernels only: the model doesn't fit.
  - SM75 has only been cross-compiled; no Turing GPU has run it.
- **The scripts** live in the notes repo
  `https://github.com/blazingbhavneek/vllm-notes` (public) under
  `Notes/02-Bolt-Support/kernel-tests/`. More background is in
  `Notes/02-Bolt-Support/08-round7-sm75-fp16-gddr.md`.

## 3. Rules

- **Never commit, push, rebase or open PRs.** Local edits are allowed only
  where this document says so. Before finishing, save them with
  `git diff > $R/local.diff`.
- **Don't tune or rewrite kernels.** Your output is measurements and
  suggested numbers.
- **If a correctness bug shows up,** save a minimal repro and the traceback.
  Attempt a fix only if it is obvious and takes under 30 minutes, and keep it
  as a diff.
- **Report exactly what happened.** A failure with its log is a useful result.
  Never round a failure into a pass.
- **If an environment problem blocks a phase for more than 20 minutes,** write
  down what failed and go to the next phase.
- **Stop servers by PID** (`kill $(cat $R/serve.pid)`). Never use
  `pkill -f "vllm serve"`: it also matches your own shell's command line and
  kills it.
- **Run one server at a time.** Check `nvidia-smi` shows no process before
  starting the next one.
- **Don't run upstream-wide test suites.** Only the tests named here.

## 4. Setup (phase 0, ~15 min)

```bash
nvidia-smi                          # driver >= 580, i.e. "CUDA Version: 13.x"
df -h /workspace 2>/dev/null || df -h ~
export WORK=/workspace; [ -d $WORK ] || export WORK=$HOME
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env)
command -v gcc tmux git || (apt-get update && apt-get install -y gcc g++ git tmux)

cd $WORK
git clone -b feature/modelopt-block-triton https://github.com/blazingbhavneek/vllm.git
git clone https://github.com/blazingbhavneek/vllm-notes.git
git -C vllm log -1 --oneline        # f5e6e1bb0b or newer
cd vllm
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest pytest-asyncio gguf datasets

# Start the 24 GB download now; it lands while phases A-C run (skip on <32 GB cards).
export HF_HOME=$WORK/hf HF_XET_HIGH_PERFORMANCE=1
tmux new -d -s dl "$WORK/vllm/.venv/bin/hf download \
  nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --local-dir $WORK/bolt 2>&1 | tee $WORK/dl.log"
```

Put these in every new shell. `<slug>` is something like `a100-80g-sm80` or
`t4-sm75`.

```bash
cd $WORK/vllm && source .venv/bin/activate
export HF_HOME=$WORK/hf
export TMPDIR=/tmp/b; mkdir -p $TMPDIR    # must stay short: zmq socket paths cap at 107 chars
export N=$WORK/vllm-notes/Notes/02-Bolt-Support/kernel-tests
export R=$WORK/results/<slug>; mkdir -p $R
export MODEL=$WORK/bolt
```

**Install problems:**
- If `tests/conftest.py` fails to import, install the missing modules one at a
  time and list them in the summary.
- On a host whose driver is CUDA 12.x, try
  `VLLM_MAIN_CUDA_VERSION=12.9 ... --torch-backend=cu129` and note whether it
  worked.

## 5. Phase A: fingerprint (~2 min)

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

## 6. Phase B: correctness (~20–40 min)

**On SM75, do section 9 first.**

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
- Kernel tests: 191 passed, 1 skipped. The int64 test skips without 12 GiB
  free.
- `-k block`: 54 passed, plus the dense-cache test, which flakes. It failed
  in 2 of 6 runs there, and that predates this work. Record its 3-run pattern;
  don't chase it.
- `check_warmup_relu2.py` must report no JIT.
- Any other failure is a real finding. Save the full output.

## 7. Phase C: cutoff calibration (~15–30 min)

```bash
python $N/bench_cutoff_check.py 2>&1 | tee $R/cutoff_dense.txt
MS=1,8,16,32,64,128 python $N/rewrite/moe_threshold_ab.py 2>&1 | tee $R/cutoff_moe_bf16.txt
MS=1,8,16,32,64,128 DTYPE=float16 python $N/rewrite/moe_threshold_ab.py 2>&1 | tee $R/cutoff_moe_fp16.txt
python benchmarks/kernels/benchmark_modelopt_block_triton.py 2>&1 | tee $R/benchmark.txt
```

**Dense layers (`cutoff_dense.txt`).**
- The header prints the cutoffs in effect.
- Each row gives:
  - `op`: the dispatched op;
  - `fused`: the Triton kernel;
  - `deq`: dequant + cuBLAS;
  - `op/best`: how far dispatch is from the faster of the two.
- The Q8_0 layers are qkv_proj, o_proj, fc1_latent and fc2_latent. The
  IQ2_XXS layers are shared_up and shared_down.
- For each codec, find the smallest M from which `deq` beats `fused` on most
  of its layers.
- If that crossover is not the current cutoff, or `op/best > 1.10` on two or
  more rows, rerun with finer steps around it and report a suggested cutoff.
  For example:
  `MS=16,24,32,48,64,96,128,192,256,384 python $N/bench_cutoff_check.py`.
- Don't edit the code.

**MoE (`cutoff_moe_*.txt`).**
- `speedup > 1` means the per-row GEMV beats `tl.dot` at that M. The header
  prints the current "gemv up to M".
- The default `E=512` is Bolt's real expert count. On OOM, rerun with
  `E=256`, then `E=128`, and record which E worked. Fewer experts shift the
  crossover lower: on the 5060 Ti, E=128 crosses below M=64.
- Report the largest M where the GEMV still wins.
- On H200, this microbenchmark disagreed with serving (it said the GEMV still
  won at M=16; serving said no). So treat it as a hint. Phase E decides.

`benchmark.txt` is a per-layer reference table. Just keep it.

## 8. Phase D: full model (≥ 32 GB cards only, ~45–90 min)

Write the GSM8K test set once:

```bash
python - <<'EOF'
import json
from datasets import load_dataset
with open("/tmp/bolt-gsm8k-test.jsonl", "w") as f:
    for r in load_dataset("openai/gsm8k", "main", split="test"):
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
EOF
```

Define the run (the accuracy script hardcodes the served name
`nemotron-bolt-h200`, so keep it):

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

Run (on SM75, see section 9 instead):

```bash
# SM120 only, before the first serve: makes the default BF16 run use b12x.
uv pip install b12x==1.5.1

serve bf16 && measure bf16 1 8 32
serve fp16 --dtype half && measure fp16 1 32
# SM120 only: also force Triton in BF16.
serve bf16_triton --linear-backend triton --moe-backend triton && measure bf16_triton 1 8 32
```

**If the server won't start:**
- If it rejects `linear_backend_per_quant`, drop that `--kernel-config` and
  record which NVFP4 backend the log shows.
- On a 32 GB card that OOMs at startup, add `--max-num-seqs 16` and note it.

**What to check per run:**
- **Backend lines.** `Using Triton kernels for ModelOpt ...` should appear for
  each codec and layer kind, or `b12x` lines on an SM120 auto run.
- **Startup.** Note "Model loading took X GiB" and the total startup time.
- **GSM8K-250.** H200 scored 236/250 with zero truncations.
  - Below 225, or any NaN, empty or looping text, is a finding.
  - Copy a few answers from the JSON into the summary.
- **Throughput.** Output tok/s and median TPOT from each `serve_*_c*.json`.
  For scale, H200 managed 132 / 492 / 865 tok/s at c = 1 / 8 / 32.
- **JIT.** Lines containing `during inference:` must be zero for our kernels
  (`_gemv_kernel`, `_gemm_kernel`, `_moe_*`, `_splitk_*`, `_dequant*`,
  `_embedding*`). JIT in other kernels: record the names, they aren't ours.

## 9. SM75 (Turing) specifics

Turing has no BF16, so vLLM runs it in FP16, and the test gates still say
SM80.

1. **Lift the gates locally.** Each file has exactly one SM80 gate:
   ```bash
   sed -i 's/has_device_capability(80)/has_device_capability(75)/' \
     tests/kernels/quantization/test_modelopt_block_triton.py tests/quantization/test_modelopt.py
   ```
2. **Phase B as written.** BF16 cases (`dtype0`, `bfloat16`, and unparametrized
   tests that build BF16 inputs) may fail. That config isn't supported on
   Turing, so list them but don't count them. Every FP16 case (`dtype1`,
   `half`) must pass. Run only `check_warmup_relu2.py float16`.
3. **Phase C in FP16 with fine steps.** This is the main result of the box:
   ```bash
   DTYPES=float16 MS=8,16,24,32,48,64,96,128,256 python $N/bench_cutoff_check.py 2>&1 | tee $R/cutoff_dense.txt
   MS=1,8,16,32,64,128 DTYPE=float16 E=128 python $N/rewrite/moe_threshold_ab.py 2>&1 | tee $R/cutoff_moe_fp16.txt
   ```
   - The current Turing cutoff is 32 for both codecs. It is a guess, based on
     fused `tl.dot` running on FMA while cuBLAS keeps tensor cores.
   - Report the measured crossover for each codec.
   - On a 48 GB card use `E=512`.
   - Skip `benchmark_modelopt_block_triton.py` if it fails on BF16.
4. **Phase D, only on the RTX 8000.** Run `serve fp16 --dtype half && measure
   fp16 1 8 32`.
   - Nemotron was trained in BF16 and may overflow in FP16. NaN, garbage or a
     GSM8K collapse is a finding either way.
   - If vLLM itself can't start on SM75 (no attention backend, Mamba kernel
     errors), save the traceback and stop. That tells us whether "SM75+" in
     the docs is honest.

## 10. Phase E (optional, only with time left): MoE cutoff in serving

The MoE GEMV cutoff on H200 was decided by serving throughput, not by the
microbenchmark. To check this GPU, you need a full-model box with phase D
done. Edit the cutoff locally, in `moe_gemv_config` in
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

## 11. Deliverable

Write `$R/SUMMARY.md` with this template, then pack everything:

```markdown
# <GPU name> (SM<cc>) — <date>
- vast offer / host: ...; driver ...; torch ...; triton ...; vllm commit ...
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

```bash
git diff > $R/local.diff
tar czf $WORK/bolt-$(basename $R).tgz -C $(dirname $R) $(basename $R)
ls -la $WORK/bolt-*.tgz
```

End with the tarball path and the full `SUMMARY.md` in your final message.
The human will copy the tarball off the box before destroying it.
