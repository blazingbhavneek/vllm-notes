"""Reproducible real-model A/B: decode repeats and GSM8K greedy outputs."""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--expand", action="store_true")
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--dense-cache", action="store_true")
ap.add_argument("--linear-backend", default=None)
ap.add_argument("--gemv-max-tokens", type=int, default=None)
ap.add_argument("--batches", default="1,8,32")
ap.add_argument("--diverse-bench", action="store_true")
ap.add_argument("--out", required=True)
ap.add_argument("--eval-count", type=int, default=64)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
args = ap.parse_args()
cache = Path("/tmp/bolt-gsm8k-test.jsonl")
if not cache.exists():
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        timeout=30,
    ) as r:
        cache.write_bytes(r.read())
all_rows = [json.loads(s) for s in cache.read_text().splitlines()]
rows = all_rows[: args.eval_count]
extra = {}
if args.gemv_max_tokens is not None:
    extra["worker_cls"] = "bolt_ablation_worker.GemvGateWorker"
    extra["additional_config"] = {"gemv_max_tokens": args.gemv_max_tokens}
if args.baseline:
    extra["worker_cls"] = "bolt_ablation_worker.BaselineWorker"
if args.dense_cache:
    extra.setdefault("kernel_config", {})["modelopt_iq2_dense_cache"] = True
if args.linear_backend:
    extra.setdefault("kernel_config", {})["linear_backend_per_quant"] = {
        "nvfp4_w4a16": args.linear_backend
    }
if args.expand:
    extra.setdefault("kernel_config", {})["modelopt_iq2_expand"] = True
t0 = time.perf_counter()
llm = LLM(
    model="/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
    trust_remote_code=True,
    language_model_only=True,
    max_model_len=4096,
    max_num_seqs=32,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=args.gpu_memory_utilization,
    enable_chunked_prefill=True,
    seed=0,
    compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
    **extra,
)
result = {
    "expanded": args.expand,
    "gemv_max_tokens": args.gemv_max_tokens,
    "baseline": args.baseline,
    "linear_backend": args.linear_backend,
    "dense_cache": args.dense_cache,
    "startup_seconds": time.perf_counter() - t0,
    "decode": {},
}
print("STARTUP", result["startup_seconds"], flush=True)
sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
prompt = "Q: A shop sells 12 pencils for $3. How much do 7 pencils cost? A: Let's think step by step.\n"
llm.generate([prompt], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
result["decode_diverse"] = {}
for diverse in [False, True] if args.diverse_bench else [False]:
    for batch in map(int, args.batches.split(",")):
        samples = []
        for repeat in range(args.repeats):
            bench_prompts = (
                [
                    f"Q: {all_rows[(repeat * batch + i) % len(all_rows)]['question']}\nA: Let's think step by step.\n"
                    for i in range(batch)
                ]
                if diverse
                else [prompt] * batch
            )
            t0 = time.perf_counter()
            out = llm.generate(bench_prompts, sp, use_tqdm=False)
            elapsed = time.perf_counter() - t0
            tokens = sum(len(o.outputs[0].token_ids) for o in out)
            samples.append(
                {"seconds": elapsed, "tokens": tokens, "tok_s": tokens / elapsed}
            )
            print(
                "DECODE_DIVERSE" if diverse else "DECODE",
                batch,
                repeat,
                samples[-1],
                flush=True,
            )
        result["decode_diverse" if diverse else "decode"][batch] = samples
params = SamplingParams(temperature=0, max_tokens=768, logprobs=5)
prompts = [f"Q: {r['question']}\nA: Let's think step by step.\n" for r in rows]
t0 = time.perf_counter()
outs = llm.generate(prompts, params, use_tqdm=True)
result["eval_seconds"] = time.perf_counter() - t0
result["eval"] = [
    {
        "question": r["question"],
        "answer": r["answer"],
        "text": o.outputs[0].text,
        "token_ids": o.outputs[0].token_ids,
        "finish_reason": o.outputs[0].finish_reason,
        "logprobs": [
            {str(k): v.logprob for k, v in step.items()}
            for step in o.outputs[0].logprobs
        ],
    }
    for r, o in zip(rows, outs)
]
Path(args.out).write_text(json.dumps(result, indent=1))
print("WROTE", args.out, flush=True)
