#!/usr/bin/env python
"""Real Bolt 74B-A9B-24GB smoke + throughput driver (single GPU).

Run from the vllm repo root:
  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/bolt_real.py [extra LLM kwargs...]
"""

import argparse
import json
import time

PROMPTS = [
    "Q: What is the capital of France? A:",
    "Q: Sort these numbers in ascending order: 9, 2, 7, 4. A:",
    "Q: Write a Python function that returns the nth Fibonacci number. A:",
    "Q: Why is the sky blue? A:",
    "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h? A:",
    "Q: Balance the chemical equation: H2 + O2 -> H2O. A:",
    "Q: Name the three laws of robotics. A:",
    "Q: Translate 'good morning' to German. A:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.27)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--mode", default=None, help="compilation_mode, e.g. none")
    ap.add_argument("--out", default=None, help="write generated texts to this json")
    ap.add_argument("--task", default="gsm8k-lite", choices=["smoke", "gsm8k-lite"])
    args, extra = ap.parse_known_args()

    from vllm import LLM, SamplingParams

    kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "language_model_only": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_mem_util,
        "enable_chunked_prefill": True,
        "enforce_eager": args.eager,
        "seed": 0,
    }
    if args.mode:
        kwargs["compilation_config"] = {"mode": args.mode}
    # extra argv becomes keyword args: --tensor-parallel-size 2
    for i in range(0, len(extra), 2):
        key, val = extra[i], extra[i + 1]
        key = key.lstrip("-").replace("-", "_")
        kwargs[key] = json.loads(val)

    t0 = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t0
    print(f"STARTUP_SECONDS {load_s:.1f}", flush=True)

    prompts = PROMPTS
    if args.task == "gsm8k-lite":
        prompts = _gsm8k(32)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"GENERATE {len(prompts)} prompts {n_tok} tokens in {dt:.1f}s "
        f"-> {n_tok / dt:.1f} tok/s aggregate",
        flush=True,
    )
    for o in outs[:4]:
        print("---", repr(o.outputs[0].text[:200]), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                [{"prompt": o.prompt, "text": o.outputs[0].text} for o in outs], f, indent=1
            )


def _gsm8k(n: int) -> list[str]:
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/test.jsonl"
    )
    try:
        rows = [json.loads(x) for x in urllib.request.urlopen(url, timeout=30)]
    except Exception as e:  # offline fallback
        print("gsm8k download failed, fallback prompts:", e)
        return PROMPTS
    out = []
    for r in rows[:n]:
        q = r["question"]
        out.append(
            f"Q: {q}\nA: Let's think step by step.\n"
        )
    return out


if __name__ == "__main__":
    main()
