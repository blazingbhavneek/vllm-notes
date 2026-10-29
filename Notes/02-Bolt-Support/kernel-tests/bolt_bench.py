#!/usr/bin/env python
"""Real Bolt decode benchmark (+ optional torch profiler trace) on one GPU.

  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/bolt_bench.py
"""

import argparse
import json
import os
import time

MODEL = "/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"


def decode_batch(llm, n: int, max_tokens: int, prompt: str):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.time()
    outs = llm.generate([prompt] * n, sp, use_tqdm=False)
    dt = time.time() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    return toks, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.27)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--profile-dir", default=None)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument(
        "--no-gemv",
        action="store_true",
        help="force the padded tl.dot MoE path (needs VLLM_ENABLE_V1_MULTIPROCESSING=0)",
    )
    ap.add_argument(
        "--gemv-k",
        type=float,
        default=None,
        help="override the MoE GEMV gate to M*top_k <= K*E (needs in-process engine)",
    )
    ap.add_argument(
        "--linear-backend",
        default=None,
        help="backend for the NVFP4 Mamba linears, e.g. marlin, b12x, cutlass",
    )
    args = ap.parse_args()

    from vllm import LLM

    if args.no_gemv or args.gemv_k is not None:
        from vllm.model_executor.layers.fused_moe.experts import (
            modelopt_block_triton_moe as moe_mod,
        )
        from vllm.model_executor.layers.quantization import (
            modelopt_block_quant_triton as kernels,
        )

        if args.gemv_k is None:
            gate = lambda *a, **k: None
        else:
            factor = args.gemv_k
            base = kernels.moe_gemv_config
            gate = lambda M, E, tk, codec: (
                base(M, E, tk, codec) if M * tk <= factor * E else None
            )
        kernels.moe_gemv_config = gate
        moe_mod.moe_gemv_config = gate
        print(f"MoE GEMV gate overridden: K={args.gemv_k}, None means off", flush=True)

    kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_chunked_prefill=True,
        enforce_eager=args.eager,
        seed=0,
    )
    if args.linear_backend:
        # Per-scheme only: a global --linear-backend trips the IQ2_XXS/Q8_0 guard,
        # which accepts auto/triton/b12x. The Mamba layers are the NVFP4 ones.
        kwargs["kernel_config"] = {
            "linear_backend_per_quant": {"nvfp4_w4a16": args.linear_backend}
        }
    if args.profile_dir:
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": args.profile_dir,
        }
    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"STARTUP_SECONDS {time.time() - t0:.1f}", flush=True)

    prompt = (
        "Q: A shop sells 12 pencils for $3. How much do 7 pencils cost? "
        "A: Let's think step by step.\n"
    )
    # warm the sampler / graphs
    decode_batch(llm, 1, 16, prompt)

    for n in [int(x) for x in args.batches.split(",")]:
        toks, dt = decode_batch(llm, n, 128, prompt)
        print(
            f"DECODE b={n:3d} {toks:6d} tok {dt:6.2f}s -> {toks / dt:7.1f} tok/s "
            f"({dt / toks * 1000:6.2f} ms/token/seq)",
            flush=True,
        )

    if args.profile_dir:
        os.makedirs(args.profile_dir, exist_ok=True)
        llm.start_profile()
        decode_batch(llm, 1, 64, prompt)
        decode_batch(llm, 8, 64, prompt)
        llm.stop_profile()
        time.sleep(3)
        print("PROFILE written to", args.profile_dir, flush=True)


if __name__ == "__main__":
    main()
