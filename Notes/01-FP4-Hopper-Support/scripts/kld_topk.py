# Usage (see note 21, section 3):
#   dump:    .venv/bin/python kld_topk.py dump  MODEL prompts.txt out.jsonl \
#                [--k 20] [--max-tokens 2048] [--llm-kwargs '{"tensor_parallel_size": 2}']
#   compare: python kld_topk.py compare ref.jsonl test.jsonl
#   self-test (no GPU): python kld_topk.py selftest
# "dump" runs vLLM offline and saves the top-k prompt logprobs of every position.
# "compare" prints mean KL(ref || test) (top-k + tail-bucket approximation),
# top-1 agreement and the mean NLL of the real next token for both runs.
# The compare and selftest parts were checked without a GPU; the dump part
# needs vLLM and a GPU and was NOT run when these notes were written.
import argparse, json, math, sys


def dump(args):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_logprobs=args.k, **json.loads(args.llm_kwargs))
    tok = llm.get_tokenizer()
    texts = [t for t in open(args.prompts).read().split("\n\n") if t.strip()]
    prompts = [{"prompt_token_ids": tok.encode(t)[: args.max_tokens]} for t in texts]
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=args.k)
    with open(args.out, "w") as f:
        for p, o in zip(prompts, llm.generate(prompts, sp)):
            pos = [None if d is None else {str(t): lp.logprob for t, lp in d.items()}
                   for d in o.prompt_logprobs]
            f.write(json.dumps({"ids": p["prompt_token_ids"], "lp": pos}) + "\n")


def kl_topk(ref, test, k):
    """KL(ref || test) over ref's top-k tokens plus one 'everything else' bucket."""
    top = sorted(ref, key=ref.get, reverse=True)[:k]
    floor = min(test.values())
    kl, pr_sum, pq_sum = 0.0, 0.0, 0.0
    for t in top:
        lr, lq = ref[t], test.get(t, floor)
        kl += math.exp(lr) * (lr - lq)
        pr_sum += math.exp(lr); pq_sum += math.exp(lq)
    r_tail, q_tail = max(1 - pr_sum, 1e-12), max(1 - pq_sum, 1e-12)
    return kl + r_tail * (math.log(r_tail) - math.log(q_tail))


def compare(ref_path, test_path, k=20):
    kls, agree, nll_r, nll_t, n = 0.0, 0, 0.0, 0.0, 0
    for lr, lt in zip(open(ref_path), open(test_path)):
        r, t = json.loads(lr), json.loads(lt)
        assert r["ids"] == t["ids"], "runs must use identical prompt token ids"
        for i in range(1, len(r["ids"])):
            dr, dt = r["lp"][i], t["lp"][i]
            real = str(r["ids"][i])
            kls += kl_topk(dr, dt, k)
            agree += max(dr, key=dr.get) == max(dt, key=dt.get)
            nll_r -= dr[real]; nll_t -= dt[real]
            n += 1
    return {"positions": n, "kld_nats_per_token": kls / n, "top1_agreement": agree / n,
            "nll_ref": nll_r / n, "nll_test": nll_t / n}


def selftest():
    import random
    random.seed(0)

    def dist(logits):
        z = math.log(sum(math.exp(x) for x in logits))
        return {str(i): x - z for i, x in enumerate(logits)}

    a = dist([random.gauss(0, 2) for _ in range(50)])
    b = dist([v + random.gauss(0, 0.1) for v in a.values()])
    exact = sum(math.exp(a[t]) * (a[t] - b[t]) for t in a)
    assert abs(kl_topk(a, a, 20)) < 1e-12
    assert abs(kl_topk(a, b, 50) - exact) < 1e-9, (kl_topk(a, b, 50), exact)
    approx = kl_topk(a, b, 20)
    assert 0 <= approx <= exact * 1.5 + 1e-9, (approx, exact)
    print(f"selftest OK: exact KL {exact:.6f}, top-20 approximation {approx:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("model"); d.add_argument("prompts"); d.add_argument("out")
    d.add_argument("--k", type=int, default=20)
    d.add_argument("--max-tokens", type=int, default=2048)
    d.add_argument("--llm-kwargs", default="{}")
    c = sub.add_parser("compare"); c.add_argument("ref"); c.add_argument("test")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args)
    elif args.cmd == "compare":
        print(json.dumps(compare(args.ref, args.test), indent=2))
    else:
        selftest()
