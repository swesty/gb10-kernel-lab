"""Compare two lp_check.py captures (original vs patched attention kernel).

Teacher-forced: per prompt position, |logprob_A - logprob_B| of the actual token and top-1 agreement.
Greedy: first divergence and the top-5 candidates of both arms at that step.
Usage: python3 lp_compare.py lp-A-original.json lp-B-patched.json
"""
import json
import math
import statistics
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])


def table(entry):
    # vLLM prompt_logprobs: one dict per position {token_id: {"logprob", "rank", "decoded_token"}} (first is None)
    return {int(k): v for k, v in entry.items()} if entry else None


pa, pb = a.get("prompt_logprobs"), b.get("prompt_logprobs")
if pa and pb:
    diffs, top_agree, n, worst = [], 0, 0, (0.0, -1)
    for i, (ea, eb) in enumerate(zip(pa, pb)):
        ta, tb = table(ea), table(eb)
        if not ta or not tb:
            continue
        actual = [k for k, v in ta.items() if v.get("rank") is not None and k in tb]
        # the actual prompt token is the entry present in both whose rank may exceed top-k; use rank-1 ids too
        best_a = min(ta.items(), key=lambda kv: kv[1]["rank"])[0]
        best_b = min(tb.items(), key=lambda kv: kv[1]["rank"])[0]
        top_agree += best_a == best_b
        n += 1
        common = set(ta) & set(tb)
        for tok in common:
            d = abs(ta[tok]["logprob"] - tb[tok]["logprob"])
            if math.isfinite(d):
                diffs.append(d)
                if d > worst[0]:
                    worst = (d, i)
    diffs.sort()
    q = lambda f: diffs[min(len(diffs) - 1, int(f * len(diffs)))]
    # top-1 log-probability per position (the model's confidence in its own best guess), where top-1 agrees
    top1 = []
    for ea, eb in zip(pa, pb):
        ta, tb = table(ea), table(eb)
        if not ta or not tb:
            continue
        ka, va = min(ta.items(), key=lambda kv: kv[1]["rank"])
        kb, vb = min(tb.items(), key=lambda kv: kv[1]["rank"])
        if ka == kb:
            top1.append(abs(va["logprob"] - vb["logprob"]))
    top1.sort()
    t1 = lambda f: top1[min(len(top1) - 1, int(f * len(top1)))]
    print(f"teacher-forced positions compared: {n}")
    print(f"  top-1 agreement: {100 * top_agree / n:.2f}%")
    print(f"  |top-1 logprob diff| where top-1 agrees: median {t1(0.5):.4f}  p90 {t1(0.9):.4f}  p99 {t1(0.99):.4f}  "
          f"max {top1[-1]:.4f}; >0.1 at {100 * sum(d > 0.1 for d in top1) / len(top1):.1f}% of positions")
    print(f"  |logprob diff| over shared top-5 entries: mean {statistics.mean(diffs):.4f}  median {q(0.5):.4f}  "
          f"p99 {q(0.99):.4f}  max {diffs[-1]:.4f} (position {worst[1]})")
else:
    print("prompt_logprobs missing:", a.get("prompt_logprobs_error"), b.get("prompt_logprobs_error"))

ga, gb = a["greedy_logprobs"], b["greedy_logprobs"]
div = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x["token"] != y["token"]), None)
print(f"\ngreedy: diverges at step {div}")
if div is not None:
    for name, g in (("original", ga), ("patched", gb)):
        top = ", ".join(f"{t!r} {lp:.3f}" for t, lp in g[div]["top"])
        print(f"  {name:8s} step {div}: chose {g[div]['token']!r}; top-5: {top}")
    shared = [abs(x["logprob"] - y["logprob"]) for x, y in zip(ga[:div], gb[:div])]
    if shared:
        print(f"  before divergence: max |logprob diff| of chosen tokens {max(shared):.4f}")
