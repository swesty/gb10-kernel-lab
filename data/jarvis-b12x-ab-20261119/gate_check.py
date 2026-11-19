#!/usr/bin/env python3
"""gate_check.py ARM PORT OUTDIR — prefix-cache correctness + concurrency/memory for one arm.

Needs the server started with VLLM_SERVER_DEV_MODE=1 (for POST /reset_prefix_cache).
Checks, all greedy (temperature 0), single request at a time:
  1. cold vs cold  (reset between)            -> determinism floor
  2. cold vs warm  (same prompt, full hit)     -> prefix-hit restoration (GDN state)
  3. multi-turn: turn k sent after turns 1..k-1 (partial hit) and again (full hit),
     compared with a cold send of the same turn-k conversation
Each comparison records exact text match, first differing token index, and top-1 logprob deltas.
Then: c=16 mixed load (vllm bench is run by the driver) — here only GPU memory sampling helpers.
"""
import json, os, subprocess, sys, time, urllib.request

ARM, PORT, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "/models/qwen38-dflash2-20260824"
os.makedirs(OUT, exist_ok=True)


def post(path, body=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=timeout)) if body is not None else \
        urllib.request.urlopen(req, timeout=timeout).read()


def reset():
    post("/reset_prefix_cache")
    time.sleep(0.5)


def chat(messages, max_tokens=200):
    r = post("/v1/chat/completions", {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        "logprobs": True, "top_logprobs": 1, "chat_template_kwargs": {"enable_thinking": False}})
    ch = r["choices"][0]
    toks = [(t["token"], t["logprob"]) for t in (ch.get("logprobs") or {}).get("content") or []]
    cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"text": ch["message"]["content"], "toks": toks, "prompt_tokens": r["usage"]["prompt_tokens"],
            "cached": cached}


def cmp(a, b):
    n = min(len(a["toks"]), len(b["toks"]))
    first = next((i for i in range(n) if a["toks"][i][0] != b["toks"][i][0]), None)
    upto = first if first is not None else n
    d = sorted(abs(a["toks"][i][1] - b["toks"][i][1]) for i in range(upto)) or [0.0]
    return {"exact": a["text"] == b["text"], "first_diff_tok": first, "compared": n,
            "max_dlogp_before_diff": round(d[-1], 5), "cached_a": a["cached"], "cached_b": b["cached"]}


TEXT = open("/usr/share/common-licenses/GPL-3").read()
DOCS = [TEXT[0:12000], TEXT[12000:24000], TEXT[24000:35000]]
QS = ["List the three most important obligations in this text.",
      "Now explain the second one in more detail, with an example.",
      "Summarise our whole discussion in two sentences."]
res = {"arm": ARM, "cold_cold": [], "cold_warm": [], "multiturn": []}

for doc in DOCS:
    msgs = [{"role": "user", "content": f"{doc}\n\n{QS[0]}"}]
    reset(); c1 = chat(msgs)
    reset(); c2 = chat(msgs)
    w = chat(msgs)  # same prompt again: full prefix hit
    res["cold_cold"].append(cmp(c1, c2))
    res["cold_warm"].append(cmp(c2, w))

for doc in DOCS:
    convo = [{"role": "user", "content": f"{doc}\n\n{QS[0]}"}]
    reset(); t1 = chat(convo)
    convo += [{"role": "assistant", "content": t1["text"]}]
    for k in (1, 2):
        convo_k = convo + [{"role": "user", "content": QS[k]}]
        partial = chat(convo_k)          # earlier turns cached, new turn not
        full = chat(convo_k)             # whole prompt cached
        reset(); cold = chat(convo_k)    # no cache
        res["multiturn"].append({"turn": k + 1, "partial_vs_cold": cmp(cold, partial),
                                 "full_vs_cold": cmp(cold, full)})
        # continue the conversation from the cold answer, warming the cache again
        convo = convo_k + [{"role": "assistant", "content": cold["text"]}]
        chat(convo + [{"role": "user", "content": "ok"}], max_tokens=1)

def tally(rows, key=None):
    rows = [r[key] if key else r for r in rows]
    return {"n": len(rows), "exact": sum(r["exact"] for r in rows),
            "first_diff": [r["first_diff_tok"] for r in rows],
            "cached": [r["cached_b"] for r in rows]}

res["summary"] = {"cold_cold": tally(res["cold_cold"]), "cold_warm": tally(res["cold_warm"]),
                  "partial_vs_cold": tally(res["multiturn"], "partial_vs_cold"),
                  "full_vs_cold": tally(res["multiturn"], "full_vs_cold")}
json.dump(res, open(f"{OUT}/gate.json", "w"), indent=1)
print(json.dumps(res["summary"], indent=1))
