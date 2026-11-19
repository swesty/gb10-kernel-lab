#!/usr/bin/env python3
"""ab_bench.py ARM PORT OUTDIR — one A/B arm against a vLLM server (stdlib only).

Measures: real-text decode (c=1, c=4) with streaming TTFT/decode tok/s, DFlash acceptance
from /metrics deltas, a JSON-mode check, a tool-call check, and teacher-forced prompt logprobs
(top-1 per position) for cross-arm equivalence. Prefill (random 16K) is run separately with
`vllm bench serve` by the driver.
"""
import json, os, re, sys, threading, time, urllib.request

ARM, PORT, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "/models/qwen38-dflash2-20260824"
os.makedirs(OUT, exist_ok=True)


def post(path, body, stream=False, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_drafts_total"):
        m = re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.e+]+)$", txt, re.M)
        out[name] = sum(float(x) for x in m) if m else None
    return out


TEXT = open("/usr/share/common-licenses/GPL-3").read()
CHUNKS = [TEXT[i:i + 6000] for i in range(0, 48000, 6000)]
PROMPTS = [f"Summarise the following licence text in plain English, as a numbered list of its "
           f"main obligations (about 300 words).\n\n{c}" for c in CHUNKS[:4]] + [
    "Write a well-documented Python module implementing an LRU cache with TTL expiry, thread safety, "
    "and unit tests using unittest. Explain the design briefly first.",
    "Explain how paged attention and prefix caching work in an LLM inference server, with a worked "
    "example of block tables, in about 400 words.",
    "Write a bash script that rotates log files in a directory: keep 7 compressed generations, skip "
    "files open by a process, and log what it did. Include comments.",
    "Compare B-trees and LSM-trees for a write-heavy key-value store. Cover write amplification, read "
    "amplification, compaction and crash recovery. Use headings.",
]


def chat_stream(prompt, max_tokens=400):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; usage = None; text = []
    with post("/v1/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if first is None:
                        first = time.time()
                    text.append(c)
    t1 = time.time()
    n = usage["completion_tokens"]
    return {"ttft": first - t0, "dec_s": t1 - first, "tokens": n, "tps": (n - 1) / (t1 - first),
            "text": "".join(text)}


def run_conc(prompts, conc):
    res = [None] * len(prompts); idx = [0]; lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = idx[0]; idx[0] += 1
            if i >= len(prompts):
                return
            res[i] = chat_stream(prompts[i])
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]
    return res, time.time() - t0


summary = {"arm": ARM}
chat_stream("Say hello.", 16)  # warm-up

for conc, prompts in ((1, PROMPTS[:4] + PROMPTS[4:6]), (4, PROMPTS + PROMPTS[:4])):
    m0 = metrics(); res, wall = run_conc(prompts, conc); m1 = metrics()
    acc = drafts = None
    a, d = "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total"
    if m0[a] is not None:
        acc, drafts = m1[a] - m0[a], m1[d] - m0[d]
    summary[f"c{conc}"] = {
        "requests": len(res), "mean_tps_per_req": sum(r["tps"] for r in res) / len(res),
        "mean_ttft_s": sum(r["ttft"] for r in res) / len(res),
        "agg_tps": sum(r["tokens"] for r in res) / wall, "accept_rate": (acc / drafts) if drafts else None,
        "accepted": acc, "drafted": drafts}
    if conc == 1:
        json.dump([r["text"] for r in res], open(f"{OUT}/greedy_c1.json", "w"))

# JSON-mode correctness
r = json.load(post("/v1/chat/completions", {
    "model": MODEL, "temperature": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False},
    "response_format": {"type": "json_object"},
    "messages": [{"role": "user", "content": "Return a JSON object with keys city (string), population "
                  "(integer) and landmarks (array of 3 strings) for Paris. JSON only."}]}))
try:
    j = json.loads(r["choices"][0]["message"]["content"])
    summary["json_ok"] = isinstance(j.get("population"), int) and len(j.get("landmarks", [])) == 3
except Exception as e:
    summary["json_ok"] = f"FAIL {e}"

# Tool-call correctness
r = json.load(post("/v1/chat/completions", {
    "model": MODEL, "temperature": 0, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content": "What's the weather in Boston in celsius?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                              "required": ["city", "unit"]}}}]}))
tc = r["choices"][0]["message"].get("tool_calls") or []
try:
    args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
    summary["tool_ok"] = bool(tc) and tc[0]["function"]["name"] == "get_weather" and \
        "boston" in args.get("city", "").lower() and args.get("unit") == "celsius"
except Exception as e:
    summary["tool_ok"] = f"FAIL {e}"

# Teacher-forced prompt logprobs (top-1 token + its logprob per position)
lp_prompt = TEXT[24000:35000]
r = json.load(post("/v1/completions", {"model": MODEL, "prompt": lp_prompt, "max_tokens": 1,
                                        "temperature": 0, "prompt_logprobs": 1}))
plp = r["choices"][0].get("prompt_logprobs") or []
top = []
for pos in plp[1:]:
    if not pos:
        top.append(None); continue
    best = max(pos.items(), key=lambda kv: kv[1]["logprob"])
    top.append([best[0], best[1]["logprob"]])
json.dump(top, open(f"{OUT}/prompt_top1.json", "w"))
summary["prompt_logprob_positions"] = len(top)

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
