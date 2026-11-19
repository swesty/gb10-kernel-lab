"""Real-text A/B checks against the running :8112 engine (Flash Next).

1. Greedy output (temperature 0, 96 tokens) on a fixed real-text prompt: compare text across arms.
2. Real-text prefill TTFT, 3 runs (a random prefix defeats the prefix cache).
3. Decode on real text: 4 concurrent requests, temperature 0, 256 tokens each; reports per-request
   decode tokens/s and the engine's MTP acceptance over the run (from /metrics deltas).
Usage: python3 ab_real.py <label>
"""
import json
import sys
import threading
import time
import urllib.request
import uuid

URL = "http://127.0.0.1:8112"
MODEL = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
TEXT = open("/home/spark/.cache/vllm/nsys-flashnext/qsa-dump/prompt.txt").read()
label = sys.argv[1]


def ask(content, max_tokens, stream=False):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        r = json.load(urllib.request.urlopen(req, timeout=900))
        return time.time() - t0, r
    first, n, last = None, 0, None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                n = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                if first is None:
                    first = time.time()
                last = time.time()
    return first - t0, last - first, n  # n = completion tokens from the usage report


def spec():
    text = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    vals = {}
    for line in text.splitlines():
        for key in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(key + "{"):
                vals[key] = float(line.rsplit(" ", 1)[1])
    return vals


print(f"=== {label}")
dt, r = ask("Summarize the following documentation in three sentences.\n\n" + TEXT[:30000], 96)
print("greedy output:", json.dumps(r["choices"][0]["message"]["content"]))
for i in range(3):
    dt, r = ask(f"[{uuid.uuid4()}] Summarize the following documentation in one sentence.\n\n" + TEXT, 1)
    print(f"real-text prefill {r['usage']['prompt_tokens']} tokens: {dt:.2f} s")

prompts = [TEXT[i * 6000:(i + 1) * 6000] for i in range(4)]
before = spec()
results = [None] * 4


def worker(i):
    results[i] = ask(f"Explain the following documentation in detail.\n\n{prompts[i]}", 256, stream=True)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]
after = spec()
drafts = after["vllm:spec_decode_num_drafts_total"] - before["vllm:spec_decode_num_drafts_total"]
dtok = after["vllm:spec_decode_num_draft_tokens_total"] - before["vllm:spec_decode_num_draft_tokens_total"]
acc = after["vllm:spec_decode_num_accepted_tokens_total"] - before["vllm:spec_decode_num_accepted_tokens_total"]
rates = [n / dec for _, dec, n in results]
print(f"decode c=4 real text: per-request {[round(x, 1) for x in rates]} tokens/s, "
      f"mean TTFT {sum(r[0] for r in results) / 4:.2f} s, "
      f"MTP acceptance {100 * acc / max(dtok, 1):.1f}% ({acc / max(drafts, 1):.2f} accepted per step)")

before = spec()
ttft1, dec1, n1 = ask("Explain the following documentation in detail.\n\n" + prompts[0], 256, stream=True)
after = spec()
acc1 = after["vllm:spec_decode_num_accepted_tokens_total"] - before["vllm:spec_decode_num_accepted_tokens_total"]
dtok1 = after["vllm:spec_decode_num_draft_tokens_total"] - before["vllm:spec_decode_num_draft_tokens_total"]
print(f"decode c=1 real text: {n1 / dec1:.1f} tokens/s (TTFT {ttft1:.2f} s), MTP acceptance {100 * acc1 / max(dtok1, 1):.1f}%")
