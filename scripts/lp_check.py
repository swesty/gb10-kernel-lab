"""Log-probability equivalence capture for one Flash Next arm (original or patched attention kernel).

1. Teacher-forced: /v1/completions on a fixed ~7K-token real-text prompt with prompt_logprobs=5, max_tokens=1.
   Every prompt position gets the model's log-probability of the actual next token plus its top-5, which
   exercises the prefill (direct) attention path on identical inputs in both arms.
2. Greedy generation: chat completion, temperature 0, 96 tokens, logprobs with top_logprobs=5, which shows how
   close the top candidates were at the point where the two arms' texts diverged (decode/split path).
Writes /home/spark/ncu-side/ab/lp-<label>.json.  Usage: python3 lp_check.py <label>
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8112"
MODEL = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
TEXT = open("/home/spark/.cache/vllm/nsys-flashnext/qsa-dump/prompt.txt").read()
label = sys.argv[1]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


out = {"label": label}
try:
    r = post("/v1/completions", {"model": MODEL, "prompt": TEXT[:30000], "max_tokens": 1, "temperature": 0.0,
                                 "prompt_logprobs": 5})
    out["prompt_logprobs"] = r["choices"][0].get("prompt_logprobs")
    out["prompt_tokens"] = r["usage"]["prompt_tokens"]
except Exception as exc:  # e.g. prompt_logprobs unsupported with speculative decoding
    out["prompt_logprobs_error"] = repr(exc)

r = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 96, "temperature": 0.0, "logprobs": True,
                                  "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False},
                                  "messages": [{"role": "user", "content":
                                                "Summarize the following documentation in three sentences.\n\n"
                                                + TEXT[:30000]}]})
choice = r["choices"][0]
out["greedy_text"] = choice["message"]["content"]
out["greedy_logprobs"] = [
    {"token": t["token"], "logprob": t["logprob"],
     "top": [(c["token"], c["logprob"]) for c in t.get("top_logprobs", [])]}
    for t in (choice.get("logprobs") or {}).get("content", [])
]
json.dump(out, open(f"/home/spark/ncu-side/ab/lp-{label}.json", "w"))
print(label, "prompt_tokens", out.get("prompt_tokens"), "prompt_logprobs",
      "ok" if out.get("prompt_logprobs") else out.get("prompt_logprobs_error"),
      "greedy tokens", len(out["greedy_logprobs"]))
