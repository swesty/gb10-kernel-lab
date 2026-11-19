#!/usr/bin/env bash
# Usage: suite.sh <label>  — benchmark the running :8112 engine; results in ~/ncu-side/ab/<label>.txt
L=$1; O=~/ncu-side/ab/$L.txt; U=http://127.0.0.1:8112; M=local-inference-lab/Qwen3.8-Flash-Next-NVFP4
B() { docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vllm_qwen38_next vllm bench serve --backend vllm --base-url $U --model $M --dataset-name random --ignore-eos --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90 "$@" 2>&1 | grep -E "Mean TTFT|Median TTFT|P90 TTFT|Mean TPOT|Median TPOT|Output token throughput|Total token throughput|Acceptance rate|Acceptance length|Successful requests|Failed requests"; }
{
echo "=== $L $(date -u +%T)"
echo "--- warmup"; B --random-input-len 1024 --random-output-len 16 --num-prompts 2 --max-concurrency 1 >/dev/null
echo "--- prefill 16K x4, c=1, out 8";  B --random-input-len 16384 --random-output-len 8 --num-prompts 4 --max-concurrency 1
echo "--- 8K in / 256 out, c=1, x6";    B --random-input-len 8192 --random-output-len 256 --num-prompts 6 --max-concurrency 1
echo "--- 8K in / 256 out, c=4, x12";   B --random-input-len 8192 --random-output-len 256 --num-prompts 12 --max-concurrency 4
echo "--- real text 14.5K, c=1, x3 (nonce defeats prefix cache), plus greedy output check"
python3 - <<"PY"
import json, urllib.request, time, uuid
text = open("/home/spark/.cache/vllm/nsys-flashnext/qsa-dump/prompt.txt").read()
def ask(content, n, temp=0.0):
    body = {"model": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "max_tokens": n, "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8112/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return time.time() - t, r
for i in range(3):
    dt, r = ask(f"[{uuid.uuid4()}] Summarize the following documentation in one sentence.\n\n" + text, 1)
    print(f"real-text prefill {r[usage][prompt_tokens]} tok: {dt:.2f} s")
dt, r = ask("Summarize the following documentation in three sentences.\n\n" + text[:30000], 96)
print("greedy output:", json.dumps(r["choices"][0]["message"]["content"]))
PY
} > $O 2>&1
cat $O
