#!/usr/bin/env bash
set -u
U=http://127.0.0.1:8112; M=local-inference-lab/Qwen3.8-Flash-Next-NVFP4
chat() { python3 - "$1" <<"PY"
import json, sys, urllib.request
body = {"model": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "max_tokens": int(sys.argv[1]), "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True,
        "messages": [{"role": "user", "content": "Explain, step by step and with a worked numerical example, how a write-ahead log lets a database recover after a crash. Cover checkpoints, redo versus undo, and group commit."}]}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8112/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200)
print("usage", json.load(r)["usage"])
PY
}
chat 32 >/dev/null
echo "=== decode window $(date -u +%T)"; curl -s -X POST $U/start_profile; chat 48; curl -s -X POST $U/stop_profile; echo "=== end $(date -u +%T)"
echo "=== prefill window $(date -u +%T)"; curl -s -X POST $U/start_profile
docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vllm_qwen38_next vllm bench serve --backend vllm --base-url $U --model $M --dataset-name random --random-input-len 16384 --random-output-len 4 --num-prompts 1 --max-concurrency 1 --ignore-eos --request-timeout 7200 2>&1 | grep -E "Mean TTFT|Successful|Failed"
curl -s -X POST $U/stop_profile; echo "=== end $(date -u +%T)"
