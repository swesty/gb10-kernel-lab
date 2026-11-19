#!/usr/bin/env bash
# Three nsys capture windows against the live engine. Each start/stop pair = one report.
set -u
U=http://127.0.0.1:8112; M=local-inference-lab/Qwen3.8-Flash-Next-NVFP4
D=/home/spark/.cache/vllm/nsys-flashnext
chat() {  # $1 = max_tokens
python3 - "$1" <<"PY"
import json, sys, urllib.request
body = {"model": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "max_tokens": int(sys.argv[1]), "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True,
        "messages": [{"role": "user", "content": "Explain, step by step and with a worked numerical example, how a write-ahead log lets a database recover after a crash. Cover checkpoints, redo versus undo, and group commit."}]}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8112/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
u = json.load(r)["usage"]; print("usage", u)
PY
}
spec() { curl -s $U/metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | sed "s/{.*}//" | tr "\n" " "; echo; }
win() { echo "=== window $1 $(date -u +%T)"; spec; curl -s -X POST $U/start_profile; }
end() { curl -s -X POST $U/stop_profile; spec; echo "=== end $(date -u +%T)"; }
chat 32 >/dev/null   # warm-up outside any window
win A-decode-bs1; chat 512; end
sleep 20
win B-prefill-16k
docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vllm_qwen38_next vllm bench serve --backend vllm --base-url $U --model $M --dataset-name random --random-input-len 16384 --random-output-len 8 --num-prompts 1 --max-concurrency 1 --ignore-eos 2>&1 | grep -E "Mean TTFT|Total input"
end
sleep 20
win C-decode-bs4; for i in 1 2 3 4; do chat 256 & done; wait; end
sleep 20; ls -la $D/*.nsys-rep 2>/dev/null
