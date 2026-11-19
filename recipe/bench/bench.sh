#!/bin/bash
# bench.sh [port] — quick prefill/decode check against a running Flash Next engine.
# Run once with the stock recipe and once with the gb10fast recipe, then compare.
# Uses the engine container's own `vllm bench serve`; HF_HUB_OFFLINE stops it from touching your HF cache refs.
set -euo pipefail
PORT=${1:-8000}
MODEL=local-inference-lab/Qwen3.8-Flash-Next-NVFP4
C=${CONTAINER:-$(docker ps --filter ancestor=vllm-node-b12x --format '{{.Names}}' | head -1)}
[ -n "$C" ] || { echo "no running vllm-node-b12x container; set CONTAINER=<name>" >&2; exit 1; }

bench() {
    docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$C" vllm bench serve --backend vllm \
        --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --dataset-name random --ignore-eos "$@" 2>&1 \
        | grep -E "Failed requests|Mean TTFT|Median TTFT|Mean TPOT|Output token throughput"
}

echo "container: $C  port: $PORT"
bench --random-input-len 1024 --random-output-len 16 --num-prompts 2 --max-concurrency 1 >/dev/null  # warm-up
echo "--- prefill: 16K random tokens, 8 output tokens, 4 requests one at a time"
bench --random-input-len 16384 --random-output-len 8 --num-prompts 4 --max-concurrency 1
echo "--- decode: 1K in, 256 out, 4 requests one at a time (random tokens: MTP acceptance is low, compare runs only)"
bench --random-input-len 1024 --random-output-len 256 --num-prompts 4 --max-concurrency 1
