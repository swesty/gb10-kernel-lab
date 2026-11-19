#!/usr/bin/env bash
# soak.sh [port] — stability/robustness run of the optimised Flash Next engine (spark-5 :8112) under mixed load.
# Exercises: long prompts (many prefill chunks -> PLE prefetch), concurrent mixed prefill/decode batches
# (prefetch must fall back; fused paths see ragged batch sizes), decode-heavy load. Then checks logs, health,
# memory and OOM kills, and runs the real-text + logprob checks.
set -u
PORT=${1:-8112}; C=vllm_qwen38_next; M=local-inference-lab/Qwen3.8-Flash-Next-NVFP4; AB=~/ncu-side/ab
START=$(date -u +%Y-%m-%dT%H:%M:%S)
mem() { awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo; }
B() {
    docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 $C vllm bench serve --backend vllm \
        --base-url http://127.0.0.1:$PORT --model $M --dataset-name random --ignore-eos \
        --percentile-metrics ttft,tpot --metric-percentiles 50,99 "$@" 2>&1 \
        | grep -E "Successful requests|Failed requests|Benchmark duration|Request throughput|Output token throughput|Total token throughput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT"
}
echo "=== soak start $START UTC, MemAvailable $(mem)"
echo "--- 1. long prompts: 32K in / 256 out, 4 requests, c=1 (8 chunks each)"
B --random-input-len 32768 --random-output-len 256 --num-prompts 4 --max-concurrency 1
echo "--- 2. mixed: 1-16K in (range 0.9) / 512 out, 24 requests, c=4"
B --random-input-len 8192 --random-range-ratio 0.9 --random-output-len 512 --num-prompts 24 --max-concurrency 4
echo "--- 3. decode-heavy: 2K in / 256 out, 32 requests, c=8"
B --random-input-len 2048 --random-output-len 256 --num-prompts 32 --max-concurrency 8
echo "--- 4. very long prompt: 128K in / 64 out, 1 request"
B --random-input-len 131072 --random-output-len 64 --num-prompts 1 --max-concurrency 1
echo "--- 5. bursty: 4-12K in / 128 out, 16 requests, c=8 (prefill-heavy concurrency)"
B --random-input-len 8192 --random-range-ratio 0.5 --random-output-len 128 --num-prompts 16 --max-concurrency 8
echo "--- 6. real-text quality + decode (greedy summaries, c=1 and c=4)"
python3 $AB/ab_real.py soak 2>&1 | grep -E "real-text|decode|greedy|text"
python3 $AB/lp_check.py soak
echo "--- checks"
echo "health: $(curl -sf localhost:$PORT/health >/dev/null && echo ok || echo FAIL)  MemAvailable $(mem)"
echo "engine errors since start: $(docker logs --since $START $C 2>&1 | grep -cE 'Traceback|ERROR|CUDA error|illegal memory')"
docker logs --since $START $C 2>&1 | grep -E "Traceback|ERROR|CUDA error|illegal memory" | head -5
echo "fusion fallbacks since start: $(docker logs --since $START $C 2>&1 | grep -c 'fell back')"
docker logs --since $START $C 2>&1 | grep "fell back" | head -5
echo "OOM kills since start: $(journalctl -k --since "$START" 2>/dev/null | grep -ci 'out of memory\|oom-kill' || true)"
echo "=== soak end $(date -u +%T) UTC"
