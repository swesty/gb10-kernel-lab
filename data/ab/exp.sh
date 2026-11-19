#!/usr/bin/env bash
# Usage: exp.sh <label>  — prefill + decode checks against the running :8112 engine
L=$1; O=~/ncu-side/ab/exp-$L.txt
{ echo "=== $L $(date -u +%T)"
  docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vllm_qwen38_next vllm bench serve --backend vllm --base-url http://127.0.0.1:8112 --model local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --dataset-name random --ignore-eos --random-input-len 1024 --random-output-len 16 --num-prompts 2 --max-concurrency 1 >/dev/null 2>&1
  echo "--- random 16K prefill x4"; docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vllm_qwen38_next vllm bench serve --backend vllm --base-url http://127.0.0.1:8112 --model local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --dataset-name random --ignore-eos --random-input-len 16384 --random-output-len 8 --num-prompts 4 --max-concurrency 1 2>&1 | grep -E "Mean TTFT|Median TTFT|Failed"
  python3 ~/ncu-side/ab/ab_real.py $L 2>&1 | grep -E "real-text|decode"
} 2>&1 | tee $O
