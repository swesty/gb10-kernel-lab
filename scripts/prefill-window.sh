#!/usr/bin/env bash
# One nsys capture window around a single random 16K-token prefill (8 output tokens).
U=http://127.0.0.1:8112; M=local-inference-lab/Qwen3.8-Flash-Next-NVFP4
B() { docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 vllm_qwen38_next vllm bench serve --backend vllm --base-url $U --model $M --dataset-name random --random-output-len 8 --num-prompts 1 --max-concurrency 1 --ignore-eos "$@" 2>&1 | grep -E "Mean TTFT"; }
B --random-input-len 1024 >/dev/null   # warm-up outside the window
curl -s -X POST $U/start_profile; B --random-input-len 16384; curl -s -X POST $U/stop_profile
