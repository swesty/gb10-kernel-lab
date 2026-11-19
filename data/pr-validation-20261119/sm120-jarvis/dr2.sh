#!/bin/bash
# dr2.sh <stock|patched> <entry> [args]  -- FlashInfer 0.7.3 (== main d3f59f0 for both files) in vllm/vllm-openai:v0.34.0
V=$1; E=$2; shift 2
B=/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x
M=(); if [ "$V" = patched ]; then M=(-v $HOME/fi-pr/patched/_cute_prefill.py:$B/sequence/mtp_feedback/_cute_prefill.py:ro -v $HOME/fi-pr/patched/_selected_forward.py:$B/attention/paged/_selected_forward.py:ro); fi
mkdir -p $HOME/fi-pr/cache-$V
docker run --rm --gpus all --ipc=host -e PYTHONDONTWRITEBYTECODE=1 -e HARNESS_FLUSH_MB=512 "${M[@]}" \
  -v $HOME/fi-pr:/w -v $HOME/fi-pr/cache-$V:/root/.cache -w /w --entrypoint $E vllm/vllm-openai:v0.34.0 "$@" 2>&1 | grep -vE "UserWarning|warnings.warn|use_fast"
