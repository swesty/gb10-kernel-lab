#!/bin/bash
V=$1; E=$2; shift 2
B=/usr/local/lib/python3.12/dist-packages/b12x
M=(); if [ "$V" = patched ]; then M=(-v $HOME/fi-pr/patched/_cute_prefill.py:$B/sequence/mtp_feedback/_cute_prefill.py:ro -v $HOME/fi-pr/patched/_selected_forward.py:$B/attention/paged/_selected_forward.py:ro); fi
mkdir -p $HOME/fi-pr/cache-$V
docker run --rm --name fi-pr-$V --gpus all --ipc=host -e CUTE_DSL_ARCH=sm_121a -e B12X_COMPILE_WORKERS=1 -e PYTHONDONTWRITEBYTECODE=1 -e HARNESS_FLUSH_MB=96 "${M[@]}" \
  -v $HOME/fi-pr:/w -v $HOME/fi-pr/cache-$V:/root/.cache -w /w --entrypoint $E 06e0b04bcba1 "$@" 2>&1 | grep -vE "UserWarning|warnings.warn|use_fast"
