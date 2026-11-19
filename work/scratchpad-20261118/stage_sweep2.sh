#!/usr/bin/env bash
# Upper-bound timing: hand-off barriers removed (wrong output, timing only)
D=/root/.cache/vllm/nsys-flashnext
run() {  # $1 = label, rest = env pairs
  label=$1; shift; envs=(-e HF_HUB_OFFLINE=1)
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "== $label"
  docker exec "${envs[@]}" vllm_qwen38_next timeout 900 python3 $D/moe_harness.py --iters 2 --live 16 --ref $D/moe-ref.pt --configs t32 2>&1 | grep -vE "Warning|warn|live route" | tail -3
}
run "no hand-off barriers, 2 stages" B12X_MOE_SOL_NO_HANDOFF=1 B12X_COMPILE_CACHE_DIR=$D/b12x-cache-sol2
run "no hand-off barriers, 4 stages" B12X_MOE_SOL_NO_HANDOFF=1 B12X_MOE_AB_STAGE=4 B12X_COMPILE_CACHE_DIR=$D/b12x-cache-sol4
