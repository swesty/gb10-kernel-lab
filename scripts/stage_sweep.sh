#!/usr/bin/env bash
# Usage: stage_sweep.sh  — MoE harness at 2 (twice, noise floor), 3, 4 TMA stages
D=/root/.cache/vllm/nsys-flashnext
rm -f /home/spark/.cache/vllm/nsys-flashnext/moe-ref.pt
run() {  # $1 = env stage ("" = default), $2 = configs
  envs=(-e HF_HUB_OFFLINE=1)
  [ -n "$1" ] && envs+=(-e B12X_MOE_AB_STAGE=$1 -e B12X_COMPILE_CACHE_DIR=$D/b12x-cache-s$1)
  docker exec "${envs[@]}" vllm_qwen38_next python3 $D/moe_harness.py --iters 2 --live 16 --ref $D/moe-ref.pt --configs "$2" 2>&1 | grep -vE "Warning|warn|live route" | tail -4
}
echo "== baseline (2 stages), twice"; run "" t32,t32
echo "== 3 stages"; run 3 t32
echo "== 4 stages"; run 4 t32
