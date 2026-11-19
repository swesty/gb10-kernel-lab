#!/usr/bin/env bash
# Usage: dense_raster_live.sh — enable the N-major dense GEMM tile walk for N = 2,560 on the running build
# (final build + MTP fix), restart, run exp.sh and lp_check.py as arm "dense-raster". Leaves it running.
set -u
C=vllm_qwen38_next; D=~/.cache/vllm/nsys-flashnext; AB=~/ncu-side/ab
docker exec $C pkill -INT -f "bin/vllm serve" 2>/dev/null
for i in $(seq 1 120); do sleep 5
  [ -z "$(docker exec $C pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && \
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && break
done
docker cp ~/ncu-side/gemm/dense_raster_patch.py $C:/tmp/dense_raster_patch.py
docker exec $C python3 /tmp/dense_raster_patch.py || exit 5
grep -q B12X_DENSE_RASTER_N_MAX_CTAS $D/extra.env || printf "B12X_DENSE_RASTER_N_MAX_CTAS=20\nB12X_DENSE_RASTER_N_MIN_CTAS=20\n" >> $D/extra.env
cat $D/extra.env
bash ~/ncu-side/start.sh patched || exit 2
bash $AB/exp.sh dense-raster
python3 $AB/lp_check.py dense-raster
