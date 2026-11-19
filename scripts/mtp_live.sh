#!/usr/bin/env bash
# Usage: mtp_live.sh  — deploy the MTP feedback projection fix (tile N 128, 2 stages, N-fastest raster) on the
# final build, then run exp.sh and lp_check.py as arm "mtp-fix". Leaves the engine running with the fix.
set -u
C=vllm_qwen38_next; D=~/.cache/vllm/nsys-flashnext; AB=~/ncu-side/ab
docker exec $C pkill -INT -f "bin/vllm serve" 2>/dev/null
for i in $(seq 1 120); do sleep 5
  [ -z "$(docker exec $C pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && \
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && break
done
docker cp ~/ncu-side/mtp_exp_patch.py $C:/tmp/mtp_exp_patch.py
docker exec $C python3 /tmp/mtp_exp_patch.py --inplace || exit 5
grep -q B12X_MTP_TILE_N $D/extra.env || printf "B12X_MTP_TILE_N=128\nB12X_MTP_STAGES=2\nB12X_MTP_RASTER_N=1\n" >> $D/extra.env
cat $D/extra.env
bash ~/ncu-side/start.sh patched || exit 2
docker logs --since 5m $C 2>&1 | grep -i "mtp" | grep -iv "warn" | tail -3
bash $AB/exp.sh mtp-fix
python3 $AB/lp_check.py mtp-fix
