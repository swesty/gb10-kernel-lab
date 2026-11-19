#!/usr/bin/env bash
# Usage: ncu.sh <label> <kv_warps> [extra harness args]  -> ~/ncu-side/harness/<label>.ncu-rep
L=$1; KW=$2; shift 2
( while true; do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); [ "$a" -lt 3 ] && docker kill attn-harness >/dev/null 2>&1 && echo "WATCHDOG KILL avail=${a}G"; sleep 1; done ) & WD=$!; trap "kill $WD" EXIT
docker run --rm --name attn-harness --gpus all --ipc=host --cap-add SYS_ADMIN -e CUTE_DSL_ARCH=sm_121a -e B12X_COMPILE_WORKERS=1 -e B12X_SELECTED_PIPELINE=${PIPE:-0} -e B12X_SELECTED_TMA=${TMA:-0} -e B12X_SELECTED_TMA_V=${TMAV:-0} ${PATCH:+-v $HOME/ncu-side/harness/_selected_forward_patched.py:/usr/local/lib/python3.12/dist-packages/b12x/attention/paged/_selected_forward.py:ro} \
  -v $HOME/ncu-side/harness:/h -v $HOME/ncu-side/hcache:/root/.cache --entrypoint /usr/local/cuda/bin/ncu 06e0b04bcba1 \
  --profile-from-start off --kernel-name regex:SelectedPositionPagedForward --launch-count 1 \
  --set detailed --section WarpStateStats --section SchedulerStats --section InstructionStats --section MemoryWorkloadAnalysis_Tables \
  --cache-control all --clock-control none -f -o /h/$L python3 /h/attn_harness.py --kv-warps $KW --profile-one $KW --iters 3 --check-rows 4 "$@" 2>&1 | grep -E "==PROF==|==ERROR==|us/call|^ +[0-9]"
