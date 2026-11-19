#!/bin/bash
# dr.sh <entry> [args...]  -- run one command in a scratch container with b12x 1.5.0 (rw copy) on sm_120.
E=$1; shift
docker run --rm --name sm120-check --gpus all --ipc=host --cap-add SYS_ADMIN \
  -e CUTE_DSL_ARCH=sm_120a -e B12X_COMPILE_WORKERS=4 -e PYTHONDONTWRITEBYTECODE=1 \
  -e B12X_SELECTED_PIPELINE=${PIPE:-0} -e B12X_SELECTED_SPLITQK=${SPLITQK:-0} -e B12X_SELECTED_SWZ333=${SWZ:-0} \
  -e B12X_MTP_TILE_N=${TN:-16} -e B12X_MTP_STAGES=${ST:-3} -e B12X_MTP_RASTER_N=${RN:-0} \
  -e B12X_COMPILE_DISK_CACHE=${DC:-0} -e HARNESS_SCHED=${SCHED:-} -e HARNESS_FLUSH_MB=${FLUSH:-512} \
  -v $HOME/sm120-check/b12x-1.5.0/b12x:/usr/local/lib/python3.12/dist-packages/b12x \
  ${PATCH:+-v $HOME/sm120-check/h/_selected_forward_patched.py:/usr/local/lib/python3.12/dist-packages/b12x/attention/paged/_selected_forward.py:ro} \
  ${NCU:+-v /opt/nvidia/nsight-compute/2026.1.1:/ncu:ro} -v $HOME/sm120-check/h:/h -v $HOME/sm120-check/out:/out -v $HOME/sm120-check/cache:/root/.cache -w /h \
  --entrypoint $E vllm-openai:v0.30.0-pr55601-b12x130 "$@" 2>&1 | grep -vE "use_fast|UserWarning|warnings.warn"
