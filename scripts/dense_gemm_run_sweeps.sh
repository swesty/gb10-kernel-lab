#!/bin/bash
# Stock-scheduler full tactic grid per shape, then scheduler-raster experiments (process-local patch).
cd /tmp/gemmexp
for s in qkvz gu down ba attn_qkv; do
  B12X_COMPILE_CACHE_DIR=/tmp/b12x-gemm-exp timeout 3000 python3 dense_gemm_harness.py sweep --recipe mxfp8 --shapes $s --iters 5 --placements 6 --json /tmp/gemmexp/sw8_${s}.json > /tmp/gemmexp/sw8_${s}.log 2>&1
done
for S in rm=0,sw=1 rm=0,sw=8 rm=1,sw=8; do
  for s in qkvz gu down ba attn_qkv; do
    B12X_COMPILE_DISK_CACHE=0 HARNESS_SCHED=$S timeout 3000 python3 dense_gemm_harness.py sweep --recipe mxfp8 --shapes $s --iters 5 --placements 6 --configs "default,64x128:128:0:1,128x128:128:0:0,128x128:64:0:0,128x64:128:0:1,64x64:128:0:1" --json /tmp/gemmexp/sw8_${s}_sched_$S.json > /tmp/gemmexp/sw8_${s}_sched_$S.log 2>&1
  done
done
echo done > /tmp/gemmexp/sweeps.done
