#!/bin/bash
# Alternating A/B of B12X_MTP_FENCE=0 vs 1. Args: none. Output: one line per run.
cd /tmp/b1fence
run() { # tile stages raster fence tokens extra...
  B12X_COMPILE_CACHE_DIR=/tmp/b1fence/cache$4 B12X_MTP_TILE_N=$1 B12X_MTP_STAGES=$2 B12X_MTP_RASTER_N=$3 B12X_MTP_FENCE=$4 \
    python3 fence_harness.py --tokens $5 "${@:6}" 2>&1 | grep -v -i warn | grep "B12X_MTP_FENCE\|stress" | sed "s/^/T=$5 $6$7 /"
}
for cfg in "128 2 1" "16 3 0"; do
  for T in 4096 1024 256; do
    for rep in 1 2 3; do
      for F in 0 1; do run $cfg $F $T --iters 50; done
      for F in 1 0; do run $cfg $F $T --iters 21 --flush-mb 512; done
    done
  done
done
for F in 0 1; do run 128 2 1 $F 4096 --iters 5 --stress 500; done
