#!/bin/bash
# fencecheck.sh: B1 PR (patched) vs B1 PR + gated proxy fence (fenced) on spark-5:
# timing at the PR-table sizes, ncu register/spill check, the test file with the new test, and a SASS fence check.
cd ~/fi-pr; mkdir -p out/fence
IMG=06e0b04bcba1; B=/usr/local/lib/python3.12/dist-packages/b12x; NCU=/opt/nvidia/nsight-compute/2025.3.1; FLUSH=96
( while true; do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); [ "$a" -lt 3 ] && docker kill fi-pr-fc-patched fi-pr-fc-fenced >/dev/null 2>&1 && echo "WATCHDOG KILL avail=${a}G" >> out/fence/watchdog.txt; sleep 1; done ) & WD=$!; trap "kill $WD" EXIT
M=launch__registers_per_thread,launch__occupancy_limit_registers,launch__shared_mem_per_block_dynamic,launch__grid_size,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,gpu__time_duration.sum
run() { V=$1; E=$2; shift 2
  docker run --rm --name fi-pr-fc-$V --gpus all --ipc=host --cap-add SYS_ADMIN -e CUTE_DSL_ARCH=sm_121a -e B12X_COMPILE_WORKERS=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e HARNESS_FLUSH_MB=$FLUSH -v $HOME/fi-pr/$V/_cute_prefill.py:$B/sequence/mtp_feedback/_cute_prefill.py:ro \
    -v $NCU:/ncu:ro -v $HOME/fi-pr:/w -v $HOME/fi-pr/cache-fc-$V:/root/.cache -w /w --entrypoint $E $IMG "$@" 2>&1 | grep -vE "UserWarning|warnings.warn|use_fast"
}
mkdir -p cache-fc-patched cache-fc-fenced
for r in 1 2; do
  for T in 16 64 256 1024 4096; do
    if [ $r = 1 ]; then O="patched fenced"; else O="fenced patched"; fi
    for V in $O; do run $V python3 mtp_gemm_harness.py --tokens $T --iters 20 --flush-mb $FLUSH | grep -E "TFLOPS" | sed "s/^/r=$r T=$T $V /" >> out/fence/mtp.txt; done
  done
done
for V in patched fenced; do for T in 16 4096; do
  echo "== $V tokens=$T" >> out/fence/regcheck.csv
  run $V /ncu/ncu --kernel-name regex:MTPPrefill --launch-skip 1 --launch-count 1 --clock-control none --cache-control none \
    --metrics $M --csv python3 mtp_gemm_harness.py --tokens $T --iters 3 --flush-mb $FLUSH | grep -E '^"' >> out/fence/regcheck.csv
done; done
echo "== fenced pytest test_mtp_feedback_plus.py ($(date -u +%FT%TZ))" > out/fence/pytest_fenced.txt
run fenced timeout 1800 bash -c "pip install -q pytest >/dev/null 2>&1; python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/sequence/test_mtp_feedback_plus.py" >> out/fence/pytest_fenced.txt 2>&1
run fenced python3 /w/sass_fenced.py /root/.cache/b12x /w/out/fence/sass > out/fence/sass_check.txt 2>&1
run patched python3 /w/sass_fenced.py /root/.cache/b12x /w/out/fence/sass_patched > out/fence/sass_check_patched.txt 2>&1
echo done > out/fence/DONE
