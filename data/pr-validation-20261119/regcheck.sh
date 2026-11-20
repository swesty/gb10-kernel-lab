#!/bin/bash
# regcheck.sh: registers / local-memory (spill) check for the MTP prefill GEMM, stock vs patched,
# plus the full test_mtp_feedback.py with the new test appended (patched only).
# Host settings come from env: IMG, B (b12x package dir in the image), ARCH, NCU (host ncu dir), FLUSH, PIP.
cd ~/fi-pr; mkdir -p out
M=launch__registers_per_thread,launch__occupancy_limit_registers,launch__shared_mem_per_block_dynamic,launch__grid_size,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,gpu__time_duration.sum
run() { # run <stock|patched> <entry> args...
  V=$1; E=$2; shift 2
  MNT=(); if [ "$V" = patched ]; then MNT=(-v $HOME/fi-pr/patched/_cute_prefill.py:$B/sequence/mtp_feedback/_cute_prefill.py:ro); fi
  docker run --rm --name fi-pr-reg-$V --gpus all --ipc=host --cap-add SYS_ADMIN ${ARCH:+-e CUTE_DSL_ARCH=$ARCH} -e B12X_COMPILE_WORKERS=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e HARNESS_FLUSH_MB=$FLUSH "${MNT[@]}" -v $NCU:/ncu:ro \
    -v $HOME/fi-pr:/w -v $HOME/fi-pr/cache-$V:/root/.cache -w /w --entrypoint $E $IMG "$@" 2>&1 | grep -vE "UserWarning|warnings.warn|use_fast"
}
for V in stock patched; do
  for T in 16 4096; do
    echo "== $V tokens=$T" >> out/regcheck_$V.csv
    run $V /ncu/ncu --kernel-name regex:MTPPrefill --launch-skip 1 --launch-count 1 --clock-control none --cache-control none \
      --metrics $M --csv python3 mtp_gemm_harness.py --tokens $T --iters 3 --flush-mb $FLUSH | grep -E '^"' >> out/regcheck_$V.csv
  done
done
echo "== patched pytest test_mtp_feedback.py + new test ($(date -u +%FT%TZ))" > out/pytest_mtp_patched_with_new_test.txt
run patched timeout 1800 bash -c "${PIP:+pip install -q pytest >/dev/null 2>&1;} python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/sequence/test_mtp_feedback_plus.py" >> out/pytest_mtp_patched_with_new_test.txt 2>&1
echo done > out/REGCHECK_DONE
