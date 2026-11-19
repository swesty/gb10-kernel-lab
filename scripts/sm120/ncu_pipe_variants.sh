#!/bin/bash
# Shared-memory counters for the pipelined attention variants on sm_120 (run beside prod; counts do not depend on load).
# The first version of this loop inside all.sh's follow-up passed the switches wrongly and produced stock counts; this is the corrected one.
cd ~/sm120-check
M=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__sass_inst_executed_op_ldsm.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,gpu__time_duration.sum
run() { n=$1; shift; env "$@" NCU=1 ./dr.sh /ncu/ncu --profile-from-start off --kernel-name regex:SelectedPositionPagedForward --launch-count 1 --metrics $M --csv python3 attn_harness.py --kv-warps 2 --profile-one 2 --iters 3 --check-rows 4 2>&1 | grep -E '^"' > out/ncu_m_$n.csv; }
run pipe PATCH=1 PIPE=1; run pipeswz PATCH=1 PIPE=1 SWZ=1; run all PATCH=1 PIPE=1 SPLITQK=1 SWZ=1
