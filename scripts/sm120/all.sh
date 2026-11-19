#!/bin/bash
# SM120 validation of the GB10 b12x findings. Stops prod qwen3.8-27b via llamactl, always restarts it on exit.
cd ~/sm120-check; O=out; mkdir -p $O
K=$(python3 -c "import yaml;print(yaml.safe_load(open(\"/home/jarvis/docker/llamactl/config.yaml\"))[\"auth\"][\"management_keys\"][0])")
log() { echo "[$(date -u +%H:%M:%S)] $* | gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB prod_ctr=$(docker ps -q --filter ancestor=vllm-openai:v0.30.0-pr55601 | wc -l)" | tee -a $O/driver.log; }
restore() {
  log "restoring prod"
  curl -s -X POST -H "Authorization: Bearer $K" http://127.0.0.1:8080/api/v1/instances/qwen3.8-27b/start >/dev/null
  t0=$(date +%s); until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 10; [ $(( $(date +%s)-t0 )) -gt 900 ] && break; done
  log "prod status $(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models) DONE"
  date -u +%FT%TZ > $O/outage_end
}
trap restore EXIT
log "stopping prod"
curl -s -X POST -H "Authorization: Bearer $K" http://127.0.0.1:8080/api/v1/instances/qwen3.8-27b/stop >/dev/null
for i in $(seq 1 60); do [ "$(docker ps -q --filter ancestor=vllm-openai:v0.30.0-pr55601 | wc -l)" = 0 ] && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2000 ] && break; sleep 3; done
date -u +%FT%TZ > $O/outage_start; log "prod stopped"
nvidia-smi -q -d CLOCK,POWER | grep -E "Graphics|SM |Memory |Power Limit|Power Draw" | head -12 > $O/gpu_state.txt

log "A mma_peak"; ./dr.sh /out/mma_peak > $O/mma_peak.txt

log "B mtp"
for F in 0 512; do
  ./dr.sh python3 mtp_gemm_harness.py --iters 20 --flush-mb $F > $O/mtp_stock_f$F.txt
  for v in 16/3/0 16/3/1 64/3/0 64/3/1 128/2/0 128/2/1 128/3/0 128/3/1; do IFS=/ read TN ST RN <<< "$v"
    TN=$TN ST=$ST RN=$RN ./dr.sh python3 mtp_gemm_harness.py --exp --iters 20 --flush-mb $F > $O/mtp_exp_${TN}_${ST}_${RN}_f$F.txt
  done
done

log "C dense"
CF="default,128x128:64:0:1,64x128:128:0:1,128x128:128:0:0,128x64:128:0:1,64x64:128:0:1"
./dr.sh python3 dense_gemm_harness.py default --recipe mxfp8 --shapes qkvz,ba,out,gu,down,attn_qkv --iters 10 --json /out/dense_default.json > $O/dense_default.log
for S in stock rm=0,sw=1; do
  SCHED=$([ $S = stock ] || echo $S) ./dr.sh python3 dense_gemm_harness.py sweep --recipe mxfp8 --shapes qkvz,ba,out,gu,down,attn_qkv --iters 5 --placements 6 --configs "$CF" --json /out/dense_sweep_$S.json > $O/dense_sweep_$S.log
done

log "D attention"
for C in 16384 32768; do
  ./dr.sh python3 attn_harness.py --context $C --kv-warps 1,2,4 --iters 50 > $O/attn_stock_c$C.txt
  PATCH=1 SWZ=1 ./dr.sh python3 attn_harness.py --context $C --kv-warps 2 --iters 50 > $O/attn_swz_c$C.txt
  PATCH=1 PIPE=1 ./dr.sh python3 attn_harness.py --context $C --kv-warps 2 --iters 50 > $O/attn_pipe_c$C.txt
  PATCH=1 PIPE=1 SWZ=1 ./dr.sh python3 attn_harness.py --context $C --kv-warps 2 --iters 50 > $O/attn_pipe_swz_c$C.txt
  PATCH=1 PIPE=1 SPLITQK=1 SWZ=1 ./dr.sh python3 attn_harness.py --context $C --kv-warps 2 --iters 50 > $O/attn_all_c$C.txt
done
PATCH=1 ./dr.sh python3 attn_harness.py --kv-warps 2 --iters 50 > $O/attn_patchedfile_defaults.txt

log "E ncu"
M=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_ldsm.sum,gpu__time_duration.sum
for V in stock swz; do
  E=(env); [ $V = swz ] && E=(env PATCH=1 SWZ=1)
  "${E[@]}" NCU=1 ./dr.sh /ncu/ncu --profile-from-start off --kernel-name regex:SelectedPositionPagedForward --launch-count 1 \
    --set detailed --section SourceCounters --section MemoryWorkloadAnalysis_Tables --metrics $M \
    --cache-control all --clock-control none -f -o /out/attn-$V-sm120 python3 attn_harness.py --kv-warps 2 --profile-one 2 --iters 3 --check-rows 4 > $O/ncu_$V.log
  NCU=1 ./dr.sh /ncu/ncu -i /out/attn-$V-sm120.ncu-rep --page details --csv > $O/attn-$V-sm120-details.csv
  NCU=1 ./dr.sh /ncu/ncu -i /out/attn-$V-sm120.ncu-rep --page raw --csv --metrics $M > $O/attn-$V-sm120-raw.csv
done
log "all steps finished"
