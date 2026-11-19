#!/usr/bin/env bash
# Usage: gemm_prof_run.sh  — restart Flash Next (final build) with the b12x GEMM timer, time one real-text
# 14.5K prefill per GEMM shape, write ~/ncu-side/ab/gemmprof-real.json, then restore the final build.
set -u
C=vllm_qwen38_next; D=~/.cache/vllm/nsys-flashnext; AB=~/ncu-side/ab
F=$(docker exec $C python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/model_executor/kernels/linear/nvfp4/b12x.py
stop_engine() {
  docker exec $C pkill -INT -f "bin/vllm serve" 2>/dev/null
  for i in $(seq 1 120); do sleep 5
    [ -z "$(docker exec $C pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && \
    [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && return 0
  done; echo "engine did not stop"; exit 4
}
ask() {  # $1 = content prefix, $2 = max chars of prompt
  python3 - "$1" "$2" <<'PY'
import json, sys, time, urllib.request, uuid
text = open("/home/spark/.cache/vllm/nsys-flashnext/qsa-dump/prompt.txt").read()[: int(sys.argv[2])]
body = {"model": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "max_tokens": 1, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": f"[{uuid.uuid4()}] {sys.argv[1]}\n\n" + text}]}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8112/v1/chat/completions",
    json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
print(f"prompt {r['usage']['prompt_tokens']} tokens, {time.time() - t:.2f} s")
PY
}
stop_engine
docker exec $C bash -c "[ -f $F.orig ] || cp $F $F.orig; cp $F.orig $F"
docker cp $AB/patch_gemm_prof.py $C:/tmp/patch_gemm_prof.py
docker exec $C python3 /tmp/patch_gemm_prof.py $F || exit 5
cp $D/extra.env $D/extra.env.final
printf "VLLM_B12X_GEMM_PROF=1\nVLLM_USE_BREAKABLE_CUDAGRAPH=0\n" >> $D/extra.env  # eager prefill so the Python timer sees each call
bash ~/ncu-side/start.sh patched || exit 2
ask "Summarize." 2000                       # warm-up
docker exec $C touch /root/.cache/vllm/nsys-flashnext/GEMMPROF_RESET
ask "Summarize the following documentation in one sentence." 1000000
docker exec $C touch /root/.cache/vllm/nsys-flashnext/GEMMPROF_DUMP
ask "Hi." 10                                # small request: triggers the dump
sleep 2; docker exec $C cat /root/.cache/vllm/nsys-flashnext/gemmprof.json > $AB/gemmprof-real.json
echo "wrote $AB/gemmprof-real.json ($(wc -c < $AB/gemmprof-real.json) bytes)"
# restore the final build
stop_engine
docker exec $C cp $F.orig $F
cp $D/extra.env.final $D/extra.env
bash ~/ncu-side/start.sh patched || exit 3
echo "final build restored $(date -u +%T) UTC"
