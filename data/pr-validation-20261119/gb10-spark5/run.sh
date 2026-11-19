#!/bin/bash
cd ~/fi-pr; mkdir -p out
( while true; do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); [ "$a" -lt 3 ] && docker kill fi-pr-stock fi-pr-patched >/dev/null 2>&1 && echo "WATCHDOG KILL avail=${a}G" >> out/watchdog.txt; sleep 1; done ) & WD=$!; trap "kill $WD" EXIT
for V in stock patched; do
  for T in 16 64 256 1024 4096; do ./dr2.sh $V python3 mtp_gemm_harness.py --tokens $T --iters 20 --flush-mb 96 | grep -E "TFLOPS" | sed "s/^/T=$T $V /" >> out/mtp_$V.txt; done
  for C in 16384 32768; do ./dr2.sh $V python3 attn_harness.py --context $C --kv-warps 1,2,4 --iters 20 > out/attn_${V}_c$C.txt; done
  echo "== $V pytest mtp" > out/pytest_mtp_$V.txt; timeout 1800 ./dr2.sh $V python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/sequence/test_mtp_feedback.py >> out/pytest_mtp_$V.txt 2>&1
done
echo done > out/DONE
