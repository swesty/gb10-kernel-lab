#!/bin/bash
cd ~/fi-pr; mkdir -p out
for V in stock patched; do
  echo "== $V pytest mtp" > out/pytest_mtp_$V.txt; timeout 1800 ./dr2.sh $V python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/sequence/test_mtp_feedback.py >> out/pytest_mtp_$V.txt 2>&1
  echo "== $V pytest qsa" > out/pytest_qsa_$V.txt; timeout 3600 ./dr2.sh $V python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/attention/test_qsa_sparse_gqa.py >> out/pytest_qsa_$V.txt 2>&1
  for T in 16 64 256 1024 4096; do ./dr2.sh $V python3 mtp_gemm_harness.py --tokens $T --iters 20 --flush-mb 512 | grep -E "TFLOPS" | sed "s/^/T=$T $V /" >> out/mtp_$V.txt; done
  for C in 16384 32768; do ./dr2.sh $V python3 attn_harness.py --context $C --kv-warps 1,2,4 --iters 50 > out/attn_${V}_c$C.txt; done
done
echo done > out/DONE
