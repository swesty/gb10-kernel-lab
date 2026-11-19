#!/bin/bash
cd ~/fi-pr
for V in stock patched; do
  echo "== $V pytest mtp" > out/pytest_mtp_$V.txt; timeout 1800 ./dr2.sh $V python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/sequence/test_mtp_feedback.py >> out/pytest_mtp_$V.txt 2>&1
  echo "== $V pytest qsa" > out/pytest_qsa_$V.txt; timeout 3600 ./dr2.sh $V python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/attention/test_qsa_sparse_gqa.py >> out/pytest_qsa_$V.txt 2>&1
done
echo done > out/PYTEST_DONE
