#!/bin/bash
cd ~/fi-pr
( while true; do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); [ "$a" -lt 3 ] && docker kill fi-pr-stock fi-pr-patched >/dev/null 2>&1 && echo "WATCHDOG KILL avail=${a}G" >> out/watchdog.txt; sleep 1; done ) & WD=$!; trap "kill $WD" EXIT
for V in stock patched; do
  echo "== $V pytest mtp" > out/pytest_mtp_$V.txt
  timeout 1800 ./dr2.sh $V bash -c "pip install -q pytest >/dev/null 2>&1; python3 -m pytest -q -p no:cacheprovider tests/experimental/b12x/sequence/test_mtp_feedback.py" >> out/pytest_mtp_$V.txt 2>&1
done
echo done > out/PYTEST_DONE
