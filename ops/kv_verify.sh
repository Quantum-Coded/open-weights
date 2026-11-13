#!/usr/bin/env bash
# Hold the math_fresh1200 runs (a4_eval.py waits while HOLD_A4 exists), check KV-grouped generation
# on the free GPU, switch it on only if every token matches, then release the queue either way.
cd ~/wsp-r1 || exit 1
source .venv/bin/activate
touch HOLD_A4
echo "KVV armed $(date -Is)"
until grep -q 'FT DONE' logs_a4.txt; do
  tmux has-session -t a4 2>/dev/null || { echo "KVV: queue gone before FT DONE $(date -Is)"; exit 1; }
  sleep 60
done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -lt 800 ]; do sleep 20; done
echo "KVV test start $(date -Is)"
python kv_group_test.py > logs_kv_group_test.txt 2>&1; rc=$?
if [ $rc -eq 0 ]; then
  touch kv_group.on; echo "KV GROUP ON $(date -Is)"
else
  echo "KV GROUP TEST FAILED rc=$rc, old path kept $(date -Is)"
fi
tail -n 6 logs_kv_group_test.txt
rm -f HOLD_A4
echo "KVV released $(date -Is)"
