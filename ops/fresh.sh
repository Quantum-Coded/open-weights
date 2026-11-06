#!/usr/bin/env bash
# Amendment 2 arms on the fresh sets. Waits for phase7.sh 1 to finish (the GPU holds one job).
set -u
while pgrep -f "phase7.sh 1" > /dev/null; do sleep 60; done
cd ~/wsp-r1 && source .venv/bin/activate
unset VLLM_BATCH_INVARIANT
echo "fresh start $(date -Is)"
[ -d work/e1x1.5 ] || python apply_patch.py --patch work/frozen_patch_run1.safetensors --scale 1.5 --out work/e1x1.5
run() {   # run <name> <model> <set>
  local out="results/$1_$3_greedy.jsonl"
  if [ -e "$out" ]; then echo "SKIP $1 $3"; return 0; fi
  echo "START $1 $3 $(date -Is)"
  ( time env VLLM_BATCH_INVARIANT=1 python eval.py --model "$2" --name "$1" --set "$3" --regime greedy ) > "logs_fresh_$1_$3.txt" 2>&1
  local rc=$?
  echo "END $1 $3 rc=$rc $(date -Is) $(grep real logs_fresh_$1_$3.txt | tail -1)"
  grep -A6 '^{' "logs_fresh_$1_$3.txt" | tr -s ' \n' ' '; echo
  return $rc
}
for S in gsm8k_fresh400 mmlupro_fresh300; do
  run base   work/base   $S || exit 1
  run e1     work/edit   $S || exit 1
  run e1x1.5 work/e1x1.5 $S || exit 1
  run rand_w work/rand_w $S || exit 1
  python analyze.py --base results/base_${S}_greedy.jsonl --arm results/e1_${S}_greedy.jsonl --arm results/e1x1.5_${S}_greedy.jsonl > logs_analyze_fresh_$S.txt 2>&1
  python analyze.py --base results/rand_w_${S}_greedy.jsonl --arm results/e1_${S}_greedy.jsonl > logs_analyze_fresh_e1_vs_randw_$S.txt 2>&1
  echo "FRESH DONE $S $(date -Is)"; tail -3 logs_analyze_fresh_$S.txt
done
