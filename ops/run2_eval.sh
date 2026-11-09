#!/usr/bin/env bash
# After editing run 2 (9 Nov): (1) phase7 arms for run 2 (suffix _r2); (2) e2 on the fresh sets (Amendment 2);
# (3) post hoc race-demo runs without the loop killer (RUNLOG 9 Nov, descriptive only); (4) lm-eval --limit 400.
# Offline-safe. The GPU holds one job, so everything runs in sequence.
set -u
cd ~/wsp-r1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
P=$HOME/wsp-r2/editor/edits/patch.safetensors bash ops/phase7.sh 2 || exit 1
source .venv/bin/activate; unset VLLM_BATCH_INVARIANT
run() {   # run <name> <model> <set> [extra args]
  local name=$1 model=$2 set=$3; shift 3
  local out="results/${name}_${set}_greedy.jsonl"
  if [ -e "$out" ]; then echo "SKIP $name $set"; return 0; fi
  echo "START $name $set $(date -Is)"
  ( time env VLLM_BATCH_INVARIANT=1 python eval.py --model "$model" --name "$name" --set "$set" --regime greedy "$@" ) > "logs_fresh_${name}_${set}.txt" 2>&1
  local rc=$?
  echo "END $name $set rc=$rc $(date -Is) $(grep real logs_fresh_${name}_${set}.txt | tail -1)"
  grep -A6 '^{' "logs_fresh_${name}_${set}.txt" | tr -s ' \n' ' '; echo
  return $rc
}
for S in gsm8k_fresh400 mmlupro_fresh300; do
  run e2 work/edit_r2 $S || exit 1
  python analyze.py --base results/base_${S}_greedy.jsonl --arm results/e2_${S}_greedy.jsonl > logs_analyze_fresh_e2_$S.txt 2>&1
done
echo "E2 FRESH DONE $(date -Is)"
for A in "base work/base" "e1 work/edit" "e2 work/edit_r2"; do
  set -- $A; run ${1}_nokill $2 gsm8k_fresh400 --no-early-stop || exit 1
done
echo "NOKILL DONE $(date -Is)"
bash ops/lmeval_gsm8k.sh
echo "RUN2_EVAL DONE $(date -Is)"
