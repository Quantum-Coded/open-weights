#!/usr/bin/env bash
# Race-demo recordings (POST HOC, descriptive). Examples by the rules fixed before looking (RUNLOG):
# gsm-794 (run 1 rule) and gsm-665 (run 2 rule). Waits for the GPU queue (run2_eval.sh / lm-eval) to finish.
set -u
cd ~/wsp-r1
while pgrep -f "run2_eval.sh|lm_eval" > /dev/null; do sleep 60; done
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_BATCH_INVARIANT=1
echo "RACE START $(date -Is)"
for A in "e2 work/edit_r2" "e1 work/edit" "base work/base"; do
  set -- $A
  python record_race.py --model "$2" --arm "$1" --ref "results/${1}_nokill_gsm8k_fresh400_greedy.jsonl" \
      --ids gsm-665,gsm-794 --out results/race > "logs_race_$1.txt" 2>&1
  echo "END race $1 rc=$? $(date -Is)"; grep -E "^(e1|e2|base) gsm" "logs_race_$1.txt"
done
echo "RACE DONE $(date -Is)"
