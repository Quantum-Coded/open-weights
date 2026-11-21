#!/usr/bin/env bash
# Game of 24 race demo (POST HOC, descriptive; RUNLOG 21 Nov). Puzzle by the run-2 rule, fixed before recording:
# among game24_300 greedy puzzles where base was stopped as a loop and e2 is correct, the shortest e2 answer
# -> g24-11-12-12-12 (e2 1,422 tokens). Waits for base math_fresh1200 to end (e2 restarts after), pauses the queue for ~15 min of single-stream
# recording, then relaunches the queue (ops/a4.sh skips finished runs).
set -u
cd ~/wsp-r1
until grep -q 'END base math_fresh1200 sampled1' logs_a4.txt; do
  tmux has-session -t a4 2>/dev/null || { echo "RACE24: queue gone before base math_fresh1200 ended $(date -Is)"; exit 1; }
  sleep 60
done
tmux kill-session -t a4; sleep 5; pkill -TERM -f "python eval.py"; pkill -TERM -f "python a4_eval.py"; sleep 10
echo "PAUSED for Game of 24 race recording $(date -Is)" >> logs_a4.txt
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_BATCH_INVARIANT=1
echo "RACE24 START $(date -Is)"
for A in "e2 work/edit_r2" "base work/base"; do
  set -- $A
  python record_race.py --model "$2" --arm "$1" --ref "results/${1}_game24_300_greedy.jsonl" \
      --set sealed/game24_300.jsonl --ids g24-11-12-12-12 --out results/race > "logs_race24_$1.txt" 2>&1
  echo "END race24 $1 rc=$? $(date -Is)"; grep -E "^(e2|base) g24" "logs_race24_$1.txt"
done
echo "RACE24 DONE $(date -Is)"
echo "RESUMED after race recording $(date -Is)" >> logs_a4.txt
tmux new -d -s a4 "bash -lc 'cd ~/wsp-r1; source .venv/bin/activate; bash ops/a4.sh >> logs_a4.txt 2>&1'"
