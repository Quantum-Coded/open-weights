#!/usr/bin/env bash
# Amendment 4 GPU queue, in the pre-registered order. Skips runs whose result file exists. Run in tmux.
set -u
cd ~/wsp-r1 && source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_BATCH_INVARIANT=1
echo "A4 START $(date -Is)"
a4() {   # a4 <name> <model> <set> <regime>
  local out="results/$1_$3_$4.jsonl"
  if [ -e "$out" ]; then echo "SKIP $1 $3 $4"; return 0; fi
  echo "START $1 $3 $4 $(date -Is)"
  ( time python a4_eval.py --model "$2" --name "$1" --set "$3" --regime "$4" ) > "logs_a4_$1_$3_$4.txt" 2>&1
  local rc=$?
  echo "END $1 $3 $4 rc=$rc $(date -Is)"; grep -A9 '^{' "logs_a4_$1_$3_$4.txt" | tr -s ' \n' ' '; echo
  return $rc
}
ev() {   # ev <name> <model> <set> <regime> [extra eval.py args]
  local out="results/$1_$3_$4.jsonl"
  if [ -e "$out" ]; then echo "SKIP $1 $3 $4"; return 0; fi
  echo "START $1 $3 $4 $(date -Is)"
  ( time python eval.py --model "$2" --name "$1" --set "$3" --regime "$4" "${@:5}" ) > "logs_a4_$1_$3_$4.txt" 2>&1
  local rc=$?
  echo "END $1 $3 $4 rc=$rc $(date -Is)"; grep -A7 '^{' "logs_a4_$1_$3_$4.txt" | tr -s ' \n' ' '; echo
  return $rc
}
# 1. Game of 24
a4 base work/base    game24_300 greedy   || exit 1
a4 e2   work/edit_r2 game24_300 greedy   || exit 1
a4 base work/base    game24_300 sampled1 || exit 1
a4 e2   work/edit_r2 game24_300 sampled1 || exit 1
python amendment4_primary.py > logs_a4_primary.txt 2>&1; echo "GAME24 DONE $(date -Is)"
# 2. Fine-tune
if [ ! -e a4_data/ft_train.jsonl ]; then
  echo "START ft_data $(date -Is)"; ( time python a4_ft_data.py ) > logs_a4_ft_data.txt 2>&1 || exit 1
  echo "END ft_data $(date -Is)"; tail -n 12 logs_a4_ft_data.txt
fi
if [ ! -d work/ft ]; then
  echo "START ft_train $(date -Is)"; ( time python a4_ft_train.py ) > logs_a4_ft_train.txt 2>&1 || { echo "ft_train FAILED"; tail -n 30 logs_a4_ft_train.txt; exit 1; }
  echo "END ft_train $(date -Is)"; tail -n 4 logs_a4_ft_train.txt
fi
ev ft work/ft math500          greedy --limit 200 || exit 1
ev ft work/ft gsm8k_fresh400   greedy             || exit 1
ev ft work/ft mmlupro_fresh300 greedy             || exit 1
a4 ft work/ft game24_300       greedy             || exit 1
python amendment4_primary.py > logs_a4_primary.txt 2>&1; echo "FT DONE $(date -Is)"
# 3. Powered sampling test
a4 base work/base    math_fresh1200 sampled1 || exit 1
a4 e2   work/edit_r2 math_fresh1200 sampled1 || exit 1
python amendment4_primary.py > logs_a4_primary.txt 2>&1; echo "MATHFRESH DONE $(date -Is)"
# 4. Secondary
ev ft work/ft math500 sampled --limit 200 || exit 1
python amendment4_primary.py > logs_a4_primary.txt 2>&1
echo "A4 DONE $(date -Is)"
