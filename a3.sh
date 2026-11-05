#!/usr/bin/env bash
# Amendment 3 pipeline. Waits for the Amendment 2 runner (fresh.sh) to finish; the GPU holds one job.
set -u
while pgrep -f "fresh.sh|phase7.sh" > /dev/null; do sleep 60; done
cd ~/wsp-r1 && source .venv/bin/activate
unset VLLM_BATCH_INVARIANT
echo "a3 start $(date -Is)"
run() {   # run <name> <model> <set> [extra args]
  local name=$1 model=$2 set=$3; shift 3
  local out="results/${name}_${set}_greedy.jsonl"
  if [ -e "$out" ]; then echo "SKIP $name $set"; return 0; fi
  echo "START $name $set $(date -Is)"
  ( time env VLLM_BATCH_INVARIANT=1 python eval.py --model "$model" --name "$name" --set "$set" --regime greedy "$@" ) > "logs_a3_${name}_${set}.txt" 2>&1
  local rc=$?
  echo "END $name $set rc=$rc $(date -Is) $(grep real logs_a3_${name}_${set}.txt | tail -1)"
  grep -A6 '^{' "logs_a3_${name}_${set}.txt" | tr -s ' \n' ' '; echo
  return $rc
}
run a3prac work/base a3_practice_mmlupro128 --max-tokens 8192 || exit 1
if [ ! -e work/a3/patch_e3.safetensors ]; then
  python a3_build.py --results results/a3prac_a3_practice_mmlupro128_greedy.jsonl --rows sealed/a3_practice_mmlupro128.jsonl --out work/a3 2>&1 | grep -v -E "Fetch|it/s" || exit 1
fi
echo "E3 PATCH sha256 $(sha256sum work/a3/patch_e3.safetensors)"
[ -e work/a3/rand_e3.safetensors ] || python make_random_control.py --patch work/a3/patch_e3.safetensors --mode same-weights --seed 0 --out work/a3/rand_e3.safetensors
[ -d work/e3 ] || python apply_patch.py --patch work/a3/patch_e3.safetensors --out work/e3
[ -d work/rand_e3 ] || python apply_patch.py --patch work/a3/rand_e3.safetensors --out work/rand_e3
S=mmlupro_fresh2_300
run base    work/base    $S || exit 1
run e3      work/e3      $S || exit 1
run rand_e3 work/rand_e3 $S || exit 1
run base_pp work/base    $S --presence-penalty 1.5 || exit 1
python analyze.py --base results/base_${S}_greedy.jsonl --arm results/e3_${S}_greedy.jsonl --arm results/rand_e3_${S}_greedy.jsonl --arm results/base_pp_${S}_greedy.jsonl > logs_analyze_a3.txt 2>&1
python analyze.py --base results/rand_e3_${S}_greedy.jsonl --arm results/e3_${S}_greedy.jsonl > logs_analyze_a3_e3_vs_rand.txt 2>&1
echo "A3 PRIMARY DONE $(date -Is)"; tail -4 logs_analyze_a3.txt; tail -1 logs_analyze_a3_e3_vs_rand.txt
run e3 work/e3 math500 --limit 200 || exit 1
python analyze.py --base results/base_math500_greedy.jsonl --arm results/e3_math500_greedy.jsonl > logs_analyze_a3_math.txt 2>&1
echo "A3 DONE $(date -Is)"; tail -1 logs_analyze_a3_math.txt
