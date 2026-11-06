#!/usr/bin/env bash
# Phase 7: freeze run <R>'s patch and run the fast-track arms, then the follow-ups. Usage: phase7.sh <R>  (R = 1 or 2)
# Arm names for run 1: edit, rand_w, rand_s, edit_pp, edit_rerun; run 2 adds the suffix "_r2".
set -u
R=${1:?run number}
S=$([ "$R" = 1 ] && echo "" || echo "_r$R")
cd ~/wsp-r1 && source .venv/bin/activate
unset VLLM_BATCH_INVARIANT
P=editor/edits/patch.safetensors
[ -f "$P" ] || { echo "NO PATCH at $P"; exit 1; }
F=work/frozen_patch_run$R.safetensors
[ -e "$F" ] || cp -n "$P" "$F"
echo "FROZEN run$R $(sha256sum "$F") $(date -Is)"
cmp -s "$P" "$F" || { echo "patch changed after freeze copy"; exit 1; }
if [ -d ~/wsp_sealed/sealed ]; then mv ~/wsp_sealed/sealed ~/wsp_sealed/results . ; echo "test data restored"; fi
python apply_patch.py --patch "$F" --out work/edit$S
python make_random_control.py --patch "$F" --mode same-weights --out work/rand_w$S.safetensors
python make_random_control.py --patch "$F" --mode same-size    --out work/rand_s$S.safetensors
python apply_patch.py --patch work/rand_w$S.safetensors --out work/rand_w$S
python apply_patch.py --patch work/rand_s$S.safetensors --out work/rand_s$S
run() {   # run <name> <env> <args...>
  local name=$1 envv=$2; shift 2
  local out="results/${name}_$(echo "$@" | sed -E 's/.*--set ([^ ]+).*--regime ([^ ]+).*/\1_\2/').jsonl"
  if [ -e "$out" ]; then echo "SKIP $name: $out exists"; return 0; fi
  echo "START $name $(date -Is)"
  ( time env $envv python eval.py "$@" --name "$name" ) > "logs_${name}.txt" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -Is) $(grep real logs_${name}.txt | tail -1)"
  grep -A6 '^{' "logs_${name}.txt" | tr -s ' \n' ' '; echo
  return $rc
}
BI=VLLM_BATCH_INVARIANT=1
run edit$S       $BI --model work/edit$S   --set math500 --regime greedy --limit 200 || exit 1
run rand_w$S     $BI --model work/rand_w$S --set math500 --regime greedy --limit 200 || exit 1
run rand_s$S     $BI --model work/rand_s$S --set math500 --regime greedy --limit 200 || exit 1
run edit_pp$S    $BI --model work/edit$S   --set math500 --regime greedy --limit 200 --presence-penalty 1.5 || exit 1
run edit_rerun$S VLLM_BATCH_INVARIANT=0 --model work/edit$S --set math500 --regime greedy --limit 200 || exit 1
python analyze.py --base results/base_math500_greedy.jsonl --arm results/edit${S}_math500_greedy.jsonl \
  --arm results/rand_w${S}_math500_greedy.jsonl --arm results/rand_s${S}_math500_greedy.jsonl \
  --arm results/base_pp_math500_greedy.jsonl > logs_analyze_fasttrack$S.txt 2>&1
python analyze.py --base results/base_pp_math500_greedy.jsonl --arm results/edit_pp${S}_math500_greedy.jsonl > logs_analyze_pp$S.txt 2>&1
python analyze.py --base results/edit${S}_math500_greedy.jsonl --arm results/edit_rerun${S}_math500_greedy.jsonl > logs_analyze_edit_rerun$S.txt 2>&1
echo "FASTTRACK DONE run$R $(date -Is)"
# follow-ups: match every Phase 4 baseline run with the edited model
run edit$S       $BI --model work/edit$S --set gsm8k_test400 --regime greedy || exit 1
run edit$S       $BI --model work/edit$S --set mmlupro300 --regime greedy || exit 1
run edit_full$S  $BI --model work/edit$S --set math500 --regime greedy || exit 1
run edit$S       $BI --model work/edit$S --set math500 --regime sampled --limit 200 || exit 1
echo "FOLLOWUPS DONE run$R $(date -Is)"
