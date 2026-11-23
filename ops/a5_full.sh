#!/usr/bin/env bash
# Amendment 5, the whole thing in one go (started by start_a5.ps1 on Windows, inside tmux session a5full):
#   1 base on cube_100   2 e2 on cube_100   (greedy, loop detector OFF, 16,384 cap; skipped if the result exists)
#   3 analysis (amendment5_primary.py)   4 race recording of the rule-picked puzzle (both arms, single stream)
#   5 video data (a5_video_data.py -> the video project's assets/cube-data.js)
#   6 report (RUNLOG, WRITEUP, POST_FACTS.md), commit and push
# Status lines go to logs_a5_full.txt; the last one is "A5 PIPELINE DONE" or "A5 PIPELINE FAILED at <stage>".
# Re-running resumes: finished runs and recordings are skipped. DRY=1 runs the same chain on the 12 PILOT puzzles
# with a 1,024-token cap under dry_* names, writes nothing to RUNLOG/WRITEUP and does not commit.
set -u
cd ~/wsp-r1
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_BATCH_INVARIANT=1
VIDEO="/mnt/c/Users/Dev Gaglani/Downloads/open-weight/videos/race-video"
OUTDIR="/mnt/c/Users/Dev Gaglani/Downloads/open-weight"
if [ "${DRY:-0}" = 1 ]; then
  P=dry_; SET=a5:cube_pilot12_v2; TAG=a5-cube_pilot12_v2; ROWS=a5_data/cube_pilot12_v2.jsonl; MAXT=1024
  RACE=results/race_dry; LOG=logs_a5_full_dry.txt; DATA="$VIDEO/assets/cube-data-dry.js"; FACTS="$OUTDIR/POST_FACTS_dry.md"
else
  P=; SET=cube_100; TAG=cube_100; ROWS=sealed/cube_100.jsonl; MAXT=16384
  RACE=results/race; LOG=logs_a5_full.txt; DATA="$VIDEO/assets/cube-data.js"; FACTS="$OUTDIR/POST_FACTS.md"
fi
PRIMARY=results/${P}a5_primary.json
say() { echo "$* $(date -u +%FT%T)" | tee -a $LOG; }
fail() { say "A5 PIPELINE FAILED at $1"; exit 1; }
push() {  # commit tracked text files only (results/ and logs are gitignored by design)
  [ "${DRY:-0}" = 1 ] && return 0
  git add RUNLOG.md WRITEUP.md amendment5_primary.py a5_video_data.py a5_report.py record_race.py ops/a5_full.sh \
      ops/a5_follow.sh ops/a5_launch.sh eval_plan_amendment5_addendum.md 2>/dev/null
  git commit -q -m "$1" && say "committed: $1" || say "nothing to commit: $1"
  git push -q origin HEAD 2>>$LOG && say "pushed" || say "PUSH FAILED (results are safe locally; push later)"
}

say "A5 PIPELINE START (DRY=${DRY:-0})"
for arm in "base work/base" "e2 work/edit_r2"; do
  set -- $arm
  out=results/${P}$1_${TAG}_greedy_nokill.jsonl
  if [ -f "$out" ]; then say "SKIP $1 run (exists: $out)"; continue; fi
  say "STAGE run_$1 START"
  python a5_eval.py --model $2 --name ${P}$1 --set $SET --no-early-stop --max-tokens $MAXT > logs_a5_${P}$1_${TAG}.txt 2>&1 \
    || fail "run_$1 (see logs_a5_${P}$1_${TAG}.txt)"
  say "STAGE run_$1 END: $(python -c "import json;s=json.load(open('$out.summary.json'));print(f\"solved {round(s['accuracy']*s['n'])}/{s['n']}, tokens {s['total_tokens']:,}, wall {s['wall_s']} s\")")"
done
[ "${DRY:-0}" = 1 ] || { say "GPU test runs done"; }

say "STAGE analysis START"
python amendment5_primary.py --base results/${P}base_${TAG}_greedy_nokill.jsonl --e2 results/${P}e2_${TAG}_greedy_nokill.jsonl \
  --out $PRIMARY > logs_a5_${P}primary.txt 2>&1 || fail "analysis (see logs_a5_${P}primary.txt)"
cat logs_a5_${P}primary.txt | tee -a $LOG
RID=$(python -c "import json;print(json.load(open('$PRIMARY'))['race_id'] or '')")

if [ -n "$RID" ]; then
  say "STAGE race START ($RID)"
  for arm in "e2 work/edit_r2" "base work/base"; do
    set -- $arm
    python record_race.py --model $2 --arm $1 --ref results/${P}$1_${TAG}_greedy_nokill.jsonl --set $ROWS --ids $RID \
      --out $RACE --max-tokens $MAXT > logs_a5_${P}race_$1.txt 2>&1 || fail "race_$1 (see logs_a5_${P}race_$1.txt)"
    grep -E "^$1 " logs_a5_${P}race_$1.txt | tee -a $LOG
  done
else
  say "STAGE race SKIPPED (the edit solved no cube; video falls back to the Game of 24 cut)"
fi
say "GPU FREE: the GPU is no longer needed by this pipeline"

say "STAGE video_data START"
python a5_video_data.py --primary $PRIMARY --race-dir $RACE --set $ROWS --out "$DATA" >> $LOG 2>&1 || fail video_data

say "STAGE report START"
FLAG=results/${P}a5_reported.flag
if [ "${DRY:-0}" = 1 ]; then
  python a5_report.py --primary $PRIMARY --race-dir $RACE --facts "$FACTS" --no-repo-edits >> $LOG 2>&1 || fail report
elif [ ! -f $FLAG ]; then
  python a5_report.py --primary $PRIMARY --race-dir $RACE --facts "$FACTS" \
    --video "renders/a5_final.mp4 in the race-video project" >> $LOG 2>&1 || fail report
  date -u +%FT%T > $FLAG
  push "Amendment 5 results: Rubik's cube (pipeline ops/a5_full.sh)"
else
  python a5_report.py --primary $PRIMARY --race-dir $RACE --facts "$FACTS" --no-repo-edits >> $LOG 2>&1 || fail report
  say "report already in RUNLOG/WRITEUP (flag $FLAG); facts file refreshed"
fi
say "A5 PIPELINE DONE"
