#!/usr/bin/env bash
# Called by start_a5.ps1: prints progress until the pipeline ends. Exit 0 = done, 1 = failed.
# Keeping this process open also keeps WSL running. usage: bash ops/a5_follow.sh [dry]
cd ~/wsp-r1
LOG=logs_a5_full.txt; S=a5full; P=
[ "${1:-}" = dry ] && { LOG=logs_a5_full_dry.txt; S=a5dry; P=dry_; }
sleep 10   # let the pipeline write its START line
seen=$(grep -n "A5 PIPELINE START" $LOG 2>/dev/null | tail -1 | cut -d: -f1)
seen=${seen:-1}; n=$((seen - 1)); k=0
while true; do
  total=$(wc -l < $LOG 2>/dev/null || echo 0)
  if [ "$total" -gt "$n" ]; then sed -n "$((n + 1)),${total}p" $LOG | sed "s/^/[$(date +%H:%M)] /"; n=$total; fi
  last=$(tail -1 $LOG 2>/dev/null)
  case "$last" in *"A5 PIPELINE DONE"*) exit 0;; *"A5 PIPELINE FAILED"*) exit 1;; esac
  if ! tmux has-session -t $S 2>/dev/null; then echo "[$(date +%H:%M)] tmux session $S ended without DONE/FAILED"; exit 1; fi
  k=$((k + 1))
  if [ $((k % 10)) -eq 1 ]; then   # every ~10 min: live progress of the newest run log
    f=$(ls -t logs_a5_${P}*.txt 2>/dev/null | grep -v -e full -e primary | head -1)
    [ -n "$f" ] && echo "[$(date +%H:%M)]   $f: $(tail -c 600 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-140)"
  fi
  sleep 60
done
