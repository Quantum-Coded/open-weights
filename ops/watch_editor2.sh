#!/usr/bin/env bash
# run 2: E=~/wsp-r2/editor bash watch_editor2.sh
# Read-only watcher for the editor tmux session. Never types into it. Exits with a reason line on:
# FINAL_REPORT.md exists, a usage-limit message, the claude process gone, long idle with nothing running,
# or the report deadline (19:30 IST = 14:00 UTC).
E=${E:-~/wsp-r1/editor}
DEADLINE=${DEADLINE:-$(date -d "2026-11-07T00:00:00Z" +%s)}
idle=0
while true; do
  now=$(date +%s)
  pane=$(tmux capture-pane -p -t editor 2>/dev/null)
  used=$(python3 -c "import json;print(json.load(open('$E/state.json'))['used'])" 2>/dev/null)
  if [ -f $E/FINAL_REPORT.md ]; then echo "EVENT final_report $(date -Is) used=$used"; break; fi
  if echo "$pane" | tail -15 | grep -qiE "usage limit|limit reached|limit will reset|resets [0-9]|out of extra usage"; then
    echo "EVENT usage_limit $(date -Is) used=$used"; echo "$pane" | grep -v "^\s*$" | tail -8; break; fi
  if ! pgrep -f "claude --model claude-opus-5-5" > /dev/null; then echo "EVENT claude_exited $(date -Is) used=$used"; break; fi
  if echo "$pane" | tail -4 | grep -q "esc to interrupt" || pgrep -f "fp.py|python" > /dev/null; then idle=0; else idle=$((idle+1)); fi
  if [ $idle -ge 20 ]; then echo "EVENT idle_20min $(date -Is) used=$used"; echo "$pane" | grep -v "^\s*$" | tail -12; break; fi
  if [ $now -ge $DEADLINE ]; then echo "EVENT deadline $(date -Is) used=$used"; break; fi
  sleep 60
done
