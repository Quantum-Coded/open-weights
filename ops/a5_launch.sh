#!/usr/bin/env bash
# Called by start_a5.ps1: starts ops/a5_full.sh in tmux session a5full (or a5dry), unless it is already running.
# usage: bash ops/a5_launch.sh [dry]
cd ~/wsp-r1
S=a5full; D=0
[ "${1:-}" = dry ] && { S=a5dry; D=1; }
if tmux has-session -t $S 2>/dev/null; then echo "already running in tmux session $S"; exit 0; fi
tmux new-session -d -s $S "DRY=$D bash ops/a5_full.sh"
echo "started tmux session $S"
