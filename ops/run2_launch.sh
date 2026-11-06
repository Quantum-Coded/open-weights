#!/usr/bin/env bash
# Start the run-2 editor session in tmux (same flags as run 1). The prompt is sent separately with send_prompt.sh.
set -eu
tmux has-session -t editor 2>/dev/null && { echo "editor session already exists"; exit 1; }
tmux new-session -d -s editor -x 220 -y 50 -c /home/dev_gaglani/wsp-r2/editor
tmux send-keys -t editor "cd ~/wsp-r2/editor && ~/.local/bin/claude --model claude-opus-5-5 --permission-mode auto" Enter
sleep 10
tmux capture-pane -p -t editor | grep -v "^\s*$" | tail -25
