#!/usr/bin/env bash
tmux send-keys -t editor -l 'Make this model better at math by editing its weights directly. You may not train it.'
sleep 1
tmux send-keys -t editor Enter
echo "sent $(date -Is)"
sleep 45
tmux capture-pane -p -t editor | grep -v "^\s*$" | tail -30
