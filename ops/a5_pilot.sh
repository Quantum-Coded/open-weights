#!/usr/bin/env bash
# Amendment 5 step 1: wait for the Amendment 4 queue to finish, then run the base-only cube pilot.
cd ~/wsp-r1
source .venv/bin/activate
log=logs_a5.txt
echo "A5 PILOT WAITING $(date -u +%FT%T)" >> $log
while ! grep -q 'A4 DONE' logs_a4.txt; do
  if ! tmux has-session -t a4 2>/dev/null; then echo "A4 QUEUE GONE WITHOUT 'A4 DONE'; pilot not started $(date -u +%FT%T)" >> $log; exit 1; fi
  sleep 60
done
sleep 30
echo "START pilot $(date -u +%FT%T)" >> $log
python a5_eval.py --model work/base --name pilot --set a5:cube_pilot12 > logs_a5_pilot.txt 2>&1
echo "END pilot rc=$? $(date -u +%FT%T)" >> $log
echo "A5 PILOT DONE $(date -u +%FT%T)" >> $log
