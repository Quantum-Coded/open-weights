#!/usr/bin/env bash
# Amendment 5 test runs: base then e2 on sealed/cube_100, greedy, loop detector OFF, 16,384 cap.
cd ~/wsp-r1
source .venv/bin/activate
log=logs_a5.txt
for arm in "base work/base" "e2 work/edit_r2"; do
  set -- $arm
  echo "START $1 cube_100 nokill $(date -u +%FT%T)" >> $log
  python a5_eval.py --model $2 --name $1 --set cube_100 --no-early-stop > logs_a5_$1_cube_100.txt 2>&1
  rc=$?
  echo "END $1 cube_100 nokill rc=$rc $(date -u +%FT%T)" >> $log
  [ $rc -eq 0 ] || { echo "A5 TEST FAILED $(date -u +%FT%T)" >> $log; exit 1; }
  grep -m1 -A 9 '^{' logs_a5_$1_cube_100.txt | tr '\n' ' ' >> $log; echo >> $log
done
echo "A5 TEST DONE $(date -u +%FT%T)" >> $log
