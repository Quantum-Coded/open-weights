# Wait until the GPU is free (< 800 MiB used on two checks 2 min apart), then run the A4 queue.
cd ~/wsp-r1
echo "GATE waiting $(date -Is)"
free=0
while [ $free -lt 2 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${m:-9999}" -lt 800 ]; then free=$((free+1)); else free=0; fi
  sleep 120
done
echo "GATE open $(date -Is)"
bash ops/a4.sh
