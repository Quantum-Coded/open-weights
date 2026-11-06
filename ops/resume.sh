#!/usr/bin/env bash
# Resume after the 6 Nov pause: Amendment 2 (fresh sets), then Amendment 3. One GPU job at a time.
cd ~/wsp-r1
echo "resume start $(date -Is)"
bash ops/fresh.sh || { echo "fresh.sh failed rc=$?"; exit 1; }
bash a3.sh
echo "resume end $(date -Is)"
