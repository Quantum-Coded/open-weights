#!/usr/bin/env bash
# Prepare editing run 2 in a clean copy (~/wsp-r2) with only the editor's tools and practice data. Run only when no GPU job is running.
set -eu
if pgrep -f "eval.py|phase7.sh|fresh.sh" > /dev/null; then echo "GPU job still running; abort"; exit 1; fi
cd ~/wsp-r1
# 1. seal test data and per-problem results again
[ -e ~/wsp_sealed/sealed ] && { echo "~/wsp_sealed/sealed already exists; abort"; exit 1; }
mv sealed results ~/wsp_sealed/ && echo "sealed: test data and results moved to ~/wsp_sealed"
# 2. clean copy, files taken from the handoff commit 3d530f4 (identical to what run 1 started with)
R=~/wsp-r2
[ -e $R ] && { echo "$R exists; abort"; exit 1; }
mkdir -p $R/editor/data/practice $R/editor/logs $R/editor/edits $R/work
for f in common.py gen.py apply_patch.py editor/CLAUDE.md editor/fp.py editor/wsp.py; do git show 3d530f4:$f > $R/$f; done
cp editor/data/practice/math_train.jsonl editor/data/practice/gsm8k_train.jsonl $R/editor/data/practice/
touch $R/editor/logs/.keep $R/editor/edits/.keep
echo '{"budget": 50, "used": 0, "passes": []}' > $R/editor/state.json
ln -s ~/wsp-r1/.venv $R/.venv
# 3. checks
for f in math_train gsm8k_train; do echo "$f sha256 $(sha256sum $R/editor/data/practice/$f.jsonl | cut -c1-16) (Phase 5: math_train 5cd02994a542a854, gsm8k_train 28cb0e9243ac55bc)"; done
git show 3d530f4:editor/state.json; echo
find $R -not -path "*/data/practice/*" | sort
ls -a ~/wsp-r1 | grep -E "^(sealed|results)$" && echo "WARNING: test data still in repo" || echo "repo holds no sealed/ or results/"
