set -e
cd /home/dev_gaglani/wsp-r1/editor
test ! -f edits/patch.safetensors
for src in math_train gsm8k_train; do ../.venv/bin/python fp.py generate --source $src --n 32 --seed 41 2>&1 | grep -E "^pass|model:|Error|Traceback"; done
cp work/patch_H4_a05.safetensors edits/patch.safetensors
for src in math_train gsm8k_train; do ../.venv/bin/python fp.py generate --source $src --n 32 --seed 41 2>&1 | grep -E "^pass|model:|Error|Traceback"; done
