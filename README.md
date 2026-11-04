# wsp-r1: weight-space programming on DeepSeek-R1-Distill-Qwen-1.5B

Claude edits the model's weights directly (no gradients, no training) to fix its reasoning loops.
This kit gives Claude a rule-bound workspace and gives you a paired, pre-registered evaluation.
The full plan and methodology are in the doc that came with this kit.

```
wsp-r1/
  common.py            prompts, grading (math-verify), loop detector
  gen.py               chunked vLLM generation with early loop kill
  prepare_data.py      downloads data; writes practice sets + sealed test sets
  apply_patch.py       base + patch (x scale) -> runnable model folder in work/
  make_random_control.py  random edits matched to Claude's patch
  eval.py              runs one arm on one test set -> results/*.jsonl
  analyze.py           paired stats table (CIs, fixed/broken, McNemar, Holm)
  eval_plan.md         fill in and commit BEFORE editing
  editor/              <- Claude Code works ONLY in here
    CLAUDE.md          the editor's brief and rules
    fp.py              budget gate: the only way to run the model (50 passes)
    wsp.py             load/edit/save weights, lab notebook
    state.json         budget counter
```

## 0. Setup (Windows + WSL2)
```bash
# in WSL2 Ubuntu, with the NVIDIA driver installed on Windows
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git init && git add . && git commit -m "kit"
python prepare_data.py
python apply_patch.py --out work/base          # downloads the model (~3.5 GB)
```

## 1. Pilot and pre-register
```bash
# 20 practice problems, no early stop: how often does it loop, and is the detector right?
python eval.py --model work/base --name pilot --set practice:math_train --limit 20 --no-early-stop
```
Gate: if fewer than ~15% of answers loop, stop and pick another model (see the doc).
Otherwise fill in `eval_plan.md` (cap, detector thresholds, false-positive rate) and commit it.

## 2. Baselines (run overnight)
```bash
python eval.py --model work/base --name base    --set math500       --regime greedy
python eval.py --model work/base --name base    --set gsm8k_test400 --regime greedy
python eval.py --model work/base --name base    --set mmlupro300    --regime greedy
python eval.py --model work/base --name base_pp --set math500       --regime greedy --presence-penalty 1.5
python eval.py --model work/base --name base    --set math500       --regime sampled
```

## 3. Seal, then edit
Move `sealed/` and `results/` OUT of the project (e.g. to `~/wsp_sealed/`) so Claude cannot see them.
```bash
mv sealed results ~/wsp_sealed/
cd editor && claude          # open Claude Code here
```
Opening message, exactly: **"Make this model better at math by editing its weights directly. You may not train it."**
Don't steer. If a session stops on a usage limit, start a new one with "Continue from your notebook."

When Claude writes `FINAL_REPORT.md`, freeze:
```bash
sha256sum editor/edits/patch.safetensors >> eval_plan.md && git add -A && git commit -m "frozen patch"
mv ~/wsp_sealed/sealed ~/wsp_sealed/results .
```

## 4. Evaluate the arms
```bash
P=editor/edits/patch.safetensors
python apply_patch.py --patch $P --out work/edit
python make_random_control.py --patch $P --mode same-weights --out work/rand_w.safetensors
python make_random_control.py --patch $P --mode same-size    --out work/rand_s.safetensors
python apply_patch.py --patch work/rand_w.safetensors --out work/rand_w
python apply_patch.py --patch work/rand_s.safetensors --out work/rand_s

for arm in edit rand_w rand_s; do
  python eval.py --model work/$arm --name $arm --set math500 --regime greedy
done
python eval.py --model work/edit --name edit    --set gsm8k_test400 --regime greedy
python eval.py --model work/edit --name edit    --set mmlupro300    --regime greedy
python eval.py --model work/edit --name edit_pp --set math500       --regime greedy --presence-penalty 1.5
python eval.py --model work/edit --name edit    --set math500       --regime sampled
```

## 5. Analyse
```bash
python analyze.py --base results/base_math500_greedy.jsonl \
  --arm results/edit_math500_greedy.jsonl --arm results/rand_w_math500_greedy.jsonl \
  --arm results/rand_s_math500_greedy.jsonl --arm results/base_pp_math500_greedy.jsonl
python analyze.py --base results/base_pp_math500_greedy.jsonl --arm results/edit_pp_math500_greedy.jsonl
python analyze.py --base results/base_gsm8k_test400_greedy.jsonl --arm results/edit_gsm8k_test400_greedy.jsonl
python analyze.py --base results/base_mmlupro300_greedy.jsonl --arm results/edit_mmlupro300_greedy.jsonl
python analyze.py --base results/base_math500_sampled.jsonl --arm results/edit_math500_sampled.jsonl
```
Contamination check: `grep -rF` a few test questions (or their first 60 characters) across `editor/logs/`.
No-gradient check: `grep -rnE "backward\(|optim\.|requires_grad_?\(True" editor/` should find nothing
outside this README's own rule text.

## Notes
- If vLLM fails in WSL2, lower `gpu_util` in `gen.py` (0.85), or use llama.cpp with an f16 GGUF of `work/edit`.
  Never compare against a 4-bit quant: quantization changes the weights you edited.
- Slow? Copy the repo plus `work/base` and `work/edit` to a free Kaggle GPU and run step 4 there with the same commands.
- IFEval is not wired in: the model's `<think>` text breaks the rule checks unless stripped. If you add it,
  use lm-evaluation-harness and strip everything up to `</think>` before scoring; label it in the write-up.
