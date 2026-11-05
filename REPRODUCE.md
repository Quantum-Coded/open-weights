# Reproducing the results

Everything here ran on one laptop: an RTX 4050 (6 GB) under WSL2 Ubuntu 24.04, with 11 GiB of RAM visible
to WSL. Expect about 30 minutes per 400 GSM8K problems and about 65–85 minutes per 200 MATH-500 problems
(greedy, 16,384-token cap).

## 1. Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # exact versions: torch 2.10.0+cu128, vllm 0.19.1, ...
```

The model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` in bf16. The Hugging Face revision is recorded in
`RUNLOG.md` (Phase 0).

## 2. Data

```bash
python prepare_data.py            # sealed/ test sets + editor/data/practice/ (seed 1234)
python decontaminate_practice.py  # drops practice problems that nearly duplicate a test question
```

`prepare_data.py` drops only exact MATH-500 duplicates. `decontaminate_practice.py` applies the stricter
rule used before the editing run: a practice problem is dropped if it shares its first 40 normalized
characters with any sealed test question *and* either has `difflib` similarity ≥ 0.80 or shares the first
60 characters. That removed 82 problems: 7,416 MATH-train and 7,473 GSM8K-train problems remain. The
sha256 prefixes of the resulting files are math_train `5cd02994a542a854` and gsm8k_train `28cb0e9243ac55bc`.

Fresh sets for Amendment 2:

```bash
python make_fresh.py     # sealed/gsm8k_fresh400.jsonl, sealed/mmlupro_fresh300.jsonl (sha256 in eval_plan_amendment2.md)
```

## 3. Models

```bash
python apply_patch.py --out work/base
python apply_patch.py --patch editor/edits/patch.safetensors --out work/edit          # run 1, sha256 4c8a5c77...
python make_random_control.py --patch editor/edits/patch.safetensors --mode same-weights --out work/rand_w.safetensors
python make_random_control.py --patch editor/edits/patch.safetensors --mode same-size    --out work/rand_s.safetensors
python apply_patch.py --patch work/rand_w.safetensors --out work/rand_w
python apply_patch.py --patch work/rand_s.safetensors --out work/rand_s
python apply_patch.py --patch editor/edits/patch.safetensors --scale 1.5 --out work/e1x1.5
```

## 4. Evaluation

Every arm uses batch-invariant kernels, so reruns are byte-identical. The noise-floor reruns use normal
kernels.

```bash
export VLLM_BATCH_INVARIANT=1
python eval.py --model work/base   --name base    --set math500 --regime greedy --limit 200
python eval.py --model work/edit   --name edit    --set math500 --regime greedy --limit 200
python eval.py --model work/rand_w --name rand_w  --set math500 --regime greedy --limit 200
python eval.py --model work/rand_s --name rand_s  --set math500 --regime greedy --limit 200
python eval.py --model work/base   --name base_pp --set math500 --regime greedy --limit 200 --presence-penalty 1.5
python eval.py --model work/edit   --name edit_pp --set math500 --regime greedy --limit 200 --presence-penalty 1.5
python eval.py --model work/edit   --name edit    --set gsm8k_test400 --regime greedy
python eval.py --model work/edit   --name edit    --set mmlupro300    --regime greedy
# noise floor: normal kernels, problem order shuffled
VLLM_BATCH_INVARIANT=0 python eval.py --model work/base --name base_rerun  --set math500 --regime greedy --limit 200
VLLM_BATCH_INVARIANT=0 python eval.py --model work/base --name base_rerun2 --set math500 --regime greedy --limit 200 --order-seed 2   # also 3, 4, 5
VLLM_BATCH_INVARIANT=0 python eval.py --model work/edit --name edit_rerun  --set math500 --regime greedy --limit 200
```

The base runs on GSM8K, MMLU-Pro, the full MATH-500 set and the sampled regime follow the same pattern. The
exact commands are in `RUNLOG.md`.

## 5. Analysis

```bash
python analyze.py --base results/base_math500_greedy.jsonl --arm results/edit_math500_greedy.jsonl \
  --arm results/rand_w_math500_greedy.jsonl --arm results/rand_s_math500_greedy.jsonl --arm results/base_pp_math500_greedy.jsonl
python analyze.py --base results/base_pp_math500_greedy.jsonl --arm results/edit_pp_math500_greedy.jsonl
python primary_run1.py          # the pre-registered primary family, Holm over 4
python phase8_loopflips.py      # loops removed / added per arm
python figures.py --base results/base_math500_greedy.jsonl --arm "Claude's edit=results/edit_math500_greedy.jsonl" ...  # see figures.py docstring
```

## 6. The editing run

The editor's own setup is in `editor/`: `CLAUDE.md` holds its rules, `fp.py` its budgeted model runs and
`wsp.py` its weight helpers.
- **Run 1's record:** `editor/FINAL_REPORT.md`, `editor/logs/notebook.md` and the scripts in `editor/work/`.
  It ran as Claude Opus 5.5 in Claude Code, in auto permission mode, started in `editor/`, and its only
  message was "Make this model better at math by editing its weights directly. You may not train it."
- **Run 2:** it ran in a clean copy of `editor/` and its helper modules, outside this repo, so it could
  not read run 1's notes or any results. See `RUNLOG.md`.

## Known differences you may see

- Normal (non-batch-invariant) kernels give 2–4.5 points higher MATH-500 accuracy for the base model on
  this GPU, and 29–36 grade flips per rerun. Compare arms only under the same kernels.
- A different GPU will give different greedy text. Expect the per-problem flips, not the direction of
  the effects, to change.
