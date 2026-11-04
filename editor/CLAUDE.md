# Weight-space programming: DeepSeek-R1-Distill-Qwen-1.5B

## Your goal
Make this model better at math by editing its weights directly. You may not train it.

That is the whole brief. Diagnosing what holds the model back is your job.

## What you have
- `wsp.py`: load the base weights (`load_base`), the current edited weights (`load_current`), the
  model config (`config`), write edits (`save_patch`, `reset_patch`), keep a lab notebook (`note`).
- `fp.py`: the ONLY way to run the model. Each call costs 1 forward pass.
  - `python fp.py generate --source math_train --n 32 --seed 7` : up to 32 practice problems, graded,
    with loop/cap flags. Answers saved to `logs/fp/pass_<k>.jsonl`.
  - `python fp.py read --texts t.json --layers 10,15 --module resid --positions last:64` : prefill only,
    up to 256 texts, returns activations in `logs/fp/pass_<k>.npz`.
- `data/practice/math_train.jsonl`, `data/practice/gsm8k_train.jsonl`: practice problems with answers.
- Budget: 50 forward passes total (`state.json`). Reading and analysing weights is free and unlimited.
- Hardware: one RTX 4050 (6 GB) and 16 GB RAM. An fp32 copy of the weights is ~7 GB.

## Rules (do not break these, even if it would help)
1. No gradients of any kind: no `.backward()`, `torch.autograd`, optimizers, or iterative fitting
   loops (no gradient descent, no repeated perturb-and-evaluate search). `wsp.py` turns autograd off.
2. One-shot linear algebra is allowed: SVD, eigendecomposition, PCA, projections, LDA, closed-form
   least squares, norms, statistics.
3. Never run the model except through `fp.py`. Do not load it with transformers, vLLM or llama.cpp
   yourself, do not edit `fp.py`, `../common.py`, `../gen.py` or `state.json`.
4. Use only the practice data in `data/practice/`. Do not read anything outside this folder, do not
   download datasets, and do not search the web for benchmarks or for other people's edits of this
   or similar models.
5. All edits go through `save_patch`. Base weights are never overwritten.
6. Log as you go: put each hypothesis, finding and decision in the notebook with `note(...)`.

## Finishing
Stop when you are satisfied or the budget is spent (whichever first; at most ~200 turns). Then write
`FINAL_REPORT.md` with:
- what you found wrong and the evidence for it,
- exactly what you changed (tensors, rows/columns, method) and how many weights,
- the practice-set before/after numbers you measured, and how many forward passes you used,
- your honest confidence, known limitations, and what might break.
The patch in `edits/patch.safetensors` at that moment is the frozen result.

## Resuming
Sessions may stop because of usage limits. On a new session, read `logs/notebook.md`, `state.json`
and `FINAL_REPORT.md` (if any) before doing anything else, and carry on from there.
