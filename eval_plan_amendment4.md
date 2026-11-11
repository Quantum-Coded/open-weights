# Amendment 4: a new application, sampling with enough power, and a fine-tuning head-to-head

- Written by the operator on 2026-11-11 after Dev approved three experiments ("Yes, start all three").
- Committed **before any model runs on the new test sets** and before the fine-tuned model exists.
- Run 2's edit (`work/edit_r2`, frozen patch sha256 in RUNLOG) is used unchanged. Nobody edits weights for this
  amendment; the editor is not involved and gets no message.
- Status: confirmatory for the three primary tests and the head-to-head rules below. Everything else is
  secondary. It never replaces or reinterprets `eval_plan.md` or Amendments 1–3.

## Why

1. **Untouched application.** Every test so far is maths or multiple choice, close to the editor's practice data.
   The Game of 24 is a search puzzle nobody edited for.
2. **Sampling.** On MATH-500/200 at T 0.6 the edit gave 82.0% → 85.25%, which was not significant with
   2 samples × 200 problems. That pilot was underpowered, so this is a new test on new problems, sized in
   advance. The old result is not re-analysed.
3. **Fine-tuning.** Is a closed-form weight edit a real alternative to the standard way of fixing a behaviour,
   a fine-tune? Compare both on the same tests, the fine-tune trained on the same practice pools the editor had.

## Data (`a4_make_data.py`; sha256 recorded in RUNLOG when built)

- **`game24_300`**: all 1,362 solvable multisets of four numbers 1–13 (exhaustive exact-fraction solver),
  shuffled with `random.Random(20261111)`. Puzzles 0–19 are the base-only pilot (`a4_data/game24_pilot20`);
  puzzles 20–319 are the sealed test set.
- **`math_fresh1200`**: MATH test problems (all 7 subjects, 5,000) that are neither MATH-500 problems nor a match
  to an editor practice problem (Phase 5 rule). Shuffled with `random.Random(20261112)`, the first 1,200 kept.
  Answer: the last `\boxed{}` of the reference solution. Graded by the same math-verify grader as MATH-500.
- **`ft_problems1200`**: 600 `math_train` + 600 `gsm8k_train` practice problems, `random.Random(20261113)`.
  Fine-tuning data only.

## Game of 24 grading (`game24.py`)

- Prompt: "Game of 24. Use the four numbers a, b, c, d exactly once each, together with +, -, ×, ÷ and
  parentheses, to make an expression equal to 24. Give the expression itself (not just the number 24) as your
  final answer." plus the standard `\boxed{}` instruction.
- Correct only if the model finished thinking (`</think>`) and the last `\boxed{}` of the answer holds an
  expression that uses exactly the four numbers (as a multiset), only + − × ÷ and parentheses, and equals 24 in
  exact arithmetic. LaTeX forms (`\times`, `\div`, `\frac`) and a trailing "= 24" are accepted. No LLM judge.
- The grader was checked on hand-made cases and on the base-only pilot (20 practice puzzles) before this commit.
  Pilot (base, greedy, 49 min): 1 of 20 correct, 19 of 20 stopped by the loop detector, all of them verbatim
  repetition (e.g. "Wait, maybe I can do (11 - 3) × (1 + 1) × (1/1). Hmm, same." over and over). On one puzzle
  the base model wrote a valid answer, (4 + 4) ÷ (1/3), and kept looping instead of stopping. The one finished
  answer was graded correctly. No change to prompt or grader was needed. No edited model ran on the pilot.

## Regimes

- `greedy`: T 0, 1 sample.
- `sampled1`: T 0.6, top-p 0.95, **1 sample per problem** (more problems instead of more samples per problem).
- Both: 16,384-token cap, batch-invariant kernels, loop kill on, the same seeds procedure as `gen.py`.

## The fine-tune `ft` (fixed before it is built)

1. **Data** (`a4_ft_data.py`): the base model answers `ft_problems1200` at T 0.6 (1 sample, 3,072-token cap,
   loop kill on). Keep the answers that finished and are graded correct (rejection-sampling fine-tuning on the
   model's own answers, the standard cheap way to teach a model to finish). The 3,072 cap is a memory limit of
   the 6 GB GPU.
2. **Training** (`a4_ft_train.py`): LoRA r 16, alpha 32, dropout 0.05 on all attention and MLP projections;
   AdamW lr 1e-4, 10 warm-up steps then cosine; 2 epochs; 16 sequences per step; grad clip 1.0; seed 0; loss on
   response tokens only, EOS appended. No hyperparameter search, no early stopping, no test data. Merged into
   the bf16 weights as `work/ft`.
3. One fine-tune is built. If training crashes or diverges (NaN loss) it is fixed and rerun; that is reported.
   Settings are not changed because of evaluation results.

## Arms and runs

| Set | Regime | Arms | Runner |
|---|---|---|---|
| `game24_300` | greedy | base, e2, ft | `a4_eval.py` |
| `game24_300` | sampled1 | base, e2 | `a4_eval.py` |
| `math_fresh1200` | sampled1 | base, e2 | `a4_eval.py` |
| MATH-500 fixed 200 | greedy | ft (base, e2 exist) | `eval.py --limit 200` |
| `gsm8k_fresh400` | greedy | ft (base, e2 exist) | `eval.py` |
| `mmlupro_fresh300` | greedy | ft (base, e2 exist) | `eval.py` |
| MATH-500 fixed 200 | sampled (2 samples) | ft (base, e2 exist) | `eval.py` |

Existing base and e2 results on the last four rows were produced by the same code and kernels and are reused.

## Primary tests (family P, Holm over 3, alpha 0.05)

- P1: e2 vs base, `game24_300` greedy accuracy.
- P2: e2 vs base, `game24_300` sampled1 accuracy.
- P3: e2 vs base, `math_fresh1200` sampled1 accuracy.
- Exact two-sided McNemar, paired by problem; 95% bootstrap CIs (10,000 resamples of problems).
- Power for P3, from the MATH-500/200 sampled pilot (29 fixed / 16 broken of 400 sample pairs), by simulation:
  85% at alpha 0.05 and 73% at 0.0167 for 1,000 problems; 1,200 chosen for about 80% at the strictest Holm level.
- Claims: "helps on a task nobody edited for" only if P1 or P2 passes; "significant at T 0.6" only if P3 passes.

## Head-to-head with the fine-tune (family H, Holm over 4, alpha 0.05)

- e2 vs ft, accuracy: MATH-500/200 greedy, `gsm8k_fresh400` greedy, `mmlupro_fresh300` greedy, `game24_300` greedy.
- Exact two-sided McNemar paired by problem, 95% bootstrap CI of (e2 − ft).
- Wording on each set:
  - "edit beats the fine-tune": Holm-significant in e2's favour.
  - "fine-tune beats the edit": Holm-significant in ft's favour.
  - "edit matches the fine-tune": neither, and the CI's lower bound is above −5 points.
  - otherwise "inconclusive".
- The write-up may say the edit "can stand in for a fine-tune for this fix" only if it beats or matches the
  fine-tune on both MATH-500/200 and `gsm8k_fresh400`.

## Secondary (reported, not used for claims)

- ft vs base on every set; loop, cap, tokens and wall-clock for every run.
- Cost of each fix: GPU time of the fine-tune (data generation + training) vs the editor's 30 runs; zero
  gradient steps for the edit; trainable parameters (LoRA) vs edited weights (9,215).
- ft vs e2 vs base on MATH-500/200 sampled.
- Game of 24: answers that never finish thinking, by arm.

## Order

1. Game of 24: base, e2 (greedy); base, e2 (sampled1).
2. Fine-tune: data, training, then ft on MATH-500/200, `gsm8k_fresh400`, `mmlupro_fresh300`, `game24_300`.
3. `math_fresh1200` sampled1: base, e2.
4. ft on MATH-500/200 sampled.
