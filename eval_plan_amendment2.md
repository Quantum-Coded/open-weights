# Amendment 2: fresh untouched test sets (exploratory)

- Written by the operator and approved by Dev on 2026-11-05 (~18:40 IST), committed before any model has
  been run on the fresh sets.
- State when written: the editing run 1 patch is frozen (sha256 4c8a5c77...). Its MATH-500 fast-track arms
  edit, rand_w, rand_s and edit_pp have been seen; edit_rerun and the GSM8K / MMLU-Pro / full MATH-500 /
  sampled follow-ups have not. Editing run 2 has not started.
- Dev's goal (verbatim): "ideally i want to have a measurable gain on an untouched application of the newly
  discovered weight editing technique without training and within the hardware limitation of this laptop."
- Status: exploratory. These results never replace or reinterpret the results in `eval_plan.md` or
  `eval_plan_amendment.md`. They are labelled "Amendment 2 (fresh sets)" wherever they are reported.

## Fresh test sets (built by `make_fresh.py`, sealed alongside the original test sets)
- `gsm8k_fresh400`: 400 GSM8K test problems not in `gsm8k_test400`.
  sha256 951b7f05ca013c634a0345eedf4a8dd6c344e7c57e4b086dbe2d31ae53517201.
- `mmlupro_fresh300`: 300 MMLU-Pro test questions not in `mmlupro300`.
  sha256 5dc658b25cf8c25f6035ad4869d693b1c3c280ad466b47fef9ffbabcf8216153.
- Procedure, fixed before building: shuffle the remaining indices with `random.Random(20261105)`, walk in
  that order, skip any question that matches a practice problem under the Phase 5 decontamination rule,
  and keep the first 400 / 300. That skipped 0 GSM8K and 4 MMLU-Pro questions.
- No model, editor or operator analysis touched these sets before this amendment was committed. The editor
  never sees them; they are sealed with the other test sets during editing run 2.

## Arms on the fresh sets
- Every arm runs on both fresh sets with greedy decoding, batch-invariant kernels, the 16,384-token cap
  and the same `eval.py` and grader.

| Arm | Model |
|---|---|
| `base` | the unedited model |
| `e1` | run 1's frozen patch, scale 1.0 |
| `e1x1.5` | run 1's frozen patch, scale 1.5 (the exploratory scale already listed as arm 7 in `eval_plan.md`) |
| `rand_w` | run 1's same-weights random control (seed 0, already built) |
| `e2` | run 2's frozen patch, scale 1.0, only if run 2 produces a patch |

- No other arms, scales or settings, and no choosing among arms. Every arm is reported.

## Primary comparisons (Holm-corrected together)
- `e1`, `e1x1.5` and (if it exists) `e2` against `base`, accuracy, on each fresh set: up to 6 tests.
- Exact McNemar, paired by problem; alpha = 0.05 after Holm.
- 95% bootstrap CIs (10,000 resamples).
- A "measurable gain on an untouched application" is claimed for an arm on a set only if its Holm-adjusted
  p < 0.05 and the accuracy change is positive.

## Secondary (reported, not used for the claim)
- `e1` against `rand_w` on each set.
- Loop and cap rates.
- MMLU-Pro accuracy by category.

## Order
- After `phase7.sh 1` finishes: `base`, `e1`, `e1x1.5` and `rand_w` on both fresh sets.
- Then editing run 2. Test data is sealed again first, the fresh sets included.
- `e2` runs on the fresh sets after run 2's freeze.
