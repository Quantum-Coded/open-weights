# Amendment 1 to the pre-registered evaluation plan

- Date: 2026-11-04. Approved by Dev before the editing run. `eval_plan.md` (pre-registration commit
  ea01b03) is not edited; this file adds to it.
- State when written: no edited model exists, the editor has not started, and no edit, random-control
  or edit + presence-penalty result exists. The only test-set results are the three Phase 4 base arms
  (`base`, `base_pp`, `base_rerun`).

## Why
The pre-registered noise floor was measured in Phase 4 (RUNLOG.md). One normal-kernel rerun of the base
model on the 200 fast-track problems flipped 31 grades (20 wrong->right, 11 right->wrong; net +9) and
changed the text of 198/200 answers. The original rule (b) compares the edit's *net* fixed − broken
count with 2 × the rerun's *total* flip count (here 62, i.e. +31 points from a 66% base). That compares
a net number with a gross one, and one rerun says little about how far the net can move by chance.

## What is added
1. **More reruns, before sealing.** Four more base reruns on the same 200 MATH-500 problems, greedy,
   normal kernels, with the problem order shuffled so that batch composition differs between runs:
   `VLLM_BATCH_INVARIANT=0 python eval.py --model work/base --name base_rerun<k> --set math500 --regime greedy --limit 200 --order-seed <k>`
   for k = 2, 3, 4, 5 (`--order-seed` changes only the order problems are fed in; the problems,
   prompts and settings are the same). Together with `base_rerun` this gives 5 reruns.
2. **Rerun noise statistic.** For each rerun r, net_r = (wrong->right) − (right->wrong) against `base`.
   N = max over the 5 base reruns of |net_r|, and over `edit_rerun` of |net| against `edit` (the
   Phase 7 rerun of the edited model, normal kernels, as in eval_plan.md).
3. **Amended decision rule (b').** The edit's effect on MATH-500 greedy accuracy counts as clearly above
   the noise floor only if (a) it passes the primary test in eval_plan.md, and (b') its net fixed − broken
   count against `base` exceeds N.

## What does not change
- Arms, primary comparisons, Holm correction, statistics, token cap, detector, example-selection rule.
- The original rule (b) is still evaluated and reported next to (b'), labelled "original pre-registered
  rule"; the write-up states which rules the edit passes.
- The write-up reports the full spread of the reruns (each rerun's accuracy, flips both ways, net, loop
  rate) next to every headline number.
- Measured values go in RUNLOG.md. This file is not edited after its commit.
