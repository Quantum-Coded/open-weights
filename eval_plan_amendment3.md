# Amendment 3: the run-1 technique applied to MMLU-Pro loops (exploratory)

- Written by the operator on 2026-11-06 and approved by Dev ("Yes so proceed with both tracks"). Committed
  before any model has run on the new data.
- State when written, run 1 on MMLU-Pro:
  - On `mmlupro300`, run 1's edit gave 14.3% → 12.7% accuracy and 55.7% → 53.0% loops, so there was no
    transfer.
  - Amendment 2's fresh-set runs are queued but have not started; no results from them exist.
  - Editing run 2 has not started.
- Why:
  - Run 1's edit was built only from MATH and GSM8K transcripts.
  - MMLU-Pro is where the base model loops most (56% of answers), so it has the most room to gain.
  - The question is whether the *technique* Claude discovered fixes MMLU-Pro loops when its loop direction
    is estimated from MMLU-Pro practice transcripts.
- Status: exploratory. It never replaces or reinterprets `eval_plan.md`, Amendment 1 or Amendment 2.
  Every result here is labelled "Amendment 3".

## Data (built by `make_a3_data.py`)
- Pool: MMLU-Pro test indices that are in neither `mmlupro300` nor `mmlupro_fresh300`.
- Selection: shuffle the pool with `random.Random(20261106)` and walk it in that order. Skip any question
  that matches, under the Phase 5 rule, an editor practice problem, a sealed MMLU-Pro question or a question
  already taken. 13 questions were skipped.
- `mmlupro_fresh2_300`: the first 300 kept questions. This is the sealed test set; no model has seen it.
  sha256 3a67923d6a7bb9e82270a515a8818e5a98f80720bb12073c1c5841c3522d18fb.
- `a3_practice_mmlupro128`: the next 128. They are used only to estimate the direction (practice data).
  sha256 9ad713a40937bdac37627d99065348c3f1f10f98d7d71640f4bd1a8e638726de.

## The edit `e3` (built by `a3_build.py`, no choices left open)
1. **Practice transcripts.** Base model, `a3_practice_mmlupro128`, greedy, 8,192-token cap (the editor's
   `fp.py` setting), batch-invariant kernels: `--name a3prac`.
2. **Direction.** Run 1's recipe from `editor/work/steer.py`:
   - Read the layer-14 residual over 1,000-token windows.
   - d = mean(failed runs, position ≥ 2000) − mean(healthy class).
   - The healthy class is the first of these with ≥ 2,000 positions:
     - A: correct runs, position ≥ 1000, before `</think>`. This is run 1's definition.
     - B: finished runs, position ≥ 1000, before `</think>`.
     - C: finished runs, position ≥ 300, before `</think>`.
   - Which class was used is reported.
3. **Patch.** Run 1's `make_h4.py`: target −0.5·d, written into `v_proj.bias` of layers 10–14 by the same
   ridge solve.
   - Check: fed run 1's direction, `a3_build.py` reproduces run 1's frozen patch exactly (0 of 1,280 values
     differ).
4. **No tuning.** α, layers, ridge λ, window rules and the cap are run 1's values. Nothing is evaluated on
   the test set before the patch is built.

## Arms on `mmlupro_fresh2_300`
- Settings: greedy, 16,384-token cap, batch-invariant kernels, the same `eval.py` and grader.

| Arm | Model |
|---|---|
| `base` | the unedited model |
| `e3` | the patch built above |
| `rand_e3` | random change on the same weights with the same per-tensor norm (`make_random_control.py`, same-weights, seed 0) |
| `base_pp` | the unedited model with presence penalty 1.5 (decoding-time reference) |

## Primary comparisons (Holm over 2)
- `e3` vs `base`, accuracy.
- `e3` vs `rand_e3`, accuracy.
- Exact McNemar, paired by problem, alpha = 0.05 after Holm, with 95% bootstrap CIs.
- "Measurable gain on an untouched application" is claimed only if `e3` vs `base` passes. "Beats a random
  edit" is claimed only if `e3` vs `rand_e3` passes.

## Secondary (reported, not used for claims)
- Loop and cap rates per arm.
- Accuracy by category.
- `e3` against `base_pp`.
- `e3` on the MATH-500 fast-track 200, greedy: does the MMLU-Pro direction hurt maths?

## Order
- After Amendment 2's runs finish: `a3prac`, then build `e3` and `rand_e3`, then `base`, `e3`, `rand_e3`
  and `base_pp` on `mmlupro_fresh2_300`, then `e3` on MATH-500 200.
- Then editing run 2, with all test data sealed first.
