# Pre-registered evaluation plan

Fill in the blanks, then commit this file (`git commit`) BEFORE the editing run starts.
The commit date is your proof that the plan came first. Anything not listed here is reported as exploratory.

- Commit date: 2026-11-04 (the git commit titled "pre-registration"; hash in RUNLOG.md)
- Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (bf16, revision hash: ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562)

## Test sets (sealed until the patch is frozen)
| Set | Size | Role |
| --- | --- | --- |
| MATH-500 | 500 | Primary |
| GSM8K test (seed 1234 subset) | 400 | Primary |
| MMLU-Pro (seed 1234 subset) | 300 | Side effects |
| IFEval | 541 | Transfer (optional, lm-eval-harness, reasoning stripped) |

Fast track (the first post): MATH-500, the fixed 200-problem subset `eval.py --limit 200`
(`random.Random(0).sample`, so the same 200 for every arm). All primary comparisons below are run on
this subset first; the full sets follow for the follow-up post and use the same tests.

## Decoding
- Greedy: temperature 0, 1 sample. Headline regime.
- Sampled: temperature 0.6, top-p 0.95, 2 samples per problem.
- Token cap: 16,384 (pilot: 7 of 20 practice answers hit it, 6 of the 7 looping). Chunk 2,048.
- Kernels: vLLM batch-invariant mode (`VLLM_BATCH_INVARIANT=1`, the default in `gen.load_llm`) for every
  run, with FLASH_ATTN attention, bf16, vLLM 0.19.1 on the RTX 4050 Laptop. Reason: without it greedy
  text depends on batch composition (pilot finding, RUNLOG.md); with it reruns are byte-identical.
  Loop early stop on (an answer is stopped and counted as a loop once the detector fires).

## Loop detector (validated in the pilot)
- Window 2,048 tokens, 20-gram, flag at >= 60% repeated positions (adjusted to: unchanged).
- Pilot false-positive rate (flagged but finished correctly): 0 of 6 flagged (0/20 problems), 20 practice
  MATH-train problems, 16,384 cap, early stop off.

## Arms
1. Base
2. Claude's edit (frozen patch; its sha256 is recorded in RUNLOG.md at freeze)
3. Random edit, same weights, same norm (seed 0)
4. Random edit, same size, random positions, same norm (seed 0)
5. Base + presence penalty 1.5
6. Edit + presence penalty 1.5
7. (exploratory) Edit scaled x0.5 and x1.5

## Primary comparisons (Holm-corrected)
- Edit vs base: MATH-500 greedy accuracy; MATH-500 greedy loop rate; GSM8K greedy accuracy.
- Edit vs random (same weights): MATH-500 greedy accuracy.

## Statistics
Paired by problem. Exact McNemar (greedy) or sign-flip permutation (sampled) p-values.
95% bootstrap CIs, 10,000 resamples of problems. alpha = 0.05 after Holm.

## Noise floor (run-to-run, measured before editing)
- Procedure: rerun the base model on the same 200 MATH-500 problems, greedy, the identical command
  in a fresh process but with normal (non-batch-invariant) kernels: `VLLM_BATCH_INVARIANT=0`,
  `--name base_rerun`. An exact batch-invariant rerun would reproduce the text byte for byte and measure
  nothing, so the floor is defined as the effect of a numerically harmless change to the arithmetic
  (kernel and batch-order rounding). After the edit eval, the same for the edited model (`edit_rerun`).
- Reported: problems whose correct/incorrect status flips (both directions), accuracy difference,
  loop-flag flips, number of responses whose text differs.
- Decision rule (confirmed by Dev, 2026-11-04): an edit effect counts as clearly above
  the noise floor only if (a) it passes the primary test above, and (b) its net fixed − broken count
  exceeds 2 × the larger flip count of `base_rerun` and `edit_rerun`.
- Every headline number in the write-up is reported with this noise floor next to it.
- Measured values are recorded in RUNLOG.md (Phase 4 / Phase 7); this file is not edited after commit.

## Cross-checks (secondary, not part of the primary comparisons)
- lm-evaluation-harness, GSM8K, base vs edit (vLLM backend, model chat template, reasoning stripped
  up to `</think>`, max tokens = token cap). Reported: agreement in direction and rough size with eval.py.
- Optional: Kaggle GPU, base and edit on the first 100 (by id) of the 200 fast-track problems.

## Example selection rule
For each MATH subject, the lowest-index problem the edit fixed. No other picking.

## Editing run
- Editor: Claude Opus 5.5 via Claude Code. Budget: 50 forward passes. Max ~200 turns.
- Opening prompt: "Make this model better at math by editing its weights directly. You may not train it."
- Hint given? no (if a second hinted run is done, it is reported separately)
- The editor's `fp.py` runs use the same kernels and settings as eval.py (shared `gen.load_llm`).
