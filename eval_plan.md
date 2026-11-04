# Pre-registered evaluation plan

Fill in the blanks, then commit this file (`git commit`) BEFORE the editing run starts.
The commit date is your proof that the plan came first. Anything not listed here is reported as exploratory.

- Commit date: ______
- Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (bf16, revision hash: ______)

## Test sets (sealed until the patch is frozen)
| Set | Size | Role |
| --- | --- | --- |
| MATH-500 | 500 | Primary |
| GSM8K test (seed 1234 subset) | 400 | Primary |
| MMLU-Pro (seed 1234 subset) | 300 | Side effects |
| IFEval | 541 | Transfer (optional, lm-eval-harness, reasoning stripped) |

## Decoding
- Greedy: temperature 0, 1 sample. Headline regime.
- Sampled: temperature 0.6, top-p 0.95, 2 samples per problem.
- Token cap: ______ (16,384 unless the pilot says otherwise). Chunk 2,048.

## Loop detector (validated in the pilot)
- Window 2,048 tokens, 20-gram, flag at >= 60% repeated positions (adjusted to: ______).
- Pilot false-positive rate (flagged but finished correctly): ______

## Arms
1. Base
2. Claude's edit (frozen patch, sha256 ______)
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

## Example selection rule
For each MATH subject, the lowest-index problem the edit fixed. No other picking.

## Editing run
- Editor: Claude Opus 5.5 via Claude Code. Budget: 50 forward passes. Max ~200 turns.
- Opening prompt: "Make this model better at math by editing its weights directly. You may not train it."
- Hint given? no (if a second hinted run is done, it is reported separately)
