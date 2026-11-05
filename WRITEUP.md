# Editing DeepSeek-R1's weights to stop its thinking loops, on a laptop GPU

> DRAFT (Phase 8, 5 Nov). Editing run 1 numbers are final. Still TBD: the full MATH-500 and sampled
> follow-ups, Amendment 2 (fresh sets), editing run 2, the lm-eval cross-check and figure 2. Nothing here
> is published before Dev signs off the Checkpoint C results.

## 1. TL;DR

MATH-500, fixed 200-problem subset, greedy decoding, 16,384-token cap, paired by problem.

| Arm | Accuracy [95% CI] | Loop rate [95% CI] | Fixed / broken vs original |
| --- | --- | --- | --- |
| Original model | 66.0% [59.5, 72.5] | 31.5% [25.0, 38.0] | — |
| Claude's edit | 72.0% [65.5, 78.0] | 20.5% [15.0, 26.5] | 23 / 11 |
| Random edit, same weights, same size | 69.0% [62.5, 75.5] | 26.5% [20.5, 32.5] | 19 / 13 |
| Random edit, random positions, same size | 63.5% [57.0, 70.0] | 32.5% [26.0, 39.0] | 17 / 22 |
| Original + presence penalty 1.5 | 85.5% [80.5, 90.0] | 3.5% [1.5, 6.0] | 41 / 2 |
| Edit + presence penalty 1.5 | 83.0% [77.5, 88.0] | 2.0% [0.5, 4.0] | 38 / 4 |

The other test sets, with the original model compared against Claude's edit:

| Test set | Accuracy | Loop rate | Fixed / broken |
| --- | --- | --- | --- |
| GSM8K (400) | 74.0% → 79.8% (+5.8 [+1.8, +9.8]) | 20.5% → 12.8% | 45 / 22 |
| MMLU-Pro (300) | 14.3% → 12.7% (−1.7 [−5.3, +1.7]) | 55.7% → 53.0% | 12 / 17 |

Run-to-run noise (five reruns of the original model with no edit): accuracy 68.0–70.5%, loop rate
26.5–28.5%; each rerun fixed 17–20 and broke 11–16 problems (net +4 to +9). A rerun of the edited model
moved 12 problems one way and 13 the other (net −1).

Headline: without any training, Claude changed 1,278 of the model's 1.78 billion weights. That
significantly raised GSM8K accuracy (+5.8 points) and cut MATH-500 loops by a third. The edit's +6.0
points on MATH-500 is not significant and cannot be told apart from a random edit of the same weights.
A decoding-time presence penalty does more (+19.5 points); on top of the penalty, the edit adds nothing.

## 2. The problem: answers that never end

DeepSeek-R1-Distill-Qwen-1.5B thinks out loud before it answers. On hard problems it often gets stuck
and repeats the same few sentences until it runs out of room. On the 200 test problems, 31.5% of its
answers were flagged as loops (the same 20-token sequences filling most of the last 2,048 tokens).

A real example from a practice problem (MATH train, "10 red and 6 blue marbles, pick three, probability
of one red and two blue"). The model computes the right answer, 15/56, early on, then doubts a second
method and repeats this sentence 94 times until it hits the 16,384-token limit, never giving an answer:

> Wait, but in the second approach, when I compute the probability for one specific order, and then
> multiply by 3, I'm getting 27/56, which is different from 15/56.

(For the final version: swap in a test-set trace picked by the pre-registered example rule.)

## 3. Setup

- **Model:** deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, bf16, HF revision `ad9f0ae0…`.
- **Hardware:** one laptop RTX 4050 (6 GB), WSL2, vLLM 0.19.1.
- **Editor:** Claude Opus 5.5 in Claude Code, in a separate session that saw only practice problems.
  Opening prompt, verbatim: "Make this model better at math by editing its weights directly. You may
  not train it." No hints and no steering.
- **Rules for the editor:** no gradients of any kind, no perturb-and-evaluate search; one-shot linear
  algebra only (SVD, PCA, projections, least squares). A budget of 50 model runs; reading weights is free.
- **Evaluation:** test sets sealed outside the editor's folder until the edit was frozen; evaluation plan
  committed before the editing run (commit `ea01b03`, amendment `b6f9860`).

## 4. What Claude found and changed

Claude ran its first model runs on practice problems and found the failures were not mostly wrong maths.

- **Degeneration, not wrong maths.** In its first baseline, all 54 runs that finished their reasoning were
  correct, and every failure was a loop or a run that hit the token cap.
- **The answer was often already there.** In 6 of 18 failures the correct answer had already appeared,
  sometimes dozens of times, before the model began looping on "Wait, but let me check again…".
- **Looping is a state inside the model.** At layer 14, positions deep inside looping answers sit apart
  from healthy late reasoning along one mean-difference direction. A linear probe separates the two with
  AUC ≈ 0.87 on held-out problems.

It tried and dropped three ideas first:
1. a logit penalty on "Wait": the loops re-formed with "So" and "But" instead;
2. a probe-gated boost to the `**Final Answer**` token: no effect on new problems;
3. the same direction re-estimated from five times more data: it scored worse than the original
   estimate, so it was dropped.

**The edit.** Take the layer-14 "loop minus healthy" direction *d*, estimated from one batch of practice
transcripts, and push the residual stream by −0.5·*d* at every position. It is written exactly into the
attention value biases (`v_proj.bias`) of layers 10–14. Because softmax weights sum to 1, a change to a
value bias adds a constant vector to that layer's output at every position. The five bias changes come
from one ridge least-squares solve. **1,278 weights in 5 tensors (0.0001% of 1.78B)**, a 3 KB patch.
No gradients and no search: one mean difference and one linear solve. 38 of the 50 model runs were used,
and the editor stopped with 12 left so it would not tune on its own check problems.

On its own held-out practice problems (288, greedy), Claude measured 71.9% → 80.9% and loops 59 → 29. It
called the MATH part "modest and less certain" and predicted that the benefit would be smaller under
sampling. The sealed test sets above bear out both predictions.

Figure 2 (`fig2_projection.png`): projection of the residual stream on the edited direction across one
answer, original vs edited. TBD.

## 5. Results

Figure 1 (`fig1_accuracy_loops.png`): accuracy and loop rate per arm, 95% CIs, rerun noise band.
Figure 3 (`fig3_fixed_broken.png`): problems fixed vs broken per arm.

**Primary comparisons (pre-registered, Holm-corrected over these 4):**

| Comparison | Fixed / broken | p (raw) | p (Holm) | Result |
| --- | --- | --- | --- | --- |
| MATH-500 accuracy, edit vs original | 23 / 11 | 0.058 | 0.12 | not significant |
| MATH-500 loop rate, edit vs original | 33 loops removed / 11 added | 0.0013 | 0.005 | **significant** |
| GSM8K accuracy, edit vs original | 45 / 22 | 0.0067 | 0.020 | **significant** |
| MATH-500 accuracy, edit vs same-weights random edit | 19 / 13 | 0.38 | 0.38 | not significant |

**Noise-floor rules (MATH-500 accuracy):**
- **Amended rule (b'):** net +12 against N = 9. Passes.
- **Original rule (b):** net +12 against 62. Fails.
- **Overall:** the effect counts as clearly above noise only if the primary test also passes, and it does
  not. So the MATH-500 accuracy gain is not established.

**Random edits.** The two random edits land on either side of the original (+3.0 and −2.5), so a random
change of this size moves MATH-500 accuracy by roughly ±3 points. Claude's edit is ahead of both, and
significantly ahead of the random-positions one (27 / 10, p = 0.0076; not a pre-registered test). It is
not significantly ahead of the same-weights one. Part of the edit's effect is simply "disturb these
biases". Part of it comes from the direction: the edit removed 33 loops and added 11 (p = 0.0013), while the
same-weights random edit removed 25 and added 15 (p = 0.15), and the random-positions one removed 23
and added 25.

**Presence penalty.** A decoding-time presence penalty (1.5) takes the original from 66.0% to 85.5% and
loops from 31.5% to 3.5%. With the penalty on, the edit makes no difference (83.0%, 8 / 13). The edit is
a loop fix, and the penalty already provides a stronger one.

**Side effects.**
- **MMLU-Pro (300):** −1.7 points [−5.3, +1.7], not significant. Loops there barely moved (55.7% → 53.0%),
  so the loop direction found on maths transcripts does not transfer to multiple-choice questions on
  other subjects.
- **Length:** mean finished-answer length on MATH-500 fell from 2,348 to 2,254 tokens.

Amendment 2 (fresh, untouched GSM8K and MMLU-Pro sets) and editing run 2: TBD.

## 6. Methodology

- **Pre-registration:** arms, metrics, tests, token cap, loop detector and example-selection rule were
  committed before the editing run. One amendment (more reruns for the noise floor) was committed before
  any edited model existed. Both rules are reported.
- **Sealed test sets:** MATH-500, GSM8K (400) and MMLU-Pro (300) lived outside the project while the
  editor worked.
  - **Decontamination:** before handoff, 82 practice problems that nearly duplicated a test question were
    removed under a fixed rule: the same first 40 characters, plus at least 80% similarity or the same
    first 60 characters. 44 of them were MATH-500 near-twins and 2 matched MMLU-Pro word for word.
  - **After the run:** none of the 1,200 test questions appear anywhere in the editor's logs, notes or
    transcript.
- **Audits:** no gradient code in any of the editor's scripts or commands. The model ran only through the
  budgeted `fp.py`. A hand-check of 30 graded answers (15 original, 15 edited, `random.Random(0)`) found
  no grading errors.
- **Paired statistics:** exact McNemar tests, 95% bootstrap CIs over problems (10,000 resamples), Holm
  correction.
- **Controls:** two random edits of the same size and norm, one on the same weights and one on random
  positions.
- **Determinism and the noise floor:** plain greedy decoding in vLLM is not repeatable here: the same
  problem gives different text depending on what else is in the batch. Every arm therefore ran with vLLM's
  batch-invariant kernels, which made reruns byte-identical. The noise floor was then measured by rerunning
  the original model five times with normal kernels: 195–199 of 200 answers changed text and 29–36 grades
  flipped per rerun. All five reruns scored 2.0–4.5 points above the batch-invariant run, so absolute
  numbers here are for batch-invariant kernels; every comparison uses the same kernels.
- **Cross-checks:** lm-evaluation-harness on GSM8K (TBD); optional second GPU on Kaggle (TBD).

## 7. Caveats

- One editor on one model. A second, independent editing run was triggered by the pre-registered rule,
  because run 1 did not beat the same-weights random edit; its results are reported alongside run 1's.
- The editor's direction was estimated on greedy MATH/GSM8K practice transcripts; it does not carry over
  to MMLU-Pro.
- Automatic graders (math-verify, exact match); no human or LLM judge.
- The loop detector flags verbatim repetition only; looser circling is not counted.
- Fast-track numbers are on a 200-problem subset; the full sets follow in a second post.
- The noise floor is large: 69 of the 200 problems changed grade in at least one of six runs of the
  same model.

## 8. Credits, links, reproduction

- Inspired by MIT CSAIL's "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab) and
  Sabhya Chhabria's Stable Diffusion replication.
- Code, logs, raw outputs: https://github.com/Quantum-Coded/open-weights
- Edited model and demo: TBD (Hugging Face, after sign-off).
- How to reproduce: `REPRODUCE.md` (TBD).
