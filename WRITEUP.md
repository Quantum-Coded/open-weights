# Editing DeepSeek-R1's weights to stop its thinking loops, on a laptop GPU

> DRAFT (Phase 6). Every number marked TBD comes from `analyze.py` after the freeze. Baseline and
> noise-floor numbers are final (Phase 4, RUNLOG.md). Nothing here is published before Dev signs off
> the Checkpoint C results.

## 1. TL;DR

MATH-500, fixed 200-problem subset, greedy decoding, 16,384-token cap, paired by problem.

| Arm | Accuracy [95% CI] | Loop rate [95% CI] | Fixed / broken vs original |
| --- | --- | --- | --- |
| Original model | 66.0% [TBD] | 31.5% [TBD] | — |
| Claude's edit | TBD | TBD | TBD |
| Random edit, same weights, same size | TBD | TBD | TBD |
| Random edit, random positions, same size | TBD | TBD | TBD |
| Original + presence penalty 1.5 | 85.5% [TBD] | 3.5% [TBD] | TBD |
| Edit + presence penalty 1.5 | TBD | TBD | TBD |

Run-to-run noise (five reruns of the original model with no edit): accuracy 68.0–70.5%, loop rate
26.5–28.5%; each rerun fixed 17–20 and broke 11–16 problems (net +4 to +9).

Headline sentence: TBD (one sentence, says whether the edit beats the noise floor and the random edits,
and how it compares with the presence penalty).

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

TBD after freeze, from the editor's notebook and FINAL_REPORT.md: which layers and modules, the direction
it found and how, the edit formula, how many weights changed (0.0Z% of 1.78B), model runs used of 50.

Figure 2 (`fig2_projection.png`): projection of the residual stream on the edited direction across one
answer, original vs edited. TBD.

## 5. Results

Figure 1 (`fig1_accuracy_loops.png`): accuracy and loop rate per arm, 95% CIs, rerun noise band.
Figure 3 (`fig3_fixed_broken.png`): problems fixed vs broken per arm.

- Primary comparisons (Holm-corrected): edit vs original on MATH-500 accuracy and loop rate and on GSM8K
  accuracy; edit vs the same-weights random edit on MATH-500 accuracy. TBD.
- Noise-floor rules: original pre-registered rule (net fixed − broken > 2 × the largest rerun flip count,
  i.e. > 62) TBD; amended rule (net > the largest |net| of any rerun, currently 9) TBD.
- Baselines: a decoding-time fix (presence penalty 1.5) already takes the original from 66.0% to 85.5%
  and loops from 31.5% to 3.5%. The edit is compared against it, and combined with it. TBD.
- Side effects: MMLU-Pro (300) and token counts. TBD.

## 6. Methodology

- **Pre-registration:** arms, metrics, tests, token cap, loop detector and example-selection rule were
  committed before the editing run. One amendment (more reruns for the noise floor) was committed before
  any edited model existed. Both rules are reported.
- **Sealed test sets:** MATH-500, GSM8K (400), MMLU-Pro (300) lived outside the project while the
  editor worked; a grep check confirmed none of their questions were in the editor's folder.
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

- One editing run by one editor on one model; a second run would show how stable the method is.
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
