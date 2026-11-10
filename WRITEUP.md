# Two neurons, no training: editing DeepSeek-R1's weights to stop its thinking loops, on a laptop GPU

> DRAFT (Phase 8, 11 Nov). All pre-registered results are final. Still TBD: lm-eval on run 1's edit
> (running), the single-stream race timings, figure 2. Nothing here is published before Dev signs off.

## 1. TL;DR

An AI editor (Claude, in a separate session that never saw a test question) was told: "Make this model
better at math by editing its weights directly. You may not train it." In its second independent run it
rewrote **two neurons in layer 19 of DeepSeek-R1-Distill-Qwen-1.5B: 9,215 of 1.78 billion weights
(0.0005%)**, computed in closed form (means, shrinkage LDA, thresholds). No gradients, no search.

Greedy decoding, 16,384-token cap, paired by problem, original model vs run 2's edit:

| Test set | Accuracy | Loop rate | Fixed / broken | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| MATH-500, fixed 200 | 66.0% → **80.5%** (+14.5 [+8.5, +20.5]) | 31.5% → 2.5% | 35 / 6 | 4.9e-6 ★ |
| MATH-500, all 500 | 70.0% → **81.8%** (+11.8 [+8.4, +15.4]) | 24.4% → 3.4% | 75 / 16 | 2.7e-10 |
| GSM8K (400) | 74.0% → **85.0%** (+11.0 [+7.2, +14.8]) | 20.5% → 3.0% | 54 / 10 | 2.0e-8 ★ |
| GSM8K fresh (400, built after the edit was frozen) | 72.2% → **84.2%** (+12.0 [+8.5, +15.5]) | 23.5% → 2.2% | 53 / 5 | 3.5e-11 ★ |
| MMLU-Pro fresh (300, not maths) | 9.7% → **22.3%** (+12.7 [+8.3, +17.3]) | 51.3% → 10.0% | 45 / 7 | 7e-8 ★ |
| MMLU-Pro (300) | 14.3% → **29.3%** (+15.0 [+10.7, +19.7]) | 55.7% → 7.7% | 50 / 5 | 2.1e-10 |

★ = pre-registered primary test, passes after Holm correction. Against a random edit of the same 9,211
weights with the same norm: 80.5% vs 58.5% on MATH-500 (48 / 4, p = 1.3e-10, pre-registered, passes).

What it costs and what it does not do:
- **It is a loop fix.** Nearly all of the gain comes from answers that used to repeat themselves until the
  token limit and now finish. A decoding-time presence penalty (1.5) does about as well on MATH-500
  (85.5% vs 80.5%; the penalty is ahead on 17 problems, the edit on 7, p = 0.064), and the edit adds nothing on top of it (83.0%).
- **It depends on the prompt.** Under lm-evaluation-harness's zero-shot GSM8K prompt ("Q: … A: Let's think
  step by step.") the gain shrinks to +3.3 points (71.8% → 75.0%, 28 / 15, p = 0.066), although loops
  still drop from 48 to 5. See §5.5.
- **Sampling.** At temperature 0.6 (2 samples per problem) the original rarely loops (1.0%), so the gain
  is small: 82.0% → 85.2% on MATH-500/200 (24 / 11, not significant).

Why it matters in practice: on the 400 fresh GSM8K problems, with every answer allowed to run to the
limit, the original model needed **6 h 48 min** on this laptop; the edited one finished in **59 min** with
61% fewer tokens (post hoc, descriptive; §5.4).

## 2. The problem: answers that never end

DeepSeek-R1-Distill-Qwen-1.5B thinks out loud before it answers. On hard problems it often gets stuck and
repeats the same few sentences until it runs out of room. On the 200 MATH-500 test problems, 31.5% of its
answers were flagged as loops (the same 20-token sequences filling most of the last 2,048 tokens). They do not
recover: on 400 fresh GSM8K problems run with no loop cut-off, 95 answers hit the 16,384-token limit and
94 of them were still repeating themselves word for word when they got there.

The race example (picked by a rule fixed before looking: among fresh GSM8K problems where the original
loops and the edit is right, the shortest edited answer). gsm-665: "Out of the 150 teachers on the school
basketball court, 60% are history teachers… calculate the total time the math teachers collectively spend
sleeping in one day." The original works out 90 history teachers and 60 maths teachers, then writes this
until the limit, never answering:

> So, 60 is two-thirds of 90, which is 60% of 150.

The edited model reaches 60 × 6 = 360 and stops after 762 tokens.

## 3. Setup

- **Model:** deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, bf16, HF revision `ad9f0ae0…`.
- **Hardware:** one laptop RTX 4050 (6 GB), WSL2, vLLM 0.19.1.
- **Editor:** Claude Opus 5.5 in Claude Code, in a separate folder that held only practice problems.
  Opening prompt, verbatim: "Make this model better at math by editing its weights directly. You may not
  train it." No hints. The only other message ever sent was "Continue from your notebook." after a
  network drop (run 2).
- **Rules for the editor:** no gradients of any kind, no perturb-and-evaluate search; one-shot linear
  algebra only. A budget of 50 model runs; reading weights is free.
- **Two independent runs.** The plan said: if run 1 does not beat a same-weights random edit, start a
  second run from a clean copy with none of run 1's notes. Run 1 did not, so run 2 happened.
- **Evaluation:** test sets sealed outside the editor's folder; evaluation plan and amendments committed
  before the edits they test (plan `ea01b03`, amendments `b6f9860`, Amendment 2, Amendment 3).

## 4. What the editor found and changed

Both runs found the same thing first: **the model's maths is mostly fine; it fails because it does not
stop.** In run 2's first baseline, all 22 answers that finished were correct and all 10 failures were loops
or cap hits. Often the right answer was already written before the loop began.

**Run 2's edit (the headline).** The loop is a state the model can see in itself: at paragraph starts, a
linear probe in the residual stream separates "loop restart" from a productive "Wait, let me check" with
AUC ≈ 0.97 (held out by problem). But at the output both predict the same next words, so a simple token
penalty cannot tell them apart. The editor therefore built two detectors inside the model:

- It took the two least-used neurons of the layer-19 MLP (3918 and 4028) and rewrote their weights. Each
  neuron computes silu(gate·x) × (up·x), which works as an AND of two conditions.
- The **up row** detects "start of a paragraph". The **gate row** is a thresholded linear detector
  (shrinkage LDA).
- **Neuron 3918, "loop exit":** fires at a paragraph start that looks like a loop restart, and writes the
  shift from that state to the average state just before "**Final Answer**".
- **Neuron 4028, "close thinking":** fires at a paragraph start right after a boxed answer, and writes the
  shift toward the state just before `</think>`. It was added because 3918 alone sometimes made the model
  repeat "**Final Answer** \boxed{X}" instead of closing its reasoning.
- Read checks: at loop restarts, 45 of 90 switch from "Wait"/"But" to "**"; productive "Wait"s and normal
  text are unchanged (inside paragraphs the top token flips at 13 of 1,350 positions, KL ≤ 0.003).
- 9,215 weights in 3 tensors. Means, LDA and quantile thresholds only. 30 of 50 model runs used.
- The editor's own fresh practice check: 63 → 83 of 96, loops/caps 26/9 → 1/3. Its stated caveats:
  sampling, non-maths tasks and other output formats were untested, and the "conclude" direction was
  learned from one closing style. The test sets bear out the first and last caveats (§1, §5.5).

**Run 1's edit (first attempt, for comparison).** A layer-14 "loop minus healthy" direction, pushed into
the residual stream at every position through the value biases of layers 10–14: 1,278 weights. It raised
GSM8K by +5.8 (replicated at +6.0 on the fresh set) and cut MATH loops by a third, but its MATH accuracy
gain was not significant and it did not beat the same-weights random edit. It did nothing on MMLU-Pro.

| | Run 1 | Run 2 |
| --- | --- | --- |
| Where | value biases, layers 10–14 | 2 MLP neurons, layer 19 |
| Weights changed | 1,278 | 9,215 |
| Acts | at every position | only at paragraph starts that look like a loop or a finished answer |
| MATH-500/200 | 72.0% (+6.0, ns) | 80.5% (+14.5) |
| MATH loops | 31.5% → 20.5% | 31.5% → 2.5% |
| GSM8K fresh | 78.2% (+6.0) | 84.2% (+12.0) |
| MMLU-Pro fresh | 10.7% (+1.0, ns) | 22.3% (+12.7) |
| vs same-weights random edit | 19 / 13, ns | 48 / 4, p = 1.3e-10 |

Run 2's edit beats run 1's directly on MATH-500/200: 27 / 10, p = 0.0076.

## 5. Results

### 5.1 Pre-registered primary tests

Run 2 family (Holm over 4), exact McNemar:

| Comparison | Fixed / broken | p (raw) | p (Holm) | Result |
| --- | --- | --- | --- | --- |
| MATH-500 loop rate, edit vs original | 60 loops removed / 2 added | 8.5e-16 | 3.4e-15 | **pass** |
| MATH-500 accuracy, edit vs same-weights random edit | 48 / 4 | 1.3e-10 | 3.9e-10 | **pass** |
| GSM8K accuracy, edit vs original | 54 / 10 | 2.0e-8 | 4.0e-8 | **pass** |
| MATH-500 accuracy, edit vs original | 35 / 6 | 4.9e-6 | 4.9e-6 | **pass** |

Amendment 2 (fresh sets built after run 1's edit was frozen, never seen by either editor; Holm over 6):

| Comparison | Fixed / broken | p (Holm) | Result |
| --- | --- | --- | --- |
| Run 2 edit, GSM8K fresh | 53 / 5 | 2.1e-10 | **pass** |
| Run 2 edit, MMLU-Pro fresh | 45 / 7 | 3.5e-7 | **pass** |
| Run 1 edit ×1.5, GSM8K fresh | 49 / 23 | 0.012 | **pass** |
| Run 1 edit, GSM8K fresh | 45 / 21 | 0.013 | **pass** |
| Run 1 edit ×1.5, MMLU-Pro fresh | 18 / 10 | 0.37 | fail |
| Run 1 edit, MMLU-Pro fresh | 15 / 12 | 0.70 | fail |

Run 1's family (reported in full in RUNLOG): GSM8K and MATH loops passed; MATH accuracy and the
random-edit comparison failed.

### 5.2 All MATH-500 arms (fixed 200 problems)

| Arm | Accuracy [95% CI of change] | Loop rate | Fixed / broken vs original |
| --- | --- | --- | --- |
| Original | 66.0% | 31.5% | — |
| **Run 2 edit** | **80.5%** (+14.5 [+8.5, +20.5]) | 2.5% | 35 / 6 |
| Run 2 edit, normal (non-batch-invariant) kernels | 80.5% (+14.5 [+8.5, +21.0]) | 6.0% | 37 / 8 |
| Random edit, same 9,211 weights, same norm | 58.5% (−7.5 [−14.0, −1.0]) | 38.5% | 17 / 32 |
| Random edit, random positions, same norm | 70.5% (+4.5 [−1.5, +10.5]) | 24.0% | 24 / 15 |
| Run 1 edit | 72.0% (+6.0 [+0.0, +11.5]) | 20.5% | 23 / 11 |
| Original + presence penalty 1.5 | 85.5% (+19.5 [+14.0, +25.5]) | 3.5% | 41 / 2 |
| Run 2 edit + presence penalty 1.5 | 83.0% (+17.0 [+11.0, +23.0]) | 1.5% | 39 / 5 |

Run-to-run noise (five reruns of the original with normal kernels): accuracy 68.0–70.5%; each rerun fixed
17–20 and broke 11–16 problems. Run 2's +14.5 (35 / 6) is well outside that band.

**Not just the 200.** The fixed 200 were the problems used while the plan was written. On the other 300
MATH-500 problems, which nobody looked at: 72.7% → 82.7% (40 / 10, p = 2.4e-5), loops 19.7% → 4.0%.

**Random edits.** Rewriting the same weights at random hurts (−7.5); the same budget at random positions
does nothing significant. The gain comes from what the weights were set to, not from touching them.

### 5.3 Transfer beyond maths: MMLU-Pro

The editor built the edit only from maths transcripts, but it transfers: MMLU-Pro loops fall from about
half of all answers to about a tenth, and accuracy roughly doubles on both sets.

**Grader caveat.** The pre-registered MMLU-Pro grader accepts only a `\boxed{}` letter, so absolute scores
are understated for every arm (the original often writes "**Answer: G**"). A lenient re-grade (post hoc)
keeps the conclusion: fresh set 19.7% → 32.0% (57 / 20); original set 20.3% → 36.3% (60 / 12). Finished
answers go from 145 to 267 of 300. The gain is not a formatting artefact.

### 5.4 What it costs: time and tokens (post hoc, descriptive)

400 fresh GSM8K problems, greedy, batched on the laptop, with the loop cut-off turned off so every answer
runs until it ends or hits 16,384 tokens:

| Model | Correct | Hit the limit (no answer) | Output tokens | Wall clock |
| --- | --- | --- | --- | --- |
| Original | 72.25% | 95 | 1,905,760 | 6 h 48 min |
| Run 1 edit | 78.25% | 55 | 1,276,625 | 4 h 02 min |
| Run 2 edit | 84.25% | 9 | 735,252 | 59 min |

Accuracy is identical to the runs with the loop cut-off: no loop recovered when allowed to run on. Of the
original's 95 non-answers, run 2's edit answers 52 correctly; it hits the limit on 2 problems where the
original did not. The edit does not make each token faster (same model size); the saving is answers that
stop looping. The 6.9× wall-clock ratio depends on this hardware and batch, because long looping contexts
dominate it.

Single-stream race on gsm-665 (each model recorded alone, replayed on a shared clock in the video): TBD.

### 5.5 Cross-check: lm-evaluation-harness (pre-registered: "direction and rough size")

lm-eval 0.4.13, gsm8k_cot_zeroshot, chat template, reasoning stripped at `</think>`, greedy, 16,384 tokens,
first 400 GSM8K test problems:

| Model | flexible-extract | Wall clock |
| --- | --- | --- |
| Original | 71.75% | 4 h 51 min |
| Run 2 edit | 75.00% (28 / 15, p = 0.066) | 30 min |
| Run 1 edit | TBD | TBD |

**Direction agrees; size does not.** +3.3 points here vs +11.0 in our harness. Post hoc diagnosis:
- On the 126 problems both harnesses share, our harness gives 75.4% → 86.5% (18 / 4) and lm-eval
  72.2% → 74.6% (6 / 3). The difference is the harness, not the problems.
- lm-eval's prompt has no `\boxed{}` instruction. The original loops less under it (12% vs 20.5%). The edit
  still removes loops (48 → 5), but more of the rescued answers are wrong (wrong finished answers 74 → 96).
- This matches the editor's own warning that its "conclude" direction was learned from one closing style.
  The edit generalises across problems and subjects; it is less robust to a different prompt format.

## 6. Methodology

- **Pre-registration:** arms, metrics, tests, token cap, loop detector, random controls and example rules
  were committed before each editing run; amendments were committed before the results they cover.
  Everything chosen after seeing results is labelled post hoc.
- **Sealed test sets:** MATH-500, GSM8K (400), MMLU-Pro (300) and the fresh sets lived outside the
  editor's folder. Practice data was decontaminated against them before handoff (82 near-duplicates
  removed). After each run: no test question appears anywhere in the editor's logs, notes or transcript.
- **Audits:** no gradient code in either editor's scripts or commands; the model ran only through the
  budgeted runner; no writes to the budget file or rules.
- **Statistics:** exact McNemar, 95% bootstrap CIs over problems (10,000 resamples), Holm within each
  pre-registered family.
- **Controls:** two random edits per run with the same size and norm: one on the same weights, one at
  random positions.
- **Determinism:** plain greedy decoding in vLLM changes with batch composition, so every arm ran with
  batch-invariant kernels (reruns byte-identical). Run 2's edit gives the same 80.5% with normal kernels.

## 7. Caveats

- One editor (Claude) on one 1.5B model; two runs. Run 1 alone would have been a modest result.
- A decoding-time presence penalty is about as good and needs no weight change. The edit's value is that it
  lives in the weights (works with any decoder or server, no tuning knob) and shows how precise a
  hand-made change can be; it is not the only way to stop loops.
- Smaller gain under a different prompt format (§5.5) and under sampling.
- Automatic graders only; MMLU-Pro absolute scores are understated by the strict grader (§5.3).
- The loop detector flags verbatim repetition only.
- Timing results are post hoc measurements on one laptop.

## 8. Credits, links, reproduction

- Inspired by MIT CSAIL's "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab) and
  Sabhya Chhabria's Stable Diffusion replication.
- Code, logs, raw outputs, the editor's notebooks and final reports:
  https://github.com/Quantum-Coded/open-weights
- Edited model: TBD (Hugging Face, after sign-off).
- How to reproduce: `REPRODUCE.md`.
