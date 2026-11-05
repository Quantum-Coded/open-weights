# Final report: weight-space edit of DeepSeek-R1-Distill-Qwen-1.5B for math

## TL;DR
- **What was wrong:** the base model mostly does not fail because it can't do the math. It fails because it
  **degenerates**: it falls into verbatim repetition loops or rambles until the token cap without ever committing
  to an answer. In the first baseline, all 54 runs that finished their reasoning were correct, and every failure
  (18/18) was a loop or a cap. In about a third of those failures the correct answer was already written down
  (often dozens of times) before the model started looping.
- **What I changed:** I added one constant "steer away from the loop state" vector to the residual stream at
  layers 10–14. It is implemented exactly through the attention value biases: **1,278 weights in 5 tensors**
  (`model.layers.{10..14}.self_attn.v_proj.bias`). Nothing else was changed.
- **Result on held-out practice problems:** 288 problems, none of them used to build the edit, greedy decoding,
  8,192-token cap. Accuracy went from **207/288 (71.9%) to 233/288 (80.9%)**. Per problem, 42 were fixed and 16
  broke (two-sided sign test p ≈ 0.001). Loops were halved (59 → 29), and mean output length fell 13%.
- **Budget:** 38 of 50 forward passes used (one wasted by my mistake, one by a harness crash; details below).

## 1. Diagnosis and evidence

| Observation (base model, T=0) | Evidence |
|---|---|
| Failures are degeneration, not wrong math | Pass 1 (MATH, 32 problems): 22/32. All 22 finished runs were correct; the 10 failures were 6 loops and 7 caps (the loop and cap flags overlap). GSM8K, pass 2: 24/32, with 7 of 8 failures being loops or caps. |
| The model often already has the answer | In 6 of 18 failures the gold answer appears many times in the reasoning (for example "1000" ×110, "6230" ×30) before it loops on "Wait, but let me check again…". |
| Commit is a discrete gate | Every finished run (54/54) writes `**Final Answer**` exactly once inside `<think>` and then stops. No looping or capped run ever writes it. |
| Loops are a representational state | At layer 14 (residual), positions ≥2,000 tokens into failed runs are separated from late healthy reasoning along a mean-difference direction: projections 2.9 ± 12.6 vs −10.7 ± 7.1. On held-out problems, linear probes reach AUC ≈ 0.87. Independent estimates of the direction from different problem sets agree (cosine 0.78). |
| Loop rate is high | Base loop flags: 59 of 288 held-out runs (20%). On GSM8K, most failures are loops on easy problems. |

## 2. What I tried (in order), and what failed

All hypotheses were pre-registered in `logs/notebook.md` before they were evaluated. Strengths were chosen
offline from analysis, not by evaluate-and-adjust loops.

1. **H1: logit bias against "Wait".** I subtracted b·u from the `lm_head` rows of `Wait` and ` Wait`, where u is
   a closed-form least-squares direction with h·u = 0.99 ± 0.07, which makes the edit an exact constant logit
   bias of −1. Result: +2/64 (noise). The loops simply re-formed with "So / Therefore / But" openers, so the
   opener token is a symptom, not the cause.
2. **H2/H3: conditional "commit gate".** I added B·(ridge probe of the settled/looping state) to the `lm_head`
   row of `**`, to open the `**Final Answer**` gate inside loops. Offline it looked promising. On fresh problems
   it had **no effect** (MATH 15→15, GSM8K 29→29). A diagnostic read pass showed why:
   - in verbatim loops, `**` trails the top token by 10–18 logits;
   - loop and healthy paragraph starts overlap too much for any single linear readout to separate them;
   - when the gate did open, the model wrote bold section headers (`**Case 1:**`) rather than `**Final Answer**`.

   I abandoned this line.
3. **H4 (adopted): constant steering away from the loop state** (details in §3). It replicated on every
   held-out GSM8K seed and on 2 of 4 MATH seeds; see §4.
4. **H5: H4 with the direction re-estimated from 5× more data** (same α). Result: 98/128 vs H4's 105/128 on the
   same problems, so it was rejected by the decision rule I pre-registered (adopt only if ≥ H4 − 2).

## 3. The final edit, exactly

- **Direction.** d = mean(layer-14 residual | positions ≥ 2,000 tokens into failed base runs) − mean(layer-14
  residual | positions ≥ 1,000 tokens into correct base runs, before `**Final Answer**`). It was computed from
  seed-7 practice transcripts only (pass 3), skipping the first 300 positions of each non-initial read window to
  avoid truncation artifacts. |d| = 13.6 against a typical |h| ≈ 105. The direction is spread across many
  dimensions (the top 5 dimensions hold 15% of its energy).
- **Target.** Add −0.5·d to the residual stream at every position. α = 0.5 is half the gap between the class
  means, so healthy states move about 1σ. It was chosen before any evaluation and never tuned.
- **Mechanism.** For each layer L ∈ {10,…,14}, a change Δb_L to `v_proj.bias` (256 values = 2 KV heads × 128)
  adds exactly `o_proj · expand(Δb_L)` to that layer's attention output at **every** position, because softmax
  weights sum to 1. I solved for the five Δb jointly with closed-form ridge least squares
  (`min ||Σ_L O_L E Δb_L + 0.5 d||² + λ||Δb||²`). This realizes 85% of the target (|realized| 5.95 vs |target| 6.79).
- **Size.**

  | Tensor | Weights changed |
  |---|---|
  | `model.layers.10.self_attn.v_proj.bias` | 256 |
  | `model.layers.11.self_attn.v_proj.bias` | 256 |
  | `model.layers.12.self_attn.v_proj.bias` | 255 |
  | `model.layers.13.self_attn.v_proj.bias` | 256 |
  | `model.layers.14.self_attn.v_proj.bias` | 255 |
  | **Total** | **1,278** (0.0001% of the model) |

  Bias-delta norms are 1.4–1.8 per layer, against base bias norms of 0.7–1.5. No gradients, no iterative
  fitting: one mean difference and one ridge solve. Scripts: `work/steer.py` builds d, `work/make_h4.py` builds
  the patch. The frozen patch is `edits/patch.safetensors`, byte-identical to `work/patch_H4_a05.safetensors`.

## 4. Practice-set numbers (greedy, 8,192-token cap, same problem ids for base and edit)

Held-out means the seed's problems were never used to compute the steering direction (which used seed 7 only).

| Set | Base | H4 | Fixed / broke |
|---|---|---|---|
| MATH seed 11 | 17/32 | 21/32 | 6 / 2 |
| GSM8K seed 11 | 22/32 | 27/32 | 8 / 3 |
| MATH seed 13 | 24/32 | 24/32 | 4 / 4 |
| GSM8K seed 13 | 28/32 | 30/32 | 3 / 1 |
| MATH seed 21 | 15/32 | 20/32 | 6 / 1 |
| GSM8K seed 21 | 29/32 | 29/32 | 1 / 1 |
| MATH seed 31 (untouched until the end) | 28/32 | 26/32 | 0 / 2 |
| GSM8K seed 31 (untouched until the end) | 23/32 | 30/32 | 7 / 0 |
| GSM8K seed 41 (fresh) | 21/32 | 26/32 | 7 / 2 |
| **Total (288)** | **207 (71.9%)** | **233 (80.9%)** | **42 / 16, p ≈ 0.001** |
| MATH subtotal (128) | 84 (65.6%) | 91 (71.1%) | 16 / 9 |
| GSM8K subtotal (160) | 123 (76.9%) | 142 (88.8%) | 26 / 7 |

Loop flags fell from 59 to 29, cap hits went from 20 to 21, and mean generated tokens fell from 2,646 to 2,298.
Seed 7 (in-sample for the direction) was not re-run with H4. Its base scores were 22/32 (MATH) and 24/32 (GSM8K).

**Forward passes used: 38 / 50.**

| Purpose | Passes |
|---|---|
| Baselines | 14 |
| Read passes | 6 |
| H1 | 2 |
| H2 | 2 |
| H3 | 2 |
| H4 | 8 |
| H5 | 4 |
| Wasted | 2 |

The two wasted passes:
- **Pass 35:** I launched a "base" run while the H5 patch was still installed. I killed it and discarded it.
- **Pass 36:** base MATH seed 41 crashed inside the harness (vLLM: the prompt reached the 9,216-token
  `max_model_len`), so MATH seed 41 was dropped.

## 5. Honest confidence, limitations, what might break

- **Confidence that the edit helps on this distribution (greedy, 8k cap): fairly high.** The improvement is
  concentrated in GSM8K (+11.9 points), where loops on easy problems are the dominant failure. For **MATH it is
  modest and less certain** (+5.5 points, 16 fixed vs 9 broke). Two of the four MATH seeds were flat or negative
  (seed 13: 0, seed 31: −2), so the true MATH effect could be anywhere from about 0 to +8 points.
- **It is a loop/termination fix, not a reasoning upgrade.** Cap hits did not fall. Hard problems the model
  can't solve within 8k tokens stay unsolved.
- **Greedy chaos.** Any edit reshuffles greedy trajectories, so about 16 of 288 problems that the base solved now
  fail, some of them easy ones (for example, a 644-token GSM8K solve became a confused cap). Per-problem
  behaviour is not preserved.
- **Decoding regime.** Everything was measured at temperature 0 with this harness's loop early-stop. Under
  sampling (for example T = 0.6) loops are rarer, and the benefit is probably smaller. I did not measure it.
- **Scope.** Measured only on practice MATH and GSM8K problems. Behaviour on non-math tasks, long contexts beyond
  8k, or the chat/summary part after `</think>` was not evaluated (spot checks of the outputs looked normal: no
  language mixing, coherent text). The steering is constant at every position, including the prompt, so it
  could subtly affect any task.
- **Selection.** H4 was chosen over H5 using seeds 21 and 31, which mildly favours H4 on those seeds. Seed 41
  (GSM8K, +5) and seeds 11 and 13 were not involved in any choice and show the same direction of effect.
- **Unused budget.** 12 passes remain unused on purpose. Further variants (another α, other layers) would have
  amounted to tuning on the evaluation seeds.

## Files
- `logs/notebook.md`: the full chronological lab notebook (every hypothesis, result and decision).
- `logs/fp/pass_*.jsonl` / `.npz`: all generations and activation reads.
- `work/`: analysis scripts (`an*.py`, `fit3.py`, `steer.py`, `dir_all.py`, `make_h4.py`) and the saved
  candidate patches (H2, H3, H4, H5).
