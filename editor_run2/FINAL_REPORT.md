# Final report: weight edit to DeepSeek-R1-Distill-Qwen-1.5B

**Result:** two rewritten MLP neurons in layer 19 (9,215 weights). On problems never used to build the edit, accuracy goes from 63/96 to 83/96. Forward passes used: 30 of 50.

## 1. What was wrong
- **The model's math is fine; it fails because it does not stop.** In the first baseline set (MATH, greedy decoding, 8,192-token cap), 22/32 were correct. All 22 answers it finished were correct. All 10 failures were non-terminations: repetition loops or hitting the cap.
- **The loops are paragraph-level.** The model repeats "Wait, let me think again…" paragraphs word for word, often after it has already written the right answer (for example 1000 and 1440).
- **Over all three baseline sets:** finished answers were 39/41 correct on MATH and 24/25 on GSM8K. The remaining errors were almost entirely loops or cap hits.
- **The model can tell when it is looping.** Read passes at paragraph starts showed that a linear probe in the residual stream separates "loop restart" from "productive self-check (Wait)". Area under the ROC curve (AUC) is about 0.97, measured with each test problem held out of fitting; the data covered 29 loop problems. "About to conclude" states separate from everything else with AUC about 1.0.
- **A simple token bias cannot fix it.** At the final layer, loop restarts and productive self-checks predict the same next tokens: about 34% "Wait", and about 1e-6 for "**" (the start of **Final Answer**).

## 2. What I changed
The final patch is **v2**. It overwrites two low-importance neurons in the layer-19 MLP: neurons **3918** and **4028**. These were the two smallest by the product of their gate, up and down weight norms.

| Tensor | What changed | Weights |
|---|---|---|
| `model.layers.19.mlp.gate_proj.weight` | rows 3918 and 4028 | 3,072 |
| `model.layers.19.mlp.up_proj.weight` | rows 3918 and 4028 | 3,071 |
| `model.layers.19.mlp.down_proj.weight` | columns 3918 and 4028 | 3,072 |
| **Total** | | **9,215 (0.0005% of the model)** |

Each neuron computes silu(gate·x) × (up·x), so it acts as an AND of two conditions.

- **Gate row: a thresholded linear detector.**
  - The direction comes from linear discriminant analysis (LDA) with shrinkage 0.9. It is computed on the input to the MLP after layer norm, approximated from the layer-18 residual stream.
  - The threshold works as a bias. The detector subtracts τ times a constant direction c, whose projection is 1.000 ± 0.044 on every token. The result is scaled so a typical positive example gives a gate value of about 8.
- **Up row: a paragraph-start detector.** It reads about 1.0 ± 0.08 at paragraph starts and about 0 ± 0.07 elsewhere. So the neuron only acts at the points where the model decides how to begin the next paragraph.
- **Neuron 3918, "loop exit":**
  - It detects loop restarts as opposed to productive self-checks, with τ = 30.
  - Its down column writes the shift from the average loop-restart state to the average "just before **Final Answer**" state, both measured at layer 19 and divided by silu(8).
- **Neuron 4028, "close thinking":**
  - It detects "paragraph start right after a boxed Final Answer", as opposed to the other paragraph-start types. τ = 104.6 is the midpoint between the 5% quantile of positives and the 99% quantile of negatives.
  - Its down column writes the shift from the average repeated-block state to the average state just before `</think>`.
  - It was added because neuron 3918 alone sometimes made the model repeat "**Final Answer** \boxed{X}" instead of closing `</think>`. That happened in 4 of 96 fresh runs.

Everything is one-shot linear algebra: means, shrinkage LDA, a closed-form constant direction, and quantile thresholds. There were no gradients and no perturb-and-evaluate search. The constants (κ = 8, τ) were set once from class statistics.

Scripts: `work/make_edit1.py`, `work/make_edit2.py`, `work/an5.py`, `work/an6.py`, `work/an16.py`. Saved patches: `work/patch_v1.safetensors` and `work/patch_v2.safetensors`. The installed patch is v2.

## 3. Mechanism checks (read passes; logits computed from the layer-27 output)
| Decision point | Effect |
|---|---|
| Loop restarts | 45/90 switch from "Wait" or "But" to "**" (Final Answer). log p("**") goes from −13.8 to −4.5. |
| Productive "Wait", normal paragraphs, real conclusions | Unchanged. |
| Tokens inside a paragraph | Argmax flips at 13 of 1,350 positions in loop contexts; KL ≤ 0.003. |
| After a boxed answer + `\n\n` | `</think>` in 110/110 synthetic cases and 11/23 real repeated-block cases. |

## 4. Practice-set results (greedy, 32 problems per set)
Generation is deterministic: re-running v2 on the same set reproduced all 32 responses byte for byte. However, any weight change flips near-tied tokens and re-rolls borderline problems. Responses diverge after a median of about 800 characters, even between v1 and v2. So single-set differences of ±3 are noise; only the totals mean much.

**Fresh problems, never used to build anything (passes 22–30):**

| Set | base | v1 | **v2 (final)** |
|---|---|---|---|
| MATH seed 303 | 22 | 25 | 25 |
| MATH seed 404 | 21 | 29 | 28 |
| GSM8K seed 303 | 20 | 30 | 30 |
| **Total (of 96)** | **63** | **84** | **83** |
| Loops / caps | 26 / 9 | 6 / 3 | 1 / 3 |
| Repeated-Final-Answer outputs | 1 | 4 | 0 |
| Total tokens generated | 314k | 275k | 255k |

**All sets where base and v2 were both measured** (MATH seeds 7, 303 and 404; GSM8K seeds 101 and 303; 160 problems): **base 112 → v2 132 (70% → 82.5%)**.

**v1 alone** (not the final patch) on its other sets:

| Set | base | v1 |
|---|---|---|
| MATH seed 7 | 22 | 26 |
| MATH seed 11 | 17 | 25 |
| GSM8K seed 7 | 24 | 27 |
| MATH seed 101 | 25 | 29 |
| MATH seed 202 | 26 | 28 |
| GSM8K seed 101 | 27 | 29 |

MATH seeds 7 and 11 and GSM8K seed 7 were in-sample: their loop states were used to fit the detector.

**Passes used: 30 of 50.**
- 6 read passes: passes 2, 5, 6, 16, 17 and 18.
- 1 determinism check: pass 21.
- 23 generate passes.

## 5. Confidence, limitations, what might break
- **Confidence: high** that the edit substantially improves greedy accuracy on MATH- and GSM8K-style problems. The gain shows up on every fresh set, and its cause, loops being removed, is directly visible in the outputs.
- **v1 versus v2 is within noise on accuracy.** I chose v2 because it removes a degenerate output pattern and uses fewer tokens.
- **Premature stops.** At a loop restart the model now sometimes concludes after a single "But wait". If a real loop state is misread as a productive check, or the other way round, the model can stop too early or keep doubting. Of the few regressions, most look like ordinary trajectory re-rolls, not forced stops.
- **Long exploration is not addressed.** Most remaining failures are caps on hard problems where the model explores without repeating itself.
- **Detectors fitted on few examples.**
  - The loop detector was fitted on about 29 problems, using paragraph starts that begin with "Wait" or "But wait".
  - The close-thinking detector's "real repeated block" examples come from only 2 problems. Its held-out recall on those is 48% per occurrence, though the repetition gives it many chances.
- **Untested conditions:**
  - sampling (temperature > 0);
  - non-math tasks;
  - contexts longer than 8k tokens;
  - other output formats, for example a response without "**Final Answer**". The "conclude" direction was learned from that specific closing style.
- **Approximations:**
  - The detectors read the layer-18 residual stream as a stand-in for the actual input to the layer-19 MLP.
  - The read texts used the standard DeepSeek prompt suffix. That is a guess, because the grader's exact prompt is defined in a file outside the editor folder, which the rules forbid reading.
