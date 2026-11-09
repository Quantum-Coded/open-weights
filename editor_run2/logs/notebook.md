
### 2026-11-09 03:59
Session start. Budget 0/50. Config: Qwen2 1.5B, 28 layers, d=1536, ffn=8960, untied lm_head.
Weight stat scan (mean/std/max per tensor per layer): nothing grossly anomalous. Large L0/L1 k_proj biases (max 402) are the usual Qwen2 pattern. Mild outliers: L14 input_layernorm max 5.84, L1 post_attn_ln max 7.03.
Hypothesis H0: dominant failure mode under greedy decoding is repetition loops / never closing </think> (cap hits), not wrong math. Baseline pass will test this.

### 2026-11-09 04:17
Pass 1 (baseline, math_train n=32 seed 7, greedy, 8192 cap): 22/32. loop 6, cap 7, finished 22. ALL 22 finished answers correct; ALL 10 failures are non-termination (greedy repetition loops of "Wait, let me think again..." paragraphs, often with the right answer already stated, e.g. 1000, 1440).
=> Diagnosis: model's math is fine when it terminates; failure is degenerate looping / never closing </think>.
Pass 2 (read, 207 decision-point texts, resid L6..27 last:16). Classes: A=paragraph start right before **Final Answer** (finished runs), B=Wait/But paragraph starts in loop region, C=Wait paragraph starts in finished runs, D=other paragraph starts.
Findings: L27 resid is post-final-norm (teacher-forced greedy match 95%). At final layer B and C have same next-token profile (Wait logp -1.1/-1.2, ** logp -13.8/-14.2) -> plain logit bias cannot separate loop from useful doubt.
Leave-one-problem-out shrunk-LDA AUC: B vs C 0.87-0.935 (best L18), A vs rest ~0.99-1.0 at all layers >=10. Model represents both "looping" and "ready to conclude".
Plan: get more loop data (math seed 11 + gsm8k seed 7 baselines), then find loop-selective MLP neurons and wire them (down_proj columns) to write the conclude direction = conditional rank-k edit.

### 2026-11-09 04:47
Edit v1 (loop-exit neuron): overwrote neuron 3918 of L19 MLP (lowest |gate|*|up|*|down| importance). gate row = 8*(w_loop - 30*c)/(medB-30) where w_loop = shrunk-LDA(0.9) B-vs-C direction on post_attn_ln input approx (from r18, pass 5), c = low-variance constant direction (proj 1.000+-0.044) used as bias. up row = centered paragraph-start detector (1.00+-0.08 at para starts, 0+-0.07 in-paragraph). down col = (mean_A - mean_B of r19)/silu(8). 4,608 weights.
Pass 6 (read pass-2 texts, L27 logits base vs edit): B decision points 45/90 now emit ** (Final Answer); logp(**) -13.8 -> -4.5. C unchanged (Wait 36/51; ** 2 = same as base). A, D unchanged. In-paragraph argmax flips 12/1350 (B), ~0 elsewhere; KL<=0.003. Mechanism works as designed in-sample. Next: generate test.

### 2026-11-09 05:31
Generate results with edit v1 (paired vs base, IN-SAMPLE problem sets: their loop states were used to fit w_loop):
- math seed7: 22->26/32 (loop 6->0, cap 7->4). 6 fixed, 2 regressed (geometry-519 now cap w/o loop; int_alg-210 different path, wrong). Only 1/32 responses byte-identical -> greedy chaos from tiny perturbations.
- math seed11: 17->25/32 (loop 10->0). 8 fixed, 0 regressed.
- gsm seed7: 24->27/32 (loop 6->1). 3 fixed, 0 regressed.
Exits look natural: "...So 1000 is the answer.

**Final Answer**...\boxed{1000}". Saved copy work/patch_v1.safetensors.
Next: held-out sets (math seeds 101, 202; gsm seed 101), edit and base.

### 2026-11-09 06:26
Held-out sets with edit v1 (problems never used for fitting): math seed101 29/32 (loop1 cap1), math seed202 28/32 (loop0 cap3), gsm seed101 29/32 (loop1 cap1). Base on same sets running (passes 13-15).
New edit-induced failure mode seen 2x in ~190 edited runs (0x in base): after **Final Answer** \boxed{X} the model writes \n\n instead of \n</think>, then neuron-1 pushes **Final Answer** again -> repeats the block until cap (gtrain-4931 x80, geometry-759 x11).
Plan v2: second neuron at L19: gate = detector of F (para start right after boxed Final Answer; synthetic F = finished run with \n</think> replaced by \n\n) vs other para starts; up = same para detector; down = mean(E) - mean(F), E = natural state right before </think>.

### 2026-11-09 07:38
Base on held-out: math101 25/32 (edit 29), math202 26/32 (edit 28), gsm101 27/32 (edit 29). Paired held-out: 11 fixed, 1 regressed (precalculus-692 -> cap). Patch v1 restored to edits/patch.safetensors. 15/50 passes used. v2 (Final-Answer-repeat fix) prepared (work/build_read16.py, work/read16_texts.json) but NOT run: session stopped at usage limit. FINAL_REPORT.md written.

### 2026-11-09 09:02
v2: added neuron 4028 at L19 (second-lowest importance). gate = 8*(w_F - tau*c)/(med-tau), w_F = shrunk-LDA(0.9) of F (para start right after boxed Final Answer: 110 synthetic + 23 real) vs A/B/C/D para starts (pass 5), tau=104.6 (midpoint of pos q05 / neg q99). LOPO: synthetic F 100% above tau, real repeats 48%, negatives A/C/D 0%, B 0.9%. up = para detector, down = (mean E - mean Freal at r19)/silu(8). Total patch 9,215 weights.
Pass 16 (read, v1): synthetic F mostly continues with summary (\[), real repeats predict ** (logp -0.67) because loop neuron fires (score ~101).
Pass 17/18 (read, v2): F -> </think> 110/110, Freal 11/23 -> </think>; E unchanged; A/B/C/D identical to v1. Next: generate gsm101 + math7 with v2.

### 2026-11-09 09:20
v2 generate: gsm101 26/32 (v1 29, base 27); math7 23/32 (v1 26, base 22). FA-repeat failure gone in both (gtrain-4931 now fails differently; base also fails it). Regressions are normal single-FA conclusions with different reasoning.
Pass 21: re-ran v2 gsm101 -> 32/32 byte-identical: vLLM greedy is deterministic. So divergences (median common prefix ~800 chars even v1 vs v2) come from tiny weight perturbations flipping near-ties: every weight change re-rolls borderline problems. Per-set +-3 is noise; v1 numbers may include selection luck.
Plan: fresh sets math303, math404, gsm303 under base, v1, v2 (9 passes) for an unbiased 3-way comparison.

### 2026-11-09 10:55
Pass 22-30 three-way on FRESH sets (math303, math404, gsm303; 96 problems): base 63 (loop 26, cap 9), v1 84 (loop 6, cap 3), v2 83 (loop 1, cap 3). FA-repeat outputs: base 1, v1 4, v2 0. Tokens: v2 255k, v1 275k, base 314k.
Decision: final = v2 (accuracy tied with v1, removes degenerate FA-repeat, fewer tokens). All v2-vs-base paired sets (math7, gsm101, math303, math404, gsm303; 160 problems): base 112 -> v2 132. Stopping at 30/50 passes; remaining failures are long non-repetitive explorations on hard problems, not addressable by this mechanism without risk.
