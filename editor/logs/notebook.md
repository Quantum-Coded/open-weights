
### 2026-11-05 03:08
Session 1 start. Plan: (1) baseline on fixed MATH/GSM8K practice sets at T=0 to measure accuracy, loop and cap rates; (2) diagnose failure modes from transcripts; (3) find directions via read passes and edit with one-shot projections; (4) paired re-eval on the same problem ids.

### 2026-11-05 03:20
Pass 1 baseline MATH (math_train n=32 seed=7, T=0, 8192 cap): 22/32. loop 6, cap 7, finished 22.
FINDING: all 22 finished answers are correct; ALL 10 failures are degeneration (loop or cap, no </think>). Several looping runs already had the right answer (nt-544 -> 1000, cp-252 -> 1440) but kept re-verifying "Wait, but let me think...". Failed runs have 16-159 paragraphs starting with Wait/But vs typically 0-8 for successes.
HYPOTHESIS: the bottleneck is a hesitation/re-verification attractor + inability to terminate, not math knowledge. Lever: the next-token decision at paragraph starts (Wait/But vs conclude), possibly conditional on a loop-state direction.

### 2026-11-05 03:39
Pass 2 baseline GSM8K (gsm8k_train n=32 seed=7): 24/32, loop 6, cap 2; 7/8 failures are loop/cap. Same failure mode.
Pass 3 read: 233 windows (1000 tok) of pass1+2 transcripts, resid L14,L27 all positions. L27 resid = final-normed hidden (offline logits h@lm_head reproduce greedy choice 90%).
Findings: paragraph-start top token in failed runs: Wait 23%, But 8.4%, Alternatively 5% vs correct runs 9%, 2%, 2%. **-conclusion token never close (gap>7 in 90%). Loop state linearly decodable only moderately (held-out-problem AUC 0.87 L14, 0.88 L27).
Method: fitted u (ridge LS, lam=1e-2) with h.u = 0.99+-0.067 held-out => subtracting b*u from lm_head row t = near-constant logit bias -b on token t. Exact, closed-form, rows only.
H1 (pre-registered): bias -1.0 on Wait(14190) and " Wait"(13824). 2,978 weights changed (2 rows of lm_head). Verified shift -0.99+-0.07 after bf16. Evaluate on same MATH seed7 + GSM8K seed7 (paired).

### 2026-11-05 03:54
Pass 4 (H1 MATH s7): 23/32 (base 22), loop 3 cap 6, mean tokens -10%. Large churn: fixed 4, broke 3. Loops reappear with So/Therefore/But openers -> hesitation token is a symptom; the attractor is paragraph-level self-repetition.
Pass 5 (H1 GSM8K s7): 25/32 (base 24), loop 6. H1 net +2/64: within noise. Not adopted alone.
KEY FINDING: every finished run (54/54) writes "**Final Answer**" once inside <think> then ends; no loop/cap run ever writes it. The "**" token (334) at a paragraph start is the commit gate; in loops it is never competitive (gap to top: median 16, p5 7.5).
Uniform ** boost rejected: healthy pre-commit paragraph starts are closer to ** than loop ones, so it would commit early in healthy runs first.
H2 (pre-registered): conditional commit gate. Ridge LS probe w (lam 1e-2, intercept via u) on L27 final hidden: target 1 at paragraph starts >=2000 tok into failed runs, 0 at healthy pre-commit paragraph starts and random positions. Window-start artifact found and removed (first 300 positions of non-initial windows excluded). 4-fold by-problem CV: B=10 -> 9/18 failed runs get a commit point (GSM8K loops at tok 2300-3800, answer already found), 2/43 healthy runs commit >300 tok early; B=14 -> 17/18 vs 3/43. Chose B=12 offline. Edit: lm_head[334] += 12*(w + b*u). Side effect check: new ** argmax at 15/66k healthy positions.

### 2026-11-05 04:11
Pass 6 (H2 MATH s7): 23/32 (base 22); only 2 outcome changes (cp-252 loop->correct; alg-596 cap->loop). Surgical, but weak.
Pass 7 (H2 GSM8K s7): 24/32 (base 24). Fixed gtrain-704 (loop->correct), broke gtrain-484: the ** boost leaked to a mid-sentence position after a quote mark, a bold quote changed trajectory, then a loop. => greedy chaos noise ~ +-2/32 per edit; probe must be more specific to paragraph starts.
Ceiling estimate: 6/18 base failures had the gold answer written many times already (cp-252, nt-544, nt-151, gtrain-3234, -5332, -704) -> commit-gate approach can at most give ~+6/64.
Decision: H2 at B=12 is too weak in real generation (+1/64). Need more diverse failure data. Reset to base, run base MATH+GSM8K seeds 11 and 13 (also enlarges the baseline for final comparison). H2 patch kept at work/patch_H2_B12.safetensors.

### 2026-11-05 04:39
Passes 8-11 (BASE, T=0): MATH s11 17/32 (loop10 cap6), GSM8K s11 22/32 (loop10), MATH s13 24/32 (loop5 cap3), GSM8K s13 28/32 (loop4).
BASE TOTAL over seeds 7,11,13: MATH 63/96, GSM8K 74/96, all 137/192 = 71.4%.

### 2026-11-05 04:54
Passes 12-13: read L27 on 505 overlapping windows (stride 700, len 1000) of base passes 8-11 (300 failed-run windows). Dataset: 30k positions (all para starts + 4% random) from passes 3,12,13, first 300 positions of non-initial windows dropped.
H3 probe: class-balanced ridge LS (lam 1e-2 rel.), target 1 = para starts >=2000 tok into failed runs AND the actual commit points of healthy runs (settled-answer state); 0 = healthy para starts >150 tok before commit + random non-para positions. 5-fold by problem out-of-fold: inline (non-para) flips ~0/18.7k for B<=12.
Offline net estimate (failed runs whose first commit comes after gold already appears minus healthy runs committing before their final answer appears), out-of-fold: B6 +2, B8 +5, B10 +5, B12 +12, B16 +9 over 176 runs. Pre-registered H3 = B12. Since seeds 7/11/13 were used to fit, honest test = FRESH seed 21: base vs H3 on MATH+GSM8K (4 passes).

### 2026-11-05 05:58
Passes 14-17 (FRESH seed 21): BASE MATH 15/32 (loop11 cap8), GSM8K 29/32. H3 (probe gate B12): MATH 15/32 (loop11 cap7), GSM8K 29/32. => H3 has NO effect on held-out problems.
Pass 18 diagnostic read of H3 seed-21 failures (hidden states identical to base since only lm_head changed): probe DOES fire in loops (s 0.8-1.0) but in verbatim loops ** trails top by 10-18 logits (min over loop needs B 11-17); where the gate opened, the model wrote bold headers (**Case 1:**, **Option A**, **So**) not **Final Answer**: after ** in failed runs, Final trails by median 11.8. In healthy runs real commits have s=0.28 at the ** position.
CONCLUSION: a linear lm_head gate cannot separate loop vs healthy paragraph starts well enough (pos 0.86 vs neg 0.38 mean, gaps ~15+-4 in both). Abandon gate line (H2/H3). Patch currently = H3; will replace.

### 2026-11-05 05:59
H4 (pre-registered): constant residual steering away from loop state. d = mean(L14 resid | failed runs, pos>=2000) - mean(L14 resid | healthy runs, pos>=1000, pre-commit), from pass 3 (|d|=13.6 vs |h|~105; distributed over dims: top5 dims=15% energy). Projection: loop 2.9+-12.6 vs healthy-late -10.7+-7.1. Steer -0.5*d (half the gap; healthy shifted ~1 sigma). Realized exactly as constant attention output via v_proj.bias of layers 10-14 (softmax weights sum to 1 => o_proj(expand(db)) is added at every position); ridge LS fit realizes 85% of target. 1,278 weights in 5 bias tensors. Eval on held-out seed 21.

### 2026-11-05 06:11
Pass 19 (H4 alpha0.5, MATH s21 held-out): 20/32 vs base 15/32; loop 11->5, cap 8->8. Fixed 6 (5 former loops), broke 1 (prealgebra-1129 -> cap). First clear signal.

### 2026-11-05 06:16
Pass 20 (H4, GSM8K s21): 29/32 vs base 29/32; fixed gtrain-3039 (loop), broke gtrain-1769 (easy, 644 tok -> confused reading -> cap). Mean tokens 1225->1592. Text quality normal (no CJK, coherent). Replicating H4 on seeds 11,13 (held-out for the steering direction, which used seed 7 only): passes 21-24.

### 2026-11-05 06:41
Passes 21-24 (H4 replication): MATH s11 21/32 (base 17), GSM8K s11 27/32 (base 22), MATH s13 24/32 (base 24), GSM8K s13 30/32 (base 28).
H4 HELD-OUT SUMMARY (seeds 11,13,21; 192 problems; direction fit on seed 7 only): base 135/192 (70.3%) -> H4 151/192 (78.6%), +16. Paired: fixed 28, broke 12, two-sided sign test p=0.017. Loops 42->21, mean tokens 2863->2559.
Decision: H4 is the working edit (saved work/patch_H4_a05.safetensors). Next: (1) untouched confirmation seed 31 base vs H4; (2) H5 = same alpha 0.5, direction re-estimated from all base failures (seeds 7,11,13,21) to reduce seed-7 overfit; needs L14 reads on BASE model.

### 2026-11-05 06:47
Passes 25-26: base-model L14 reads of 474 windows (258 failed-run windows at pos>=2000, 218 healthy windows) from base passes 8-11,14,15. Pooled loop direction d_all (206k loop / 62k healthy-late positions incl. pass 3): |d_all|=11.96; cos(d_seed7, d_seeds11/13/21)=0.78 (independent estimates agree), cos(d_seed7,d_all)=0.89. H5 = same alpha 0.5, direction d_all, same layers 10-14 v_proj.bias.

### 2026-11-05 07:07
Passes 27-30 (UNTOUCHED seed 31): base MATH 28/32, GSM8K 23/32 (loop 7). H4: MATH 26/32 (broke 2: cp loop, IA cap; no fixes), GSM8K 30/32 (fixed 7, 6 of them loops; mean tok 1921->1191). Net +5/64.
H4 over all held-out seeds 11,13,21,31 (256 problems): base 186 -> H4 207.
H5 pre-registered decision rule: evaluate H5 (d_all, alpha .5) on seeds 21+31 (128 problems, H4 already measured: 20+29+26+30=105). Adopt H5 if acc >= H4-2 (more data -> less overfit prior); else keep H4.

### 2026-11-05 07:34
Passes 31-34 (H5 = pooled direction d_all, alpha .5): MATH s21 19, GSM8K s21 28, MATH s31 24, GSM8K s31 27 = 98/128 vs H4 105/128 vs base 95/128. By pre-registered rule (H5 >= H4-2) -> REJECT H5, keep H4.
Note: H4 (seed-7 direction) beats the pooled direction; cannot tell if this is noise (diff 7/128) or the seed-7 direction being more loop-specific. Either way H4 is the measured one.
Final: restore H4; fresh seed 41 base vs H4 (passes 35-38) for one more unbiased estimate.

### 2026-11-05 07:37
MISTAKE: pass 35 was launched as "base seed 41" but edits/ still held the H5 patch (forgot reset). Killed mid-run; no output used. 1 pass wasted. Rerunning with explicit reset.

### 2026-11-05 07:45
Pass 36 (base MATH s41) CRASHED inside the harness generator: vLLM ValueError, decoder prompt length 9216 > max_model_len 9216 (a sample hit the context limit). Base model, not an edit effect. Pass wasted; MATH s41 dropped from the plan (H4 on it would crash the same way). Pass 37 = base GSM8K s41; next H4 GSM8K s41.

### 2026-11-05 07:53
Passes 37-38 (fresh seed 41, GSM8K only since MATH s41 crashes the harness): base 21/32 (loop 9) -> H4 26/32 (loop 4); fixed 7 broke 2.
FINAL HELD-OUT TALLY for H4 (seeds 11,13,21,31 both sets + GSM8K s41; 288 problems, none used to fit the direction): base 207/288 (71.9%) -> H4 233/288 (80.9%), +9.0 pts. Fixed 42, broke 16, two-sided sign test p=0.0009. Loops 59->29, caps 20->21, mean tokens 2646->2298. MATH 84->91/128 (fixed 16, broke 9: weaker, 2 of 4 MATH seeds flat/negative); GSM8K 123->142/160 (fixed 26, broke 7).
DECISION: freeze H4 (edits/patch.safetensors == work/patch_H4_a05.safetensors). Stop with 12 passes unused: further variants would amount to tuning on the eval seeds.
