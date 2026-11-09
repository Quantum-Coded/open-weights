# RUNLOG

Dated log of every command that matters: runtime and headline output. Becomes the methods section.

## Environment (2026-11-04)
- Host: Windows 11, NVIDIA RTX 4050 Laptop GPU (6 GB), driver 592.82 (CUDA 13.1 driver API), 16 GB RAM.
- WSL2 Ubuntu 24.04.4, Python 3.12.3; WSL memory limit currently 7.6 GB (default; see note below).
- torch 2.10.0+cu128, vllm 0.19.1, transformers 5.22.1, datasets 5.1.0, math-verify 0.9.0, pillow 13.0.0,
  imageio-ffmpeg (ffmpeg 9.0.2 static build, libx264). Full pin list: requirements.lock.txt.
- Project lives at ~/wsp-r1 inside WSL (ext4), not /mnt/c, for I/O speed.

## Log
- 2026-11-04 Kit unpacked from wsp-r1-starter-kit.zip; file modes reset to 644 (zip had them read-only).
- 2026-11-04 pip install -r requirements.txt (+ imageio-ffmpeg, pillow for the demo video). OK.
- 2026-11-04 Verified: torch.cuda.is_available() True (RTX 4050 Laptop GPU); vllm 0.19.1 imports.
- 2026-11-04 PLAN.md Phase 6 video spec revised by Dev (30 s loopable cut + 60 s explainer); make_demo_video.py written.
- NOTE: wsp.load_base() builds a ~7 GB fp32 copy; WSL has 7.6 GB. Needs a larger WSL memory limit
  (.wslconfig) before the editor run (Phase 5).
- 2026-11-04 PLAN.md: added §6 Reproducibility and stability (Dev) and §7 GitHub; eval_plan.md template
  gained the noise-floor procedure and cross-checks. Repo: https://github.com/Quantum-Coded/open-weights (push per phase).

## Phase 1: smoke test (2026-11-04)
- python prepare_data.py (1m28s): sealed math500=500 gsm8k=400 mmlu_pro=300; practice math_train=7498
  (dropped 0 overlaps with MATH-500 by normalized-text hash) gsm8k_train=7473.
- python apply_patch.py --out work/base (1m47s, incl. download). Model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,
  HF revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562; work/base/model.safetensors 3,554,214,752 bytes.
- FIX gen.load_llm: gpu_memory_utilization 0.90 -> 0.80 (env WSP_GPU_UTIL overrides). vLLM refused to start:
  only 4.95/6.0 GiB free at startup (Windows desktop holds ~1 GB), 0.90 asks 5.4 GiB; 0.85 would ask 5.1.
  Result: KV cache 0.84 GiB = 31,584 tokens (plan assumed ~50k), ~1.8 concurrent 17k-token sequences.
- python eval.py --model work/base --name smoke --set practice:math_train --limit 5 --max-tokens 4096 (2m24s):
  accuracy 0.20, loop_rate 0.20, cap_rate 0.80, mean_tokens_finished 1039.
  Hand check: prompt ends "<｜Assistant｜><think>\n" (template adds it; no double tag). 1/5 finished with
  </think> + \boxed{-\dfrac{2}{3}}, graded correct vs gold -\frac{2}{3} (agree). 4/5 hit the 4,096 cap with no
  </think> -> graded wrong (agree). mtrain-prealgebra-662 flagged loop at 4,096 (to examine in the pilot).
  Harmless warning: transformers notes tokenizer_class LlamaTokenizerFast vs ByteLevel tokenizer.json.
- 2026-11-04 PLAN.md Phase 9: blog article on devgaglani.in/blog added as a deliverable (Dev). Site: Substack removed
  from the blog page (portfolio-site commit 0d84ace; lint + build clean, /blog checked locally).

## Phase 2: pilot (2026-11-04)
- python eval.py --model work/base --name pilot --set practice:math_train --limit 20 --max-tokens 16384
  --no-early-stop (19m11s): accuracy 0.55, loop_rate 0.30, cap_rate 0.35, mean_tokens_finished 3098.
  Loop detector false positives (flagged but finished): 0/20. Tokens spent after the first loop flag:
  72% of all generated tokens (what early stopping would save). Go criterion (loop rate >= 15%) met.
- Practice baseline, same 20: --presence-penalty 1.5 (name pilot_pp, early stop on, 8m53s): accuracy 0.80,
  loop_rate 0.00, cap_rate 0.05, mean_tokens_finished 4379. A decoding-time fix is a strong baseline the
  weight edit should be compared against.
- FINDING: greedy decoding is not reproducible across batch composition. smoke (batch 5) and pilot
  (batch 20) outputs for the same 5 problems diverge at tokens 66-517 (bf16 near-ties in the logits).
  So "greedy" alone does not make a run repeatable; outputs depend on what else is in the batch.
- Test: VLLM_BATCH_INVARIANT=1 (4,096 cap, --no-early-stop): bi5a (5 problems, 206 s), bi20 (20, 262 s),
  bi5b (5, rerun, 163 s). bi5a == bi5b byte for byte, and the 5 shared problems in bi20 match bi5a byte
  for byte. KV cache 27k tokens in this mode (31.6k without). Cost ~15-40% throughput.
- gen.load_llm: attention_backend pinned to FLASH_ATTN (what auto-selection already picks here); batch-invariant
  mode refuses to start without an explicit backend.
- Projected runtime per 200-problem arm (16k cap, early stop): ~2.3 h; ~3 h in batch-invariant mode.
- Demo video redesigned (Dev): 1920x1080 landscape, light neobrutalist look, bottom illustration strip
  (Claude -> 28-layer network with the edited layers lit -> output loop/answer), standalone strip GIF (--gif).
  Test renders on practice data only (pilot vs pilot_pp, layers are placeholders): demo/TEST_*.
  Pane text falls back to DejaVu Sans Mono for characters Consolas lacks.

## Phase 3: pre-registration (2026-11-04)
- Checkpoint A decision (Dev): every run uses vLLM batch-invariant kernels; the noise-floor reruns
  (base_rerun, edit_rerun) use normal kernels (VLLM_BATCH_INVARIANT=0). PLAN.md §6 and Phases 4/7 updated.
- gen.load_llm now sets VLLM_BATCH_INVARIANT=1 unless the environment says 0 (covers eval.py and editor fp.py);
  eval.py writes "batch_invariant" into each summary. Check: bidefault (no env var, 5 practice problems, 4,096
  cap, 2m43s) is byte-identical to bi5a, 5/5; summary batch_invariant=true.
- eval_plan.md filled in: token cap 16,384, detector unchanged (pilot false positives 0 of 6 flagged),
  fast track MATH-500 --limit 200, kernel mode, noise-floor procedure and decision rule (confirmed by Dev).
- Pre-registration commit: ea01b035c4ff76ad8a6116a635b24607d45988aa (eval_plan.md is read-only from here).

## Phase 4: baselines (2026-11-04)
- Started base / base_pp / base_rerun at 07:32 UTC; stopped by Dev a few minutes in, during the first base
  batch (no results file written). Dev will say when to start Phase 4. Partial log kept as
  logs_base_stopped_20261104.txt. Note: that run's log overwrote the earlier logs_base.txt (apply_patch output);
  its runtime and the HF revision hash are recorded above under Phase 1.
- Approved by Dev: raise the WSL memory limit to 12 GB in .wslconfig once the baselines finish (needs a WSL
  restart, so not while a run is going).
- Restarted by Dev 10:47 UTC. Fast track, MATH-500 --limit 200, greedy, 16,384 cap, early stop on:
  - base (batch-invariant, 64m19s): accuracy 0.660, loop_rate 0.315, cap_rate 0.010, mean_tokens_finished 2348.
  - base_pp (presence penalty 1.5, batch-invariant, 94m16s): accuracy 0.855, loop_rate 0.035, cap_rate 0.060,
    mean_tokens_finished 3013.
  - base_rerun (normal kernels, VLLM_BATCH_INVARIANT=0, 50m52s): accuracy 0.705, loop_rate 0.265, cap_rate 0.030,
    mean_tokens_finished 2503.
- NOISE FLOOR (base vs base_rerun, analyze.py): accuracy 66.0% -> 70.5% (+4.5 pts, 95% CI [-1.0, +10.0]),
  McNemar p 0.15; correct->wrong 11, wrong->correct 20 (31 grade flips of 200); loop rate 31.5% -> 26.5%
  (-5.0 pts, CI [-11.0, +1.0]); loop-flag flips 38; response text differs 198/200.
  Under the pre-registered rule (net fixed - broken > 2 x the larger rerun flip count) the edit would need a
  net gain above 62 problems (+31 pts) if edit_rerun is no smaller. Raised with Dev before the editing run.
- Runtimes were ~1 h per arm, not the projected ~3 h: early stop ends looping answers and finished answers
  average ~2.4k tokens.
- WSL memory raised (approved by Dev): C:\Users\Dev Gaglani\.wslconfig created with [wsl2] memory=12GB
  (host 15.6 GB), wsl --shutdown. Now: Mem 11 GiB total, swap 3.0 GiB; GPU still visible.
- Amendment 1 (Dev approved): eval_plan_amendment.md adds 4 more normal-kernel base reruns (order-shuffled,
  eval.py --order-seed) and rule (b'): edit net fixed - broken > largest |net| of any rerun. eval.py gained
  --order-seed (shuffles feed order only; results file sorted by id, so existing commands are unchanged).
- Amendment 1 commit: b6f9860ffe34f9f2c85b0009a8ab5bbce76fa7ca.
- Amendment 1 reruns (normal kernels, --order-seed 2..5; 44m40s, 47m32s, 46m29s, 46m42s), vs base (66.0%, loops 31.5%):

  | rerun | acc | loop | wrong->right | right->wrong | net | text differs | loop flips |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | base_rerun | 0.705 | 0.265 | 20 | 11 | +9 | 198 | 38 |
  | base_rerun2 | 0.680 | 0.285 | 20 | 16 | +4 | 195 | 44 |
  | base_rerun3 | 0.685 | 0.275 | 17 | 12 | +5 | 198 | 34 |
  | base_rerun4 | 0.685 | 0.280 | 19 | 14 | +5 | 197 | 41 |
  | base_rerun5 | 0.695 | 0.270 | 19 | 12 | +7 | 199 | 39 |

  N (largest |net| over the base reruns) = 9, so rule (b') needs the edit's net fixed - broken > 9 (unless
  edit_rerun's |net| is larger). Original rule (b): > 2 x 31 = 62. Of 200 problems, 97 are correct in all 6 runs,
  34 wrong in all 6, 69 change grade in at least one. Rerun pairs share identical text on 23 of 2000 answers.
- OBSERVATION: all 5 normal-kernel reruns score above the batch-invariant base (+2.0 to +4.5 pts; loops -3.0 to
  -5.0 pts). Five of five in the same direction is unlikely by chance (one-sided sign test p = 1/32), so the
  batch-invariant kernels may cost ~3 pts on this model. Comparisons stay fair (every arm uses the same kernels),
  but the write-up should say that absolute numbers are for batch-invariant kernels.

## Phase 6: tooling while waiting (2026-11-05, CPU only, practice data)
- Claude Code 2.1.318 installed in WSL (official installer, ~/.local/bin, Dev approved) for the editor session.
- matplotlib 3.11.2 added to the venv (new packages only, nothing upgraded); requirements.txt and
  requirements.lock.txt updated.
- figures.py: fig1 accuracy/loop rate per arm with 95% CIs and the rerun noise band, fig2 projection (JSON input
  after freeze), fig3 fixed vs broken. Statistics imported from analyze.py. Test renders on practice data
  (demo/test_figures, watermarked).
- space/app.py: Gradio side-by-side demo (greedy, 4,096 new tokens, loop badge via common.is_looping); both models
  at once on >= 10 GB GPUs, in turn on the 4050. Own venv space/.venv (gradio 6.31.0 pins pillow < 13, so it stays
  out of the main venv). UI build checked; generation untested until the edited model exists. Not deployed.
- WRITEUP.md draft: baselines and noise floor filled in, edit numbers TBD.

## Phase 4 item 4: full base sets overnight (2026-11-04/05, batch-invariant kernels, night.sh)
- 2026-11-04 18:31-18:59 UTC (28 min) base gsm8k_test400 greedy: acc 0.740, loops 0.205, cap 0.000,
  mean tokens (finished) 1,223.
- 2026-11-04 18:59-20:19 UTC (81 min) base mmlupro300 greedy: acc 0.143, loops 0.557, cap 0.010, mean tokens 1,332.
- 2026-11-04 20:19-23:13 UTC (173 min) base_full math500 greedy (all 500): acc 0.700, loops 0.244, cap 0.030,
  mean tokens 2,631.
- 2026-11-04 23:13 - 2026-11-05 02:56 UTC (224 min) base math500 sampled (200 problems x 2 samples, T 0.6,
  top-p 0.95; started before the 01:00 UTC no-new-run cutoff): acc 0.820 over 400 answers, loops 0.010,
  cap 0.065, mean tokens (finished) 3,852. Sampled answers rarely repeat verbatim but run much longer.
- Nothing deferred: all four night runs finished (NIGHT DONE 02:56 UTC).
- MMLU-Pro check (operator, base model only, no grading change): of 300 answers 167 looped; of the 133 that
  finished, 55 put the letter in \boxed{} (43 right), 72 wrote "Answer: X" without a box (18 of those right),
  3 never closed </think>, 3 other. The pre-registered grader (common.grade, letter inside \boxed{}) stays as is
  and applies to every arm the same way. A lenient reading would give 61/300 = 0.203 instead of 0.143; any
  such secondary number would be labelled post hoc and computed for every arm, only if Dev asks for it.

## Checkpoint B decisions (2026-11-05 ~08:10 IST, Dev, in chat)
- Dev: "my only directive is to modify the open weights without training to produce a significant
  improvement, all other decisions are up to you"; Dev away until 20:00 IST.
- Dev logs in to Claude Code in WSL; the operator then starts the editor session itself (instead of Dev typing
  the prompt): an interactive `claude` session in a tmux window, cwd ~/wsp-r1/editor, --permission-mode auto
  (Dev's choice), first message verbatim "Make this model better at math by editing its weights directly. You
  may not train it." The only other message the operator may send is the plan's "Continue from your
  notebook." (new session after a usage limit or a dead session). Every message sent is logged here.
- Second editing run: Dev allows at most one more fresh run (same prompt, budget reset) if run 1 does not
  beat the noise floor and the random controls; both runs are reported, neither is picked over the other.

## Pre-handoff smoke test (2026-11-05 02:57-03:00 UTC, operator; state and logs reset afterwards)
- `fp.py generate --source math_train --n 2 --seed 0 --max-tokens 1024`: 1m52s, 1/2 correct (1 finished at 725
  tokens, 1 hit the 1,024 cap). `fp.py read --layers 10,15 --module resid --positions last:8`: 36 s, (2, 8, 1536)
  per layer, no NaN. `fp.py read --layers 15 --module mlp_act --positions mean --chat`: 20 s, (2, 1, 8960). So
  `fp.py read` works as written; no fix needed.
- `wsp.load_base()`: 339 tensors, 1,777,088,000 params, 13-20 s. After loading: 7.2 GB anonymous RAM
  (RssAnon); the 10.9 GB peak RSS includes the memory-mapped safetensors pages, which the kernel can drop.
  WSL has 11 GiB visible (editor/CLAUDE.md says 16 GB RAM: the host's figure). A one-tensor-at-a-time variant of
  load_base gave the same numbers, so wsp.py was restored to its committed version (no change).

## Phase 5: seal and hand off (2026-11-05 03:05-03:40 UTC)
- Editor reset: state.json = {"budget": 50, "used": 0, "passes": []}; editor/logs and editor/edits emptied
  (.keep kept); work/fp_current removed. `mv sealed results ~/wsp_sealed/`. Commit c547592 "sealed; editor ready".
- Leak check found test/practice overlap (before the editor started): 0 MATH-500 or GSM8K questions are exact
  duplicates of practice problems, but 2 MMLU-Pro questions are word-for-word MATH-train problems (MMLU-Pro
  includes MATH-derived items) and 44 test questions have a near-twin in MATH train (same problem, one number
  changed; MATH's train/test split shares templates). Fixed rule, chosen before any edit and applied once:
  drop every practice problem that shares its first 40 normalized characters with a sealed question and either
  has difflib ratio >= 0.80 with it or shares its first 60 normalized characters. Result: math_train.jsonl
  7,498 -> 7,416 (82 removed, matching 52 test questions), gsm8k_train.jsonl 7,473 unchanged. sha256 prefixes
  after: math_train 5cd02994a542a854, gsm8k_train 28cb0e9243ac55bc. Originals in ~/wsp_sealed/practice_orig/,
  the removed list (with the test ids each matched) in ~/wsp_sealed/practice_removed.json (published after
  freeze). The app's two example problems are still in the practice set.
- Re-check over all 1,200 sealed questions (whitespace-normalized): exact 0, same first 60 characters 0,
  >= 0.80 similar 0. PLAN's grep (first 60 characters of 20 random MATH-500 questions, random.Random(0)) shows
  3 "hits", all false positives: those 60 characters contain a newline, so grep -F treats each line as its own
  pattern (e.g. "Given"); the flattened and the JSON-escaped strings have no hits.
- editor/data/practice/ holds only math_train.jsonl and gsm8k_train.jsonl.

## Editing run 1 (started 2026-11-05 03:08:13 UTC = 08:38 IST)
- Editor: Claude Code 2.1.318 in WSL, tmux session `editor`, cwd ~/wsp-r1/editor,
  `claude --model claude-opus-5-5 --permission-mode auto` (Opus 5.5, default effort "medium", Claude Pro login
  by Dev). First-run screens answered by the operator: theme (default), security note (Enter), "trust this
  folder" (Yes). Message 1 (operator, verbatim): "Make this model better at math by editing its weights
  directly. You may not train it." No other message sent. GPU left to the editor; the operator only reads the
  tmux pane, state.json and the editor's files (read-only watcher).

- 2026-11-05 10:00 IST: the Windows keep-awake ran out at 10:00:05; restarted it to hold until 2026-11-06 02:00 IST (machine-level only, nothing sent to the editor).
- 2026-11-05 10:00 IST (04:30 UTC), read-only check: editor at 10/50 passes, still running base-model baselines (base MATH seed 11: 17/32 with 10 loops and 6 caps; base GSM8K seed 11: 22/32 with 10 loops). No patch yet. The editor's input box shows the greyed Claude Code prompt suggestion "keep going until the budget is spent"; it is the app's own unsent suggestion, not a message, and the operator has not typed or pressed anything in that pane.

## Editing run 1: finished and frozen (2026-11-05)
- 13:23 IST (07:53 UTC): the editor wrote FINAL_REPORT.md and ended its turn on its own. 1 session, no usage-limit stops, no operator messages beyond the verbatim first prompt. Passes used 38 of 50 (it left 12 unused on purpose; 2 were wasted, per its report).
- The final input box showed another greyed Claude Code prompt suggestion ("commit this"); it was never sent. 13:25 IST: operator saved the pane scrollback to ~/wsp_sealed/editor_pane_run1_final.txt and closed the tmux session with `tmux kill-session` (no keystrokes sent to the model). A leftover wait-loop the editor had started (`while pgrep -f "fp.py generate"`, which matched itself) was killed.
- The edit (from FINAL_REPORT): a constant "steer away from the loop state" vector, -0.5*d with d a layer-14 residual mean difference (late failed-run positions minus late healthy positions, seed-7 practice transcripts), written exactly through `v_proj.bias` at layers 10-14 by one ridge least-squares solve. 1,278 weights in 5 tensors (0.0001% of 1,777,088,000).
- Editor's own practice numbers (greedy, 288 held-out practice problems): 207 -> 233 (71.9% -> 80.9%), 42 fixed / 16 broke; MATH 84 -> 91 of 128, GSM8K 123 -> 142 of 160; loop flags 59 -> 29, caps 20 -> 21. These are the editor's claims on practice data, not test results.
- Freeze: `sha256sum editor/edits/patch.safetensors` = 4c8a5c779ec0b5068e141b60798b74b9058e12a48378ba6085486af60664cc81 (byte-identical to editor/work/patch_H4_a05.safetensors). Copied to work/frozen_patch_run1.safetensors. apply_patch.py: changed 1,278 of 1,777,088,000 weights.
- 13:26 IST: test data moved back from ~/wsp_sealed; phase7.sh 1 started (edit, rand_w, rand_s, edit_pp with batch-invariant kernels, edit_rerun with normal kernels; MATH-500 greedy --limit 200; then the follow-ups).
- Run 1 is final. Run 2 happens only if run 1 fails the noise floor or the random controls (decided before any test results).
- Git: editor/logs/fp/ (3.2 GB of generations and activation reads) and editor/work/ data files stay out of git; the editor's scripts, notebook, FINAL_REPORT, state.json and the 3 KB frozen patch are committed.

## Phase 8 audits of editing run 1 (2026-11-05 ~13:40 IST, operator, CPU only; scripts audit.py / audit3-5 in the operator scratchpad)
- Transcript: one session, ~/.claude/projects/-home-dev-gaglani-wsp-r1-editor/81d09a96-ba48-4fad-abeb-05ce0539c42a.jsonl (2.8 MB). Tool calls: Bash 194, Monitor 14, ToolSearch 1, TaskStop 1, Write 1.
- Test leakage: the first 60 normalized characters of all 1,200 sealed questions (MATH-500, GSM8K-400, MMLU-Pro-300) were searched for in FINAL_REPORT, notebook, editor/logs/fp/*.jsonl, editor/work scripts/json/logs and the full transcript (raw and JSON-unescaped): 0 hits in each set. No mention of wsp_sealed, sealed/ or results/. The strings "math500/test400/mmlupro" occur only as operator log *filenames* in the editor's one `ls ..` of the repo root; it never opened any of them, nor RUNLOG, PLAN, eval_plan or WRITEUP.
- No gradients: 0 matches for backward/requires_grad=True/torch.optim/autograd.grad/.grad/optimizer.step/enable_grad in editor/work/*.py, summ.py, or any tool-call input.
- Model only via fp.py: no tool call loads vLLM/transformers models or calls generate outside fp.py (the one `from gen import load_llm` hit is inside a tool *result*, the editor reading fp.py's source). The editor did import operator helpers read-only (apply_patch.base_dir/load_base_tensors/patch_stats/build, common.build_prompt_ids, wsp.*) to load base tensors and tokenize practice text; its offline analysis multiplies saved activations by lm_head / layer weights (linear algebra, not forward passes). No Write/Edit on state.json, fp.py, wsp.py or CLAUDE.md.
- Verdict: run 1 passes the contamination and no-gradient audits.

## Phase 7 arm results (run 1)
- 2026-11-05 14:32 IST: edit, MATH-500 greedy --limit 200, batch-invariant: 65m23s. acc 0.720 (base 0.660), loop 0.205 (base 0.315), cap 0.020 (base 0.010), mean tokens (finished) 2,254. vs base: +6.0 pts [+0.5, +11.5], 23 fixed / 11 broken (net +12), McNemar p 0.058 (Holm, single arm). Noise-floor verdict waits for edit_rerun and the random controls.
- 2026-11-05 15:27 IST: rand_w (random change on the same 1,278 weights, same per-tensor norm, seed 0), MATH-500 greedy --limit 200: 54m29s. acc 0.690, loop 0.265, cap 0.010. vs base: +3.0 [-2.5, +8.5], 19 fixed / 13 broken (net +6). Edit vs rand_w directly: +3.0 [-2.5, +8.5], 19 / 13, McNemar p 0.38 -> the pre-registered primary comparison "edit vs random (same weights)" is NOT passed.
- Decision (by the rule fixed this morning, before any test result: "one more fresh run if run 1 does not beat the noise floor and the random controls"): run 1 does not beat the same-weights random control, so editing run 2 will happen. Run 1 stays final and is reported in full; neither run is picked over the other. Order: let phase7.sh 1 finish (fast-track + follow-ups; the GPU can only hold one job), then re-seal the test data, archive run 1s
 [line above cut off by a shell-quoting slip; continued:] ... archive run 1's editor/ state outside editor/, reset editor/ and start run 2 with the same verbatim prompt and a fresh 50-pass budget.
- 2026-11-05 16:40 IST: rand_s (random change on 1,276 random positions in the same 5 tensors, same per-tensor norm, seed 0), MATH-500 greedy --limit 200: 74m04s. acc 0.635, loop 0.325, cap 0.025. vs base: -2.5 [-8.5, +3.5], 17 fixed / 22 broken (net -5). Edit vs rand_s (secondary, not a pre-registered primary comparison): +8.5 [+2.5, +14.5], 27 / 10, McNemar p 0.0076. Note: each tensor has only 256 entries and the patch changes 255-256 of them, so rand_s is essentially a second random draw of rand_w; the two random edits (+3.0 and -2.5) show the spread of random perturbations of this size.
- 2026-11-05 18:04 IST: edit_pp (edit + presence penalty 1.5), MATH-500 greedy --limit 200: 84m03s. acc 0.830, loop 0.020, cap 0.050, mean tokens (finished) 3,057. vs base_pp (0.855, loop 0.035, cap 0.060; Phase 4): -2.5 [-7.0, +2.0], 8 fixed / 13 broken, p 0.38. vs base: +17.0 [+11.0, +23.0], 38 / 4. For reference base_pp vs base: +19.5 [+13.5, +25.5], 41 / 2. Reading: a decode-time presence penalty alone removes almost all loops and gives about 3x the edit's gain; on top of the penalty the edit adds nothing (slightly negative, within noise). The edit's MATH gain is a loop fix that the penalty already provides.

## Amendment 2: fresh untouched test sets (2026-11-05 ~18:40 IST)
- Dev asked for "a measurable gain on an untouched application" and approved adding this track ("Yes, add it").
- Built gsm8k_fresh400 (0 skipped as practice-contaminated) and mmlupro_fresh300 (4 skipped) with
  make_fresh.py. They are written to sealed/ and are gitignored. sha256 values are in eval_plan_amendment2.md.
- No model had been run on them when eval_plan_amendment2.md was committed (next line has the commit).
- Amendment 2 commit: d1290be (2026-11-05T13:02:47+00:00).
- 2026-11-05 18:51 IST: edit_rerun (edit, normal kernels), MATH-500 greedy --limit 200: 46m56s. acc 0.715, loop 0.240, cap 0.025. vs edit: -0.5 [-5.5, +4.5], 12 wrong->right / 13 right->wrong (net -1, 25 flips). N = max(9, 1) = 9 (unchanged). Original rule (b) threshold = 2 x max(31, 13) = 62.
- Run 1 MATH-500 verdict (pre-registered rules): (a) primary test NOT passed (edit vs base raw McNemar p 0.058 > 0.05 before Holm; edit vs rand_w p 0.38). (b') net +12 > N = 9: passed. (b) original rule net +12 < 62: not passed. So the MATH effect is NOT "clearly above the noise floor" under either rule, because (a) fails. Loop-rate and GSM8K primary comparisons are evaluated when the GSM8K follow-up lands.
- 18:51 IST: FASTTRACK DONE; follow-ups started (edit on gsm8k_test400 first).
- 2026-11-05 19:17 IST: edit on gsm8k_test400, greedy: acc 0.7975 (base 0.740), loop 0.1275 (base 0.205), cap 0. vs base: +5.8 [+1.8, +9.8], 45 fixed / 22 broken, McNemar p 0.0067.
- Pre-registered primary family (eval_plan.md, Holm over 4; primary_run1.py):
  | Test | n | fixed/broken (or loops added/removed) | p raw | p Holm | |
  |---|---|---|---|---|---|
  | MATH-500 acc, edit vs base | 200 | 23 / 11 | 0.058 | 0.12 | fail |
  | MATH-500 loop rate, edit vs base | 200 | 11 added / 33 removed | 0.0013 | 0.005 | PASS |
  | GSM8K acc, edit vs base | 400 | 45 / 22 | 0.0067 | 0.020 | PASS |
  | MATH-500 acc, edit vs rand_w | 200 | 19 / 13 | 0.38 | 0.38 | fail |
- Run 1 summary: the edit significantly improves GSM8K accuracy (+5.8 pts) and cuts MATH-500 loops (31.5% -> 20.5%); its MATH-500 accuracy gain (+6.0) is not significant and not distinguishable from the same-weights random control. Run 2 stays triggered (rule: run 1 must beat the noise floor AND the random controls).
- 2026-11-05 21:08 IST: edit on mmlupro300, greedy: 111m10s. acc 0.127 (base 0.143), loop 0.530 (base 0.557), cap 0.027. vs base: -1.7 [-5.3, +1.7], 12 fixed / 17 broken, p 0.46. No transfer to MMLU-Pro.
- Note: phase7.sh writes logs_<name>.txt, so logs_edit.txt (the MATH-500 fast-track edit log) was overwritten by the GSM8K and then the MMLU-Pro edit runs. The results files are separate and intact; the run times survive in the phase7 output (65m23s, 25m29s, 111m10s). Fresh-set logs use per-set names.

## Phase 8, run 1 (2026-11-05 ~23:40 IST, CPU only)
- Loop flips on MATH-500 (phase8_loopflips.py): edit removed 33 / added 11 (p 0.0013); rand_w removed 25 / added 15 (p 0.15); rand_s removed 23 / added 25 (p 0.89).
- Hand-check of 30 graded answers (phase8_handcheck.py: 15 base, 15 edit, random.Random(0), MATH-500 fast track): 30/30 graded correctly. 22 boxed answers equal the gold answer up to formatting (\dfrac vs \frac, 76 vs 76^\circ, Navin vs \text{Navin}); 7 loops have no answer; 1 (edit m500-66) is a genuinely wrong polynomial. No grader errors.
- Example rule (phase8_examples.py; lowest-index fixed problem per MATH subject, fast track): m500-75 (Counting & Probability), m500-76 (Intermediate Algebra), m500-123 (Geometry), m500-128 (Algebra), m500-197 (Number Theory), m500-200 (Prealgebra), m500-248 (Precalculus); 23 fixed in total. First pick m500-75: base loops (onset at token 2,418) and is stopped at 4,096; edit answers 11/36 in 1,825 tokens. make_demo_video.py reports no weak flags.
- Rendered demo/m500-75.mp4 (30 s), demo/m500-75_explainer.mp4 (60 s), demo/m500-75_strip.gif. Two fixes to make_demo_video.py first: the percentage is printed in plain decimals (it showed "7.2E-05%"), and the caption groups tensors by type ("self_attn.v_proj.bias (1,278 weights)") so it no longer runs under the layer label.
- figures/fig1_accuracy_loops.png and fig3_fixed_broken.png drawn for the fast track (the net labels in fig3 moved right so they no longer touch the bars). fig2 (projection) needs a GPU read pass; it is pending.
- WRITEUP.md: run 1 numbers filled in (TL;DR, what Claude found, results, methodology, caveats).

## Amendment 3: run-1 technique on MMLU-Pro loops (2026-11-06 ~00:40 IST)
- Dev saw an MIT Weight-Space Programming post (Qwen3.5-0.8B, unfinished/looping reasoning, 7.9% -> 48.1%) and asked for an equally strong, honest result ("proceed with both tracks").
- Built mmlupro_fresh2_300 (sealed test) and a3_practice_mmlupro128 (practice) with make_a3_data.py (13 near-duplicates skipped). sha256 values are in eval_plan_amendment3.md. Both are in sealed/ (gitignored).
- a3_build.py: run 1's steer.py + make_h4.py with nothing changed except the transcripts. a3_selftest.py: fed run 1's direction, it reproduces run 1's frozen patch exactly (0 of 1,280 values differ).
- No model has been run on either new set at commit time. Queue: fresh.sh (Amendment 2) -> a3.sh -> editing run 2.
- Amendment 3 commit: 396f2db (2026-11-05T19:00:21+00:00).

## Phase 7 follow-ups (run 1), finished 2026-11-06 02:22 UTC
- Full MATH-500 (500, greedy, batch-invariant): base_full 70.0% -> edit_full 74.0%, +4.0 [+0.4, +7.6],
  54 fixed / 34 broken, exact McNemar p = 0.042 (single test, not part of the Holm family). Loops 24.4% ->
  18.4% (63 removed / 33 added, p = 0.0029). `logs_analyze_full.txt`.
- Split (`phase7_full_split.py`, `logs_analyze_full_split.txt`): the 200 fast-track problems in the full runs
  are byte-identical to the fast-track runs (200/200 base, 200/200 edit). On the other 300 problems alone:
  72.7% -> 75.3%, 31 / 23, p = 0.34; loops 19.7% -> 17.0%, 30 removed / 22 added, p = 0.33. Not significant:
  the full-set p < 0.05 leans on the fast-track 200. Reported as such.
- Sampled MATH-500 (200 x 2, T 0.6, top-p 0.95): base 82.0% -> edit 79.2%, -2.8 [-6.2, +0.8], 19 / 29 by
  problem, sign-flip p = 0.17. Loops are already ~1% under sampling (1.0% -> 0.8%), so the loop fix has
  nothing to act on; consistent with the editor's own prediction that the benefit would shrink under
  sampling. `logs_analyze_sampled.txt`.
- Next on the GPU: fresh.sh (Amendment 2), then a3.sh (Amendment 3), then editing run 2.

## Paused by Dev, 2026-11-06 (IST morning)
- Dev asked to stop all processes and resume on their say-so. Stopped: fresh.sh (Amendment 2) during its
  first arm (base on gsm8k_fresh400, about 1 minute in, nothing written to results/), and the queued a3.sh
  (Amendment 3, had not started). Keep-awake timers stopped. GPU idle, no eval processes left.
- The killed run's partial log is kept as `logs_fresh_base_gsm8k_fresh400_killed_20261106.txt`. No results
  file existed, so the resumed run uses the same name and command (nothing has been seen from it).
- Queue scripts copied into `ops/` so they survive outside the session scratchpad: fresh.sh, phase7.sh,
  run2_prep.sh, run2_launch.sh, send_prompt.sh, watch_editor2.sh.
- To resume, in this order: `bash ops/fresh.sh` (skips any arm whose results file exists), then
  `bash a3.sh`, then editing run 2 (`ops/run2_prep.sh`, `ops/run2_launch.sh`, `ops/send_prompt.sh`,
  `E=~/wsp-r2/editor bash ops/watch_editor2.sh`, with its deadline moved to match the new start).

## Resumed by Dev, 2026-11-06 21:29 IST
- Dev: "start". Launched `ops/resume.sh` (fresh.sh, then a3.sh) and a keep-awake timer until 2026-11-09 20:00 IST.

## Paused again by Dev, 2026-11-06 23:43 IST
- Kept: `results/base_gsm8k_fresh400_greedy.jsonl` (complete, 400 problems, finished 23:05 IST).
- Stopped: arm e1 on gsm8k_fresh400, under a minute in, nothing written; partial log kept as
  `logs_fresh_e1_gsm8k_fresh400_killed_20261106.txt`. Keep-awake stopped.
- Resume with `bash ops/resume.sh`: fresh.sh skips the finished base arm and starts at e1.

## Resumed by Dev, 2026-11-08 11:04 IST
- Dev: "Continue". Launched `ops/resume.sh` (fresh.sh skips the finished base arm on gsm8k_fresh400 and starts at e1,
  then a3.sh). Keep-awake until 2026-11-11 20:00 IST.

## Offline run, 2026-11-08 (Dev going offline)
- The GPU queue (ops/resume.sh: fresh.sh, then a3.sh) needs no internet. Checked under a dead proxy: the
  only Hub call, `apply_patch.base_dir()` (used by a3_build.py and apply_patch.py), falls back to the cached
  snapshot ad9f0ae0 after about 50 s of retries; with `HF_HUB_OFFLINE=1` it resolves locally at once.
- a3.sh now exports HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, VLLM_NO_USAGE_STATS=1 before any step. No
  change to what it computes. fresh.sh was already running and is left untouched (its eval.py runs load
  only local model folders).
- Editing run 2 needs the internet (the editor is a Claude Code session), so it waits until Dev is back
  online. Logging, analysis write-up and pushes also wait.

## Amendment 3: crash and fix, 2026-11-08
- a3prac, the e3 build and the controls finished (details logged with the A3 results). The first arm on
  mmlupro_fresh2_300 (base) crashed after 84 min: question mmlup-11603 has a 1,555-token prompt, above the
  1,024-token allowance in `max_len = max_tokens + 1024`, so its answer hit the model's context limit
  (17,408) before the 16,384-token cap, and the next chunk request raised a vLLM ValueError. Nothing was
  written; the log is kept as `logs_a3_base_mmlupro_fresh2_300_crashed_20261108.txt`.
- Fix in `gen.py`: an answer whose prompt + output reaches the context limit is stopped and counted as a cap.
  Every earlier run is unaffected (only answers that would have crashed take the new branch; no earlier set
  has a prompt over 1,024 tokens whose answer reached the limit). Checked with a throwaway run, base model,
  600-token cap (`fixtest_ctxlimit`, accuracy meaningless): mmlup-11603 ends as cap at 69 tokens, no crash.
  The same code runs for every Amendment 3 arm. No edited-model output on this set existed at the time.

## Amendment 2 results (fresh sets), finished 2026-11-08 19:36 IST
Run 1's frozen edit on sets built after the edit was frozen (greedy, 16,384 cap, batch-invariant).
`logs_analyze_fresh_*.txt`, `amendment2_primary.py` -> `logs_amendment2_primary.txt`.

| Set | base | e1 | e1x1.5 | rand_w |
|---|---|---|---|---|
| gsm8k_fresh400 accuracy | 72.2% | 78.2% (+6.0 [+2.2, +9.8]) | 78.8% (+6.5 [+2.5, +10.5]) | 74.8% |
| gsm8k_fresh400 loop rate | 23.5% | 13.8% | 11.0% | 20.2% |
| mmlupro_fresh300 accuracy | 9.7% | 10.7% (+1.0 [-2.3, +4.3]) | 12.3% (+2.7 [-0.7, +6.0]) | 11.7% |
| mmlupro_fresh300 loop rate | 51.3% | 50.0% | 47.0% | 55.7% |

- Primary family (pre-registered: e1, e1x1.5, e2 vs base on each fresh set, Holm over up to 6). e2 does not
  exist yet. Raw exact McNemar p: e1 GSM8K 45/21 p = 0.0043; e1x1.5 GSM8K 49/23 p = 0.0029; e1 MMLU-Pro
  15/12 p = 0.70; e1x1.5 MMLU-Pro 18/10 p = 0.18.
  - Holm over the 4 existing tests: GSM8K e1 0.013 PASS, e1x1.5 0.012 PASS; MMLU-Pro both fail.
  - Whatever e2 gives, with m = 6 the largest multiplier either GSM8K test can get is 6, so both stay below
    0.05 (0.026 and 0.018). The GSM8K passes are final; the full Holm table is written once e2 exists.
- Result: run 1's GSM8K gain replicates on 400 problems nobody had looked at (+6.0 vs +5.8 on the original
  set). MMLU-Pro: no effect, as before.
- Secondary: e1 vs rand_w on fresh GSM8K 41/27, p = 0.11 (not significant). Loops on fresh GSM8K: e1 removed 59
  and added 20 (p = 1.3e-5); rand_w removed 48 and added 35 (p = 0.19).

## Amendment 3 build (2026-11-08 20:06 IST)
- a3prac (base, 128 MMLU-Pro practice questions, 8,192 cap): 10.9% accuracy, 48.4% loops, 114 failed, 63 finished.
- Healthy class A had 1,971 positions (< 2,000), so class B was used, as the rule fixes (B: 4,397 positions;
  loop: 135,827). |d| = 18.31, cosine with run 1's direction 0.51, realized fraction 0.84.
- patch_e3 sha256 80dbfa8aa11772d00b4033c5e27505a902104316635244809c230705755617b5 (1,279 weights in
  v_proj.bias of layers 10-14); rand_e3 same weights, same per-tensor norm, seed 0 (1,277 changed).

## While online, 2026-11-08 ~22:30 IST
- Installed lm-eval 0.4.13 into .venv for the pre-registered GSM8K cross-check, with every existing package
  pinned (pip constraints from `pip freeze`): only new packages were added, none changed (torch 2.10.0+cu128,
  vllm 0.19.1, transformers 5.22.1 unchanged). Added packages: `requirements.lmeval-added.txt`. 0.4.9 failed
  to import with transformers 5.x, so 0.4.13 is used. GSM8K test data cached; task load checked offline.
- `ops/lmeval_gsm8k.sh` queued: waits for a3.sh, then base and edit with gsm8k_cot_zeroshot, chat template,
  </think> stripping, 16,384 max tokens, greedy. Output in results/lmeval/.

## Amendment 3 results, finished 2026-11-09 06:52 IST
`logs_analyze_a3*.txt`, `amendment3_primary.py` -> `logs_amendment3_primary.txt`. mmlupro_fresh2_300, greedy,
16,384 cap, batch-invariant.

| Arm | Accuracy | Loop rate | Cap rate |
|---|---|---|---|
| base | 12.7% | 51.3% | 1.0% |
| e3 | 11.0% | 49.3% | 1.3% |
| rand_e3 | 12.7% | 50.0% | 0.7% |
| base_pp (presence penalty 1.5) | 20.3% | 10.3% | 5.0% |

- Primary (Holm over 2): e3 vs base 14 better / 19 worse, p raw 0.49, Holm 0.97, FAIL. e3 vs rand_e3 14 / 19,
  p raw 0.49, Holm 0.97, FAIL. No claim of a gain and no claim of beating a random edit.
- Secondary: e3 loops removed/added vs base 51/45 (p = 0.61). Presence penalty vs base 31/8 (p = 0.0009),
  e3 vs base_pp 6/34. By category nothing moves (largest: chemistry 7 -> 5). e3 on MATH-500 200: 66.0% -> 68.0%,
  21/17, p = 0.63, loops 31.5% -> 25.0% (does not hurt maths).
- Reading: run 1's recipe, re-estimated on MMLU-Pro transcripts, does not fix MMLU-Pro loops. The healthy class
  had to fall back to B (finished runs) because only 16 of 128 practice answers were correct; the loop/healthy
  contrast on MMLU-Pro is weaker and only half-aligned with run 1's (cosine 0.51).
- Note for audit: base scores 0 in law (0/35), health (0/18), biology (0/15), history and other, below the
  10-option chance rate. Grader extraction on MMLU-Pro to be hand-checked (CPU, during run 2). The comparisons
  above are paired and use the same grader, so they stand either way.

## lm-eval cross-check stopped, 2026-11-09 09:30 IST
- At 88 output tok/s it had done 190 of 1,319 problems in 2.5 h (about 30 h for both arms; no loop early stop).
  Stopped so editing run 2 can use the GPU while Wi-Fi is on. Partial log kept as
  `logs_lmeval_base_stopped_20261109.txt`; no results were written and no accuracy was seen.
- Rerun after run 2 on a fixed subset, chosen now: `--limit 400` (lm-eval's first 400 GSM8K test problems).

## Editing run 2 started, 2026-11-09 ~09:45 IST
- `ops/run2_prep.sh`: sealed/ and results/ moved to ~/wsp_sealed; clean copy ~/wsp-r2 built from handoff commit
  3d530f4 (common.py, gen.py, apply_patch.py, editor/CLAUDE.md, fp.py, wsp.py) plus the decontaminated practice
  files (sha256 prefixes match Phase 5); state.json reset to 0/50; .venv symlinked. No run-1 notes, logs, edits or
  results in the copy.
- `ops/run2_launch.sh`: tmux `editor`, cwd ~/wsp-r2/editor, Claude Code 2.1.318 (same as run 1), Opus 5.5, auto mode.
  The folder-trust dialog was answered "Yes, I trust this folder" (the operator built the folder).
- Prompt sent verbatim with `ops/send_prompt.sh`: "Make this model better at math by editing its weights directly.
  You may not train it." No other message has been sent.
- Watcher: `ops/watch_editor2.sh` with E=~/wsp-r2/editor, deadline 2026-11-11 17:30 IST.

## MMLU-Pro grader check (POST HOC, exploratory), 2026-11-09
- Hand look at wrong base answers in law found a false negative: mmlup-1167 ends "**Answer: G**", gold G, graded
  wrong, because the pre-registered MCQ grader accepts only a \boxed{} letter. So absolute MMLU-Pro accuracies
  are understated for every arm.
- `posthoc_mmlupro_lenient.py` (`logs_posthoc_mmlupro_lenient.txt`) rescored with a lenient extractor (boxed, else
  explicit "Answer: X" / "the correct answer is X" / "option X" after </think>). Not pre-registered; the
  pre-registered grader stays primary. (Its "boxed" count column is a broken diagnostic; ignore it.)
  - mmlupro_fresh2_300: base 12.7% -> 21.0% lenient; e3 21.0% vs 20.7% (31/32), rand_e3 21.3%, base_pp 34.0% (58/19).
  - mmlupro_fresh300: base 19.7%, e1 21.3% (25/20, p 0.55), e1x1.5 20.3%, rand_w 20.7%.
  - mmlupro300: base 20.3%, edit 20.3% (23/23).
- Conclusions unchanged: no edit moves MMLU-Pro under either grader; the presence penalty does (+13 points lenient).
  Report MMLU-Pro absolute numbers with this caveat.

## Token cost on gsm8k_fresh400 + race-demo plan (POST HOC, descriptive), 2026-11-09
- `posthoc_tokens_gsm8k_fresh.py` (`logs_posthoc_tokens_gsm8k_fresh.txt`), from the existing Amendment 2 results.
  Not pre-registered; report as descriptive, never as a tested claim.
  - Tokens actually generated (loop detector on): base 816k, e1 644k (-21%), e1x1.5 629k, rand_w 715k (-12%).
  - Counting each loop at the 16,384 cap (what a user without our loop killer pays): base 1.91M, e1 1.28M (-33%),
    rand_w 1.70M (-11%). Speed per token is unchanged (same size model); the saving is answers that stop looping.
  - 43 problems: base loops, e1 finishes correct. 14 the other way.
- Race demo, rule fixed now before any GPU run: example = the base-loops / e1-correct problem with the shortest
  e1 answer, ties by id -> gsm-794 (e1 608 tokens, answer 32; base reaches "32" ~6% into its answer and repeats it
  42 times without committing). Captions must say it is 1 of 43 such problems and that 14 go the other way.
- After run 2 (GPU busy now): (1) base and e1 on gsm8k_fresh400 with --no_early_stop, new names `base_nokill`,
  `e1_nokill`, for measured (not modelled) token totals; (2) single-stream tok/s benchmark on this laptop;
  (3) record each model streaming gsm-794 alone with per-token timestamps, replay side by side on a shared clock
  (both cannot fit on the 6 GB GPU at once; say so in the clip).

## Editing run 2: usage limit, 2026-11-09 13:07 IST
- Watcher event usage_limit at 2026-11-09T07:37Z (13:07 IST), 15/50 passes used, no edit files yet. Pane: "Usage
  limit reached · wrapping up · limit resets 8:50am". Watcher exited. No message sent to the editor. After the
  reset: if the session is dead, start a new one and send only "Continue from your notebook." (log it).
- Usage limit cleared on its own; the session continued to 30/50 passes. Its turn then ended on "API Error: Can't
  reach the API server (EAI_AGAIN)" (Wi-Fi drop, ~16:06 IST) and it sat idle at the prompt. Internet back, so at
  2026-11-09T10:54:31+00:00 the operator sent "Continue from your notebook." into the same session (no other text). Watcher restarted.
- 16:25 IST (10:55Z): FINAL_REPORT.md written, 30/50 passes used. Watcher event final_report.

## Editing run 2 closed, audited, frozen (2026-11-09 ~16:30 IST)
- 16:26 IST: pane scrollback saved to ~/wsp_sealed/editor_pane_run2_final.txt (only 50 lines survive: tmux reported
  a bogus 131072x1 screen size), then `tmux kill-session -t editor`. The input box showed a greyed Claude Code
  prompt suggestion ("use the remaining 20 passes to fix the long caps"); it was never sent (transcript confirms).
- The edit (from FINAL_REPORT): two low-importance layer-19 MLP neurons (3918, 4028) rewritten as AND-gated
  detectors (gate row = shrinkage-LDA detector with a threshold; up row = paragraph-start detector). 3918 "loop
  exit": at a loop restart, writes the shift toward the pre-"**Final Answer**" state. 4028 "close thinking": after
  a boxed Final Answer, writes the shift toward the pre-</think> state. 9,215 weights in 3 tensors (0.0005%).
  Closed-form (means, LDA, quantile thresholds); no gradients, no perturb-and-evaluate search.
- Editor's own practice numbers (greedy): on 96 fresh practice problems 63 -> 83 (v2), loops/caps 26/9 -> 1/3,
  tokens 314k -> 255k; over 160 problems with base and v2 both measured, 112 -> 132. Editor's claims on practice
  data, not test results.
- Freeze: sha256 be59f234f2f108a3877e36b3985a17641142a37f53aa777cb7c5d388c95ee984 (= editor work/patch_v2), copied to
  work/frozen_patch_run2.safetensors (82 MB: whole layer-19 MLP tensors, so it stays out of git with work/).
  apply_patch: changed 9,215 of 1,777,088,000 weights. rand_w_r2 changes 9,211 (4 random values round back to the
  base bf16 value); rand_s_r2 same norm at random positions.
- Audit (operator, CPU; scratchpad audit_r2*.py). Transcript ~/.claude/projects/-home-dev-gaglani-wsp-r2-editor/
  83021052-b1bb-4d07-a5d1-21459b627ff1.jsonl (2.0 MB). Tool calls: Bash 89, Write 1.
  - User-role messages: the prompt; Claude Code's own automatic usage-limit notices (07:38Z wrap-up note, 08:51Z
    "limit has reset, continue" + "earlier usage-limit notes no longer apply"); one task-notification; and the
    operator's "Continue from your notebook." (10:54Z). Nothing else.
  - Test leakage: first 60 normalized characters of every sealed question searched in the transcript, report,
    work/, logs/ (incl. fp/*.jsonl) and data/. 0 hits in gsm8k_test400, gsm8k_fresh400, mmlupro300,
    mmlupro_fresh300, mmlupro_fresh2_300, a3_practice. math500: 4 prefix hits (m500-118, -152, -294, -492), all
    stock openings that also start *different* practice problems (e.g. mtrain-geometry-739/834, prealgebra-1002);
    no test problem text appears. No mention of wsp_sealed, wsp-r1, sealed/, results/, eval_plan, RUNLOG, WRITEUP, PLAN.
  - No gradients: 0 matches (backward/requires_grad=True/torch.optim/autograd.grad/optimizer.step/enable_grad).
  - Model only via fp.py: AutoModelForCausalLM appears only in a tool *result* (the editor reading fp.py's
    source); other from_pretrained calls are AutoTokenizer. No writes to state.json, fp.py, wsp.py or CLAUDE.md.
  - Verdict: run 2 passes the contamination and no-gradient audits.
- Archived to editor_run2/: FINAL_REPORT, state.json, notebook, baseline logs, work scripts/configs (no data).
- 16:30 IST: `ops/run2_eval.sh` started (offline-safe): phase7.sh 2 (P = ~/wsp-r2 patch; test data restored),
  then e2 on both fresh sets (Amendment 2), then the post hoc no-loop-kill runs base/e1/e2_nokill on
  gsm8k_fresh400 (e2_nokill added now, descriptive, before any run-2 result), then lm-eval --limit 400.
