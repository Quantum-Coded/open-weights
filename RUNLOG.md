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
