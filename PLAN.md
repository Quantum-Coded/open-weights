# PLAN.md: Weight-space programming on DeepSeek-R1-Distill-Qwen-1.5B

Hand this file to Claude Code at the root of the `wsp-r1/` folder. Read it fully before doing anything.

---

## 1. Your role

You are the **operator** (research engineer) for this project. You set up the environment, run the
pilot and all evaluations, keep the books, and build the demo assets.

**You are NOT the editor.** A separate, fresh Claude Code session started inside `editor/` does the
weight editing, following `editor/CLAUDE.md`. You never edit model weights, never write to
`editor/edits/`, and never give the editor hints. If the editor is steered, the result is worthless.

Dev (the user) owns every decision marked **CHECKPOINT**. Stop there, report, and wait.

## 2. The project in five lines

- Recent result: MIT CSAIL ("Sparks of Weight-Space Programming", 2 Nov 2026) had GPT-6 Astra edit
  0.016% of Qwen3.5-0.8B's weights, with no training, to remove reasoning loops: MATH-500 7.9% to 48.1%.
- A follow-up had Claude Opus 5.5 fix negation in Stable Diffusion 3.5 the same way.
- Here, Claude edits **deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B**, which loops a lot under greedy
  decoding, on a laptop RTX 4050.
- The claim is tested with a pre-registered, paired before/after evaluation against random-edit controls.
- Goal: a credible result posted on LinkedIn by about **11 Nov 2026** (fast track), full eval a week later.

References: https://rikiyat.github.io/blog/weight-space-programming/ ,
https://github.com/RikiyaT/lm_understand-logs ,
https://sabhyachhabria.com/writing/weight-space-negation/

## 3. Hardware and environment

- Windows laptop, NVIDIA RTX 4050 Laptop GPU (6 GB VRAM), 16 GB RAM.
- Everything runs in **WSL2 Ubuntu** (vLLM is Linux-only). Both Claude Code sessions (yours and the
  editor's) must run inside WSL2, so `fp.py` can reach vLLM.
- Model: ~3.5 GB in bf16. That leaves ~1.5 GB of KV cache, about 50k tokens: roughly three 16k-token answers at a time.

## 4. Hard rules for you (the operator)

1. Never modify model weights, `editor/edits/`, `editor/state.json`, or `editor/CLAUDE.md`'s rules.
2. Never put test data (anything in `sealed/` or `results/`) where the editor can read it. Test data
   lives outside the project while the editor works.
3. After `eval_plan.md` is committed, never change it. New ideas go in `RUNLOG.md` as "exploratory".
4. Never pick examples, subsets or thresholds after seeing edited-model results. Use the rules in
   `eval_plan.md`.
5. Never delete or overwrite a results file. Re-runs get a new `--name`.
6. Ask Dev before anything outward-facing or hard to undo: `git push` (except the standing
   phase-by-phase pushes to the project repo, §7), Hugging Face uploads, deleting files, spending
   money, installing system packages with sudo.
7. Keep `RUNLOG.md` at the repo root: a dated line for every command that matters, its runtime, and
   the headline output. This becomes the methods section.
8. Commit to git after each phase, with a message saying what changed.

## 5. What's in the kit (read these files first)

| File | Purpose |
| --- | --- |
| `README.md` | Command-level walkthrough (this plan wraps it) |
| `common.py` | Prompts, answer extraction, grading (`math-verify`), loop detector |
| `gen.py` | Chunked vLLM generation (2,048-token chunks) with early loop kill |
| `prepare_data.py` | Downloads MATH-500, GSM8K, MMLU-Pro, MATH train; writes `sealed/` and `editor/data/practice/` |
| `apply_patch.py` | Base + patch (× scale) into a runnable folder under `work/` |
| `make_random_control.py` | Random edits matched to the editor's patch |
| `eval.py` | One arm on one test set into `results/<name>_<set>_<regime>.jsonl` |
| `analyze.py` | Paired table: accuracy, loop rate, 95% bootstrap CIs, fixed/broken, McNemar, Holm |
| `eval_plan.md` | Pre-registration template |
| `editor/CLAUDE.md`, `editor/fp.py`, `editor/wsp.py` | The editor's brief, budget gate (50 runs) and weight helpers |

The kit's offline logic was unit-tested on CPU, but the GPU paths (vLLM, `fp.py read`) have **not**
been run yet. Expect to fix small issues in Phase 1; record each fix in `RUNLOG.md`.

## 6. Reproducibility and stability (added 4 Nov, Dev)

The question a reader will ask is "is this effect bigger than the noise?". These steps answer it.
Each is wired into the phase named below.

| # | Step | Phase | Output |
| --- | --- | --- | --- |
| 1 | Version lock: `pip freeze > requirements.lock.txt`; GPU, driver and CUDA in `RUNLOG.md`. Publish the lock file with the code. | 0 | `requirements.lock.txt`, RUNLOG env block |
| 2 | Run-to-run noise: rerun the base model on the same 200 MATH-500 problems, greedy, `--name base_rerun`. Pre-registered as the **noise floor**. | 4 | flips (correct↔incorrect) and accuracy difference vs `base` |
| 3 | Rerun the headline arm: `edit_rerun` on the same 200 problems. | 7 | same flip rate for the edited model |
| 4 | Cross-hardware check (optional, **ask Dev first**): base and edit on 100 of the 200 problems on a free Kaggle GPU, same commands. | 8 | difference from the laptop numbers |
| 5 | Independent harness cross-check: GSM8K for base and edit through EleutherAI's lm-evaluation-harness. | 8 | agreement in direction and rough size with `eval.py` |
| 6 | Reproduction kit: `REPRODUCE.md`, fresh machine → headline table, with expected numbers and the noise floor. | 9 | `REPRODUCE.md` |
| 7 | Editor stability (follow-up post only, **ask Dev first**): a second independent editing run in a fresh `editor/` copy, same one-line prompt. | follow-up | same layers / direction? similar effect size? |

**Reporting rule:** in the write-up, every headline number is reported with its noise floor next
to it (e.g. "accuracy 41.0% → 52.5% (+11.5 pts; run-to-run noise ±1.0 pt, 4 flips/200)").

**Why reruns can differ at all:** greedy decoding is deterministic in exact arithmetic, but vLLM's
batching, prefix caching and bf16 kernels can change the order of floating-point operations, so a
near-tie between two tokens can flip and the rest of the answer diverges. Steps 2–3 measure how
often that changes the grade.

## 7. GitHub

Repo: https://github.com/Quantum-Coded/open-weights . Dev's standing instruction (4 Nov): push after
every phase, so the repo tracks the project phase by phase. Commits and pushes show **only Dev
(Quantum-Coded)** as author: no co-author trailers and no AI attribution lines. Everything in
`.gitignore` (`sealed/`, `results/`, `work/`, `editor/data/`, weights) stays out until Phase 9 decides
what to publish.

---

## Phase 0: Environment (target: 4 Nov)

- [ ] Confirm WSL2 sees the GPU: `nvidia-smi` inside WSL shows the RTX 4050.
- [ ] Python 3.10–3.12 venv: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
      and `python -c "import vllm; print(vllm.__version__)"`
- [ ] `git init`, add a `.gitignore` for `.venv/`, `work/`, `*.safetensors`, `sealed/`, `results/`,
      `editor/data/`, then commit.
- [ ] Create `RUNLOG.md` with the environment versions (torch, vllm, transformers, CUDA, driver).
- [ ] Version lock (§6 step 1): `pip freeze > requirements.lock.txt`, committed and published with the
      code; GPU model, driver version, CUDA driver version and torch CUDA build recorded in `RUNLOG.md`.

**Done when:** torch sees CUDA and vLLM imports.

## Phase 1: Smoke test (target: 4–5 Nov)

- [ ] `python prepare_data.py`. Record the counts it prints.
- [ ] `python apply_patch.py --out work/base` (downloads the model). Record the model's HF revision hash.
- [ ] `python eval.py --model work/base --name smoke --set practice:math_train --limit 5 --max-tokens 4096`
- [ ] Inspect the 5 outputs by eye. Check that:
  - the prompt opens with `<think>`;
  - finished answers contain `</think>` and a `\boxed{}`;
  - `correct` matches your own reading of each answer;
  - `n_tokens` and `status` look sane.
- [ ] If vLLM runs out of memory: lower `gpu_util` in `gen.load_llm` to 0.85, add `enforce_eager=True`
      (CUDA graphs cost VRAM), or lower `--max-tokens`. Note any change in `RUNLOG.md`.
- [ ] Commit.

**Done when:** 5 problems generate, grade correctly, and results files appear.

## Phase 2: Pilot (target: 5 Nov)

- [ ] `python eval.py --model work/base --name pilot --set practice:math_train --limit 20 --no-early-stop --max-tokens 16384`
- [ ] Record:
  - greedy loop rate and cap-hit rate;
  - accuracy;
  - mean tokens of finished answers;
  - wall-clock minutes;
  - the detector's false-positive rate (printed: flagged answers that still finished correctly).
- [ ] Read 3 looping transcripts. Confirm they really are loops and note what they look like.
- [ ] Estimate the throughput (problems per hour, with early stop) and project the runtime of a
      200-problem MATH-500 greedy run.
- [ ] If the false-positive rate is above ~5%, propose new detector thresholds (window, n-gram, fraction).
      Don't apply them yet.

**CHECKPOINT A (report to Dev):** a pilot table, 2 short loop excerpts, projected runtimes, and a
go/no-go recommendation.
- Go: greedy loop rate ≥ ~15%.
- Below that: recommend switching to Qwen3.5-2B or OpenThinker3-1.5B; the kit works for any HF causal LM if `BASE_MODEL` changes.

Wait for Dev's decision.

## Phase 3: Pre-register (target: 5 Nov)

- [ ] Fill in `eval_plan.md` with Dev's approval:
  - token cap and detector thresholds, plus the pilot false-positive rate;
  - **fast-track set: MATH-500, fixed 200-problem subset = `eval.py --limit 200`** (seed 0, so it is
    the same 200 for every arm); full sets for the follow-up;
  - arms, statistics and the example-selection rule as written in the template;
  - the **noise-floor** procedure and decision rule (§6 step 2; the template's "Noise floor"
    section). The value itself is measured in Phase 4 and recorded in `RUNLOG.md`; the plan is
    not edited after commit.
- [ ] `git commit -m "pre-registration"` and record the commit hash in `RUNLOG.md`.

**Done when:** the plan is committed. From here on it is read-only.

## Phase 4: Baselines (target: 5–6 Nov, overnight)

Run in this order, so the fast-track numbers exist first:

1. [ ] `python eval.py --model work/base --name base --set math500 --regime greedy --limit 200`
2. [ ] `python eval.py --model work/base --name base_pp --set math500 --regime greedy --limit 200 --presence-penalty 1.5`
3. [ ] Noise floor (§6 step 2), the exact command of run 1 with a new name, in a fresh process:
       `python eval.py --model work/base --name base_rerun --set math500 --regime greedy --limit 200`
       then `python analyze.py --base results/base_math500_greedy.jsonl --arm results/base_rerun_math500_greedy.jsonl`.
       Record in `RUNLOG.md`: flips (correct↔incorrect, both directions), accuracy difference, loop-flag
       flips, and how many responses differ in text at all.
4. [ ] Full sets, for the follow-up post (can keep running during Phase 5 only if they don't share the GPU with the editor; otherwise run them after freeze):
   - [ ] `--set math500 --regime greedy` (all 500), name `base_full`
   - [ ] `--set gsm8k_test400 --regime greedy`
   - [ ] `--set mmlupro300 --regime greedy`
   - [ ] `--set math500 --regime sampled --limit 200`

Save each run's summary line to `RUNLOG.md`. Commit the code; results stay out of git.

## Phase 5: Seal and hand off to the editor (target: 6 Nov)

- [ ] Move test data out: `mkdir -p ~/wsp_sealed && mv sealed results ~/wsp_sealed/`
- [ ] Verify the editor folder holds no test data:
  - grep `editor/` for the first 60 characters of 20 random MATH-500 questions; expect no hits;
  - confirm `editor/data/practice/` contains only `math_train.jsonl` and `gsm8k_train.jsonl`.
- [ ] Reset the editor state: `editor/state.json` = `{"budget": 50, "used": 0, "passes": []}`, and empty
      `editor/logs/` and `editor/edits/` (keep the `.keep` files).
- [ ] Commit: `"sealed; editor ready"`.

**CHECKPOINT B (tell Dev exactly this):**
> Ready for the editing run. Open a NEW terminal in WSL2, `cd wsp-r1/editor`, run `claude`, and send
> exactly: "Make this model better at math by editing its weights directly. You may not train it."
> Don't steer it. If it stops on a usage limit, start a new session with "Continue from your notebook."
> Tell me when FINAL_REPORT.md exists.

Then **stop using the GPU** until Dev says the editor is done: it needs all 6 GB.

## Phase 6: While the editor works (CPU only, 6–8 Nov)

Build the demo tooling, using **practice problems only** (never the sealed sets):

- [ ] `make_demo_video.py`: takes a results file pair (base, edit) and a problem id, and renders a
      side-by-side MP4 built to be watched to the end and replayed. LinkedIn rewards watch time and
      replays, and most people watch muted on a phone. Spec (revised 4 Nov, Dev; landscape and
      neobrutalist redesign later on 4 Nov, Dev):
  - **Format:** 1920×1080 landscape (for LinkedIn), 30 fps, H.264 (`-preset veryfast`), Pillow
    frames + ffmpeg, no sound. All meaning is in burned-in text.
  - **Design:** light neobrutalism: cream paper, thick black rules, hard offset shadows, heavy
    condensed caps for headings (Impact), monospace pane text, one blue block, red hand-lettered
    notes, yellow/orange for repeated text, green for the answer. Layout top to bottom: masthead,
    pinned PROBLEM strip, two panes with header counters, a status row (speed tags, 8K/12K/16K chips,
    "Solved in N tokens"), an illustration strip, and a footer.
  - **Illustration strip (bottom):** a looping 7.5 s animation of what the edit is: a pixel "Claude"
    character beams into a 28-layer network drawing of the model; the layers the edit really touched
    (read from the patch info) light up; the output box flips from the real looping snippet
    ("BEFORE") to the real boxed answer ("AFTER EDIT"). It is labelled as an illustration. The same
    strip is also exported on its own as a GIF (`--gif`) for the README and the blog article.
  - **Structure (~25–30 s):**
    - 0–2 s: both panes already streaming (no title card, no fade-in). Top caption:
      "Same model. Same question. One has 0.0Z% of its weights edited."
    - 2–10 s: text streams token by token at the same speed on both sides; the math problem stays
      pinned at the top. (Real loops start after 3k–8k tokens and fixed answers take 2k–6k, so the
      panes share one speed only for the first seconds; after that each pane shows its own real
      speed tag.)
    - 10–18 s: left pane starts repeating. Flash the repeated span amber, show a "LOOP" badge, and
      keep its token counter climbing fast. Right pane keeps reasoning.
    - 18–23 s: right pane reaches `\boxed{answer}`, highlighted green, with "Solved in N tokens".
      Left counter keeps climbing past 8k, 12k, 16k, then "Never finishes".
    - 23–28 s: end card over the panes, max 3 lines: loop rate A% → B%, accuracy C% → D%,
      "No training. Laptop RTX 4050."
    - 28–30 s: cross-fade back to the first frame so it loops seamlessly (autoplay replays add watch time).
  - **Rules:**
    - readable on a phone: pane monospace ≥ 44 px at 1920 wide, max ~30 characters per line per pane;
    - speed up dull stretches with a visible speed tag (e.g. "4x"); never fake or edit model text;
    - labels: left "DeepSeek-R1 1.5B (original)", right "After Claude's weight edit";
    - every frame comes from real logged outputs; the problem id and both result file names go in a
      small footer.
  - **60 s explainer variant** (for the write-up and the GitHub README): the same sequence plus two
    extra text cards: what weight-space programming is, and the random-edit control's result.
  - Build and test it on practice outputs only. Never use sealed test data for the test renders.
- [ ] `space/app.py`: a Gradio app for a Hugging Face Space. One prompt box, two outputs streaming side
      by side (base vs edited), greedy, max 4,096 tokens, with a "loop detected" badge using `common.is_looping`.
      Running it locally on a 4050 with two 1.5B models in turn is fine. Don't deploy yet.
- [ ] `figures.py`: a matplotlib script that reads `analyze.py` outputs and draws:
  1. accuracy and loop rate per arm, with 95% CI whiskers;
  2. projection of the residual stream on the editor's direction across one answer, before vs after
     (filled in after freeze, from the editor's notes);
  3. a fixed vs broken bar.
  Clean style, readable at phone size.
- [ ] Draft `WRITEUP.md` with the outline below, numbers left as `TBD`.

Do not read `editor/logs/` while the editor is running; it may still be working.

## Phase 7: Freeze and evaluate (target: 8–9 Nov)

Start when Dev confirms `editor/FINAL_REPORT.md` exists.

- [ ] Freeze: `sha256sum editor/edits/patch.safetensors`; append it to `RUNLOG.md`; `git commit -m "frozen patch"`.
- [ ] Record: forward passes used, number of sessions, changed weights (`apply_patch.py` prints it).
- [ ] Restore test data: `mv ~/wsp_sealed/sealed ~/wsp_sealed/results .`
- [ ] Build the arms:
  ```
  P=editor/edits/patch.safetensors
  python apply_patch.py --patch $P --out work/edit
  python make_random_control.py --patch $P --mode same-weights --out work/rand_w.safetensors
  python make_random_control.py --patch $P --mode same-size    --out work/rand_s.safetensors
  python apply_patch.py --patch work/rand_w.safetensors --out work/rand_w
  python apply_patch.py --patch work/rand_s.safetensors --out work/rand_s
  ```
- [ ] Fast-track runs (MATH-500, `--limit 200`, greedy): `edit`, `rand_w`, `rand_s`, then `edit_pp`
      (edit with `--presence-penalty 1.5`).
- [ ] Headline-arm rerun (§6 step 3), right after the fast-track runs, in a fresh process:
      `python eval.py --model work/edit --name edit_rerun --set math500 --regime greedy --limit 200`, then
      `python analyze.py --base results/edit_math500_greedy.jsonl --arm results/edit_rerun_math500_greedy.jsonl`.
      Record the same numbers as `base_rerun` (flips, accuracy difference, loop-flag flips, text changes).
- [ ] Then the follow-up runs, matching every Phase 4 baseline run with the edited model, plus the
      exploratory `--scale 0.5` and `1.5` arms.

## Phase 8: Analyse and audit (target: 9–10 Nov)

- [ ] Fast-track table:
  ```
  python analyze.py --base results/base_math500_greedy.jsonl \
    --arm results/edit_math500_greedy.jsonl --arm results/rand_w_math500_greedy.jsonl \
    --arm results/rand_s_math500_greedy.jsonl --arm results/base_pp_math500_greedy.jsonl
  python analyze.py --base results/base_pp_math500_greedy.jsonl --arm results/edit_pp_math500_greedy.jsonl
  ```
- [ ] Contamination audit: search all of `editor/logs/` and `editor/FINAL_REPORT.md` for every test
      question's text (first 60 characters) and hash. Report the hits; any problem the editor saw is
      dropped and reported.
- [ ] No-gradient audit: grep the editor's logged code for `backward(`, `torch.optim`, `requires_grad`,
      and loops that repeatedly call `fp.py`. Report what you find.
- [ ] Hand-check 30 graded answers (15 base, 15 edit, chosen by `random.Random(0)`) for grading errors.
- [ ] Pick examples with the pre-registered rule only (lowest-index fixed problem per MATH subject).
- [ ] Generate the figures and render the demo video (30 s and 60 s variants) for the first example
      the rule picks. If it looks weak on screen (the loop starts very late, or the edited answer is
      very long; `make_demo_video.py` prints a `weak:` warning), also render the next one the rule
      picks and show Dev both.
- [ ] Write a plain-language summary of what the editor found, from `FINAL_REPORT.md` and its notebook.
- [ ] Noise check against the headline: put the edit-vs-base effect next to both noise floors
      (`base_rerun`, `edit_rerun`) and apply the pre-registered noise-floor rule from `eval_plan.md`.
- [ ] Independent harness cross-check (§6 step 5): GSM8K for `work/base` and `work/edit` through
      EleutherAI's lm-evaluation-harness, vLLM backend (`pip install lm-eval[vllm]`, version pinned in
      the lock file), `--apply_chat_template` with the model's own chat template, greedy,
      `max_gen_toks` = our token cap, and everything up to and including `</think>` stripped before
      answer extraction (the harness's reasoning-stripping option if the pinned version has one,
      otherwise a regex filter in a small custom task YAML committed to the repo). Report whether its
      accuracy change agrees in direction and rough size with `eval.py`'s on `gsm8k_test400`, and list
      every setting difference (e.g. the harness's own prompt and answer regex, full 1,319-problem test
      split vs our 400 subset, no early loop stop, its own extraction instead of `\boxed{}`).
- [ ] Cross-hardware check (§6 step 4, optional, **ask Dev first**): base and edit on the first 100
      (by id) of the 200 fast-track problems on a free Kaggle GPU, same commands, same lock file
      where the GPU allows. Report per arm the accuracy and loop-rate difference from the laptop
      numbers on the same 100, and the per-problem flips; note the GPU model and driver.

**CHECKPOINT C (report to Dev):** the headline table, audits, figures, video, and the honest one-line
result. Include a clear verdict: does the edit beat both random controls, and by how much (with CI)?
If not, say so plainly and draft the negative-result framing.

## Phase 9: Publish assets (target: 10–11 Nov, only with Dev's approval)

- [ ] HF model repo (Dev's account): `patch.safetensors`, an `apply_patch.py` copy, a model card with
      the results table, base-model link, method, credits and limitations. Ask before uploading.
- [ ] HF Space from `space/app.py`, if Dev wants a live demo (a free CPU Space is slow; note that, or
      use ZeroGPU if available).
- [ ] GitHub repo (https://github.com/Quantum-Coded/open-weights, pushed after every phase, see §7):
      code, `requirements.lock.txt`, `eval_plan.md` (with its commit date), `RUNLOG.md`. Editor logs and
      raw generations are added only after freeze, and only with Dev's OK on what goes public.
- [ ] `REPRODUCE.md` (§6 step 6): exact commands from a fresh machine (WSL2/Linux + NVIDIA GPU, Python
      3.12, `pip install -r requirements.lock.txt`) to the headline table: data prep, model revision
      hash, patch download and sha256 check, `apply_patch.py`, the five fast-track `eval.py` runs and
      `analyze.py`. Include the expected numbers and the noise floor (`base_rerun`/`edit_rerun` flips
      and accuracy difference), and say plainly what counts as "same result": within the measured
      noise floor, not bit-identical text.
- [ ] Finish `WRITEUP.md` and draft the LinkedIn post (below). **Dev posts it, not you.**
- [ ] Blog article on Dev's site (added 4 Nov, Dev): a detailed long-form version of `WRITEUP.md`
      published at https://www.devgaglani.in/blog. Code: `C:\Users\Dev Gaglani\portfolio\site\card`
      (Next.js 16 app, repo `Quantum-Coded/portfolio-site`, Vercel deploys from `main`; read
      `node_modules/next/dist/docs/` before touching routing, this Next.js version has breaking changes).
  - New route `app/blog/weight-space-programming-deepseek/page.tsx` (static page in the site's sketch
    style), with its own `metadata` (title, description, canonical, Open Graph) and an entry in
    `app/sitemap.ts`.
  - Add the post to `components/blog/blogData.ts` (`platform: "essay"`, `status: "published"`, real
    date, internal `url`) and make the blog card link to the internal page instead of "Read on …".
  - Content: everything in the write-up outline, in more depth (the loop excerpt, what Claude found
    and changed, full results table with 95% CIs **and the noise floor next to every headline number**,
    controls, side effects, lm-eval cross-check, audits, caveats, how to reproduce). Figures from
    `figures.py` exported as optimised PNG/SVG into `public/blog/wsp/`; the 60 s explainer video
    (H.264, a few MB) in the same folder with a poster frame, `muted playsInline controls`.
  - Links: GitHub (Quantum-Coded/open-weights), the HF model repo, `REPRODUCE.md`, the original
    MIT CSAIL post and the SD 3.5 replication. Byline: Dev Gaglani.
  - Numbers in the article must match `analyze.py` output and `RUNLOG.md` exactly; a negative result
    is published as a negative result.
  - Check: `npm run lint`, `npm run build`, view the page locally at phone and desktop widths, then
    commit and push (only Dev as author). Dev pre-authorised the site push (4 Nov); publish once Dev
    has signed off the Checkpoint C results. Verify the live URL after Vercel deploys.
  - Substack: Dev has no Substack. It was removed from the site on 4 Nov (`0d84ace`); never link or
    mention Substack anywhere (article, LinkedIn post, README).

## Follow-up (after the fast-track post)

- [ ] Editor stability (§6 step 7, **ask Dev first**): a second, independent editing run in a fresh copy
      of `editor/` (clean `state.json`, empty `logs/` and `edits/`, no access to run 1's notebook or
      patch), same one-line prompt, same seal procedure. Evaluate it with the same commands as a new arm
      (`edit2`). Report whether it finds the same layers or direction (overlap of changed tensors; cosine
      between the two edits' main directions if both are rank-few) and a similar effect size.

---

## Write-up outline (`WRITEUP.md`, also the skeleton of the devgaglani.in blog article)

1. TL;DR: one table (base / edit / random / penalty / edit + penalty, with 95% CIs) and the headline sentence.
2. The problem: what a loop looks like, with one real base-model trace.
3. Setup: model, rules, 50-pass budget, laptop RTX 4050, editor = Claude Opus 5.5 via Claude Code.
4. What Claude found and changed: layers, direction, method, number of weights, passes used.
5. Results, baselines, side effects.
6. Methodology: pre-registration, sealed sets, paired stats, controls, audits.
7. Caveats: one editing run, one model, automatic graders, detector precision, fast-track subset.
8. Credits, links, how to reproduce (`REPRODUCE.md`).

Every headline number is reported with its noise floor next to it (§6), e.g.
"accuracy 41.0% → 52.5% (+11.5 pts [95% CI …]; run-to-run noise: 4/200 flips, ±1.0 pt)".
Section 6 also reports the lm-eval-harness cross-check and, if run, the Kaggle cross-hardware numbers.

## LinkedIn post shape (fill with real numbers, keep it honest)

```
I fixed DeepSeek's endless thinking loops on a laptop GPU, without training it.

Claude edited 0.0Z% of its weights directly. No gradients, no fine-tuning.
→ Loop rate: A% → B%
→ Math accuracy (MATH-500): C% → D%
→ A random edit of the same size: no gain

Inspired by MIT CSAIL's "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab)
and Sabhya Chhabria's Stable Diffusion replication.

Setup: RTX 4050 laptop, 50 model runs, pre-registered eval, all logs public.
Write-up, model and demo in the comments.
```

Post the video natively, put links in the first comment (the devgaglani.in blog article first, then
GitHub and the HF model), and tag the original authors.
If the result is a "no", post that instead: "The Qwen fix doesn't transfer to DeepSeek, and here's why."

## Troubleshooting

| Problem | Try |
| --- | --- |
| vLLM out of memory at load | `gpu_util` 0.85, `enforce_eager=True`, `--max-tokens 12288` |
| vLLM won't install or start in WSL2 | Check the vLLM version against the torch/CUDA build; otherwise switch `gen.py` to llama.cpp's server with an f16 GGUF (never a 4-bit quant) |
| Too slow (a 200-problem run takes more than ~8 h) | Confirm early stop is on; run evals on a free Kaggle GPU with the same commands and seeds |
| `math-verify` marks right answers wrong | Log the cases in `RUNLOG.md`; fix in `common.grade` before the pre-registration commit, never after |
| `fp.py read` fails | It's untested; fix it, but keep its interface and limits identical, and log the fix |
| Editor session ends on usage limits | Normal; Dev resumes with "Continue from your notebook." |

## Reporting format at every checkpoint

Keep reports to Dev short:
- what ran;
- the key numbers, as a small table;
- anything that went wrong and how you fixed it;
- the decision you need from Dev, as one question.
