# Amendment 5: the Rubik's cube, with the loop detector off

- Written by the operator on 2026-11-21 after Dev asked for a Rubik's cube for the post ("rubiks cube is a much
  better visual hook"), with real data behind it.
- Committed after the base-only pilot (results below) and **before any model runs on the test set**, as in
  Amendment 4. The earlier file `eval_plan_amendment5_DRAFT.md` is superseded by this one.
- Run 2's edit (`work/edit_r2`) is used unchanged. Nobody edits weights for this amendment; the editor is not
  involved and gets no message.
- Status: confirmatory for the two primary tests below. Everything else is secondary or descriptive. It never
  replaces or reinterprets `eval_plan.md` or Amendments 1–4.

## Why

1. **A second application nobody edited for**, and one people recognise at a glance.
2. **Real cost.** Every earlier test ran with the loop detector on, which stops a looping answer early and so
   hides most of what a loop costs. A user's chat app has no such detector: a loop runs until the token limit.
   Here the detector is **off**, so the token counts and the wall-clock times are what a user would pay. (The
   Game of 24 race recording showed one such loop running to the 16,384-token limit.)

## Task and grading (`cube.py`)

- A solved 3×3 cube is scrambled with a short sequence of face turns. The prompt (v2, `cube.make_row`) first
  explains the notation (U D L R F B, `'` = counter-clockwise, `2` = half turn), then says the cube starts solved
  and is scrambled with "this sequence of N moves", with the scramble on its own line, then asks for a move
  sequence that returns the cube to solved as the final answer, plus the standard `\boxed{}` instruction.
- Correct only if the model finished thinking (`</think>`) and the last `\boxed{}` of the answer is a sequence of
  the 18 face turns that, applied after the scramble in a sticker-level simulator, leaves every face one colour.
  Any valid solution counts, not only the inverse of the scramble. Primes written as `’`, `′`, `^{\prime}`, `3`
  (three quarter turns) and separators such as commas or arrows are accepted. Slice moves, wide moves and cube
  rotations are not. No LLM judge.
- The simulator passed a self-test before this draft: each turn four times is the identity; (R U R' U') six times
  is the identity and not before; R, U and F move stickers the standard way; every scramble followed by its
  inverse is solved; hand-made answers are graded as intended.

## Data (`a5_make_data.py`; sha256 in RUNLOG when built)

- Scrambles: random face turns, no two turns of one face in a row, no three turns on one axis in a row.
- **Pilot** `a5_data/cube_pilot12` (prompt v1) and `a5_data/cube_pilot12_v2` (prompt v2, same scrambles):
  2 scrambles at each depth 1–6, `random.Random(20261121)`. Base model only, greedy, loop detector on. Used only
  to check the prompt and grader. Never a test set.
- **Test** `sealed/cube_100`: 20 scrambles at each depth 2–6 (100 puzzles), `random.Random(20261122)`, none equal
  to a pilot scramble. Depths are fixed here, not chosen from results. (The draft said depths 1–5, but depth 1
  has only 18 distinct scrambles, 2 of them used in the pilot, so 20 new ones do not exist; the builder looped
  forever. Depth 1 is dropped and depth 6 added. This is a feasibility fix made before any test run.)

## Pilot (base only, greedy, loop detector on; 22 Nov)

- Prompt v1 (`results/pilot_a5-cube_pilot12_greedy.jsonl`, 4.8 min): 0 of 12 correct, all 12 stopped by the loop
  detector. In 2 of 12 the model misread the prompt: on the 1-move scramble "F." it did not notice the scramble
  ("I don't know exactly what moves were made"), and on another it read the notation key that followed the
  scramble ("U, D, L, R, F, B") as more scramble moves. The other 10 read the scramble correctly and looped.
- So the prompt was fixed (v2: notation first, then the scramble on its own line with its length). Scrambles,
  depths, grader and test size are unchanged.
- Prompt v2 (`results/pilot2_a5-cube_pilot12_v2_greedy.jsonl`, 4.5 min): 0 of 12 correct, all 12 stopped by the
  loop detector, and all 12 restate the scramble correctly. They loop on the inversion itself, for example
  "Wait, perhaps I can think about the effect of the inverse sequence on the cube." repeated. Even the 1-move
  scrambles loop.
- No answer finished, so the grader was not exercised on real answers; it was checked on hand-made answers
  (self-test). No edited model ran on either pilot.

## Runs

| Set | Arm | Decoding | Loop detector | Runner |
|---|---|---|---|---|
| `a5:cube_pilot12`, `a5:cube_pilot12_v2` | base | greedy | on | `a5_eval.py` (pilot, done) |
| `cube_100` | base | greedy | **off** | `a5_eval.py --no-early-stop` |
| `cube_100` | e2 | greedy | **off** | `a5_eval.py --no-early-stop` |

Both test runs: 16,384-token cap, batch-invariant kernels, the same code, GPU and vLLM settings, one after the
other on an otherwise idle GPU. The fine-tune is not run (cost; the head-to-head is Amendment 4's family H).

If the pilot shows the prompt or grader is broken (for example, correct answers in a form the grader rejects),
they are fixed and the pilot rerun before this amendment is committed; that is reported. The depths and test
size do not change because of the pilot.

## Primary tests (family R, Holm over 2, alpha 0.05)

- R1: e2 vs base, accuracy on `cube_100`. Exact two-sided McNemar, paired by puzzle; 95% bootstrap CI of the
  difference (10,000 resamples of puzzles).
- R2: e2 vs base, tokens per puzzle on `cube_100`. Two-sided Wilcoxon signed-rank test, paired by puzzle. Reported
  with the ratio of total tokens (e2 / base) and its 95% bootstrap CI.
- Claims: "solves more cubes" only if R1 passes in e2's favour; "uses fewer tokens on the cube" only if R2 passes
  in e2's favour. If R1 passes in base's favour, that is reported just as prominently.

## Secondary and descriptive (reported, not used for claims)

- Wall-clock time of each test run (load + generate), total tokens, answers that hit the cap, answers where the
  loop detector would have fired, accuracy and tokens by depth.
- Tokens per **solved** cube (total tokens / cubes solved), by arm.
- **Video race** (descriptive, for the post): one puzzle recorded single-stream for each arm with
  `record_race.py` (greedy, no loop kill, 16,384 cap), as for Game of 24. The puzzle is fixed by this rule now:
  among puzzles e2 solves, the one whose token difference (base − e2) is the median; ties go to the lowest id.
  It is described as "a typical cube the edit solved", never as the average. The post's headline numbers are the
  totals over all 100 puzzles, not the race.

## Order

1. Pilot (base), done. Build `cube_100` (sha256 in RUNLOG), commit and push this amendment.
2. `cube_100`: base, then e2.
3. Analysis (`amendment5_primary.py`), then the race recording.
