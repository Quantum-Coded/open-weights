# Amendment 5 addendum: automation, tie-breaks and the video fallback

- Written by the operator on 2026-11-23, **before any model runs on `sealed/cube_100`**. It adds details that
  Amendment 5 left open; it changes none of its tests, data, decoding or claim wording.
- Why: Dev asked for the whole of Amendment 5 to run unattended, from the test runs to the rendered video, so
  every choice that would otherwise be made after seeing results is fixed here.

## Pipeline

`start_a5.ps1` (Windows) keeps the laptop awake and runs `ops/a5_full.sh` (WSL, tmux `a5full`):
base run, e2 run, `amendment5_primary.py`, the race recording (`record_race.py`), `a5_video_data.py`,
`a5_report.py` (RUNLOG, WRITEUP section, `POST_FACTS.md`), commit and push; then the video render on Windows.
It was tested end to end (`DRY=1`) on the 12 **pilot** puzzles with a 1,024-token cap under `dry_*` names; no
model ran on any test puzzle. `record_race.py` gained `--max-tokens` (default 16,384, unchanged) and cube grading.

## Details fixed now

1. **R2 Wilcoxon:** zero differences dropped, normal approximation with tie correction, no continuity correction,
   two-sided (scipy's defaults for n > 50; scipy is not installed, so it is implemented in the script).
   Bootstrap CIs: 10,000 resamples of puzzles, `random.Random(0)`, percentile method.
2. **Race puzzle "median":** among puzzles e2 solves, sort the token differences (base − e2); the median is the
   lower median when the count is even; among puzzles with that difference, the lowest id.
3. **Video mode:** the cube cut is rendered only if e2 solves at least one cube **and** R2 passes in e2's favour.
   Otherwise the earlier Game of 24 cut (real recordings, RUNLOG 21 Nov) is rendered instead, and the cube numbers
   go only into the write-up. The cube cut shows the race puzzle's real recordings, the per-puzzle solved/unsolved
   boards for all 100 test puzzles, and the total tokens and batched wall-clock time of both test runs.
4. **Post wording:** `POST_FACTS.md` lists only the claims Amendment 5 allows for the results ("solves more cubes"
   if R1 passes in e2's favour; "uses fewer tokens on the cube" if R2 does), plus any result in base's favour.
