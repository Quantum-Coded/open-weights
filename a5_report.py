"""Amendment 5 report: turns results/a5_primary.json (+ race recordings) into
  - a RUNLOG.md entry (appended),
  - a WRITEUP.md section (appended once, replaced on rerun between the A5 markers),
  - POST_FACTS.md (numbers and the claim wording the pre-registered tests allow, for Dev's LinkedIn post).

  python a5_report.py [--primary results/a5_primary.json] [--facts POST_FACTS.md] [--no-repo-edits]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def hms(s):
    if s is None:
        return "n/a"
    s = int(round(s))
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=str(ROOT / "results/a5_primary.json"))
    ap.add_argument("--race-dir", default=str(ROOT / "results/race"))
    ap.add_argument("--facts", default=str(ROOT / "POST_FACTS.md"))
    ap.add_argument("--video", default="(not rendered)")
    ap.add_argument("--no-repo-edits", action="store_true", help="dry run: write only the facts file")
    a = ap.parse_args()
    P = json.loads(Path(a.primary).read_text())
    b, e, r1, r2 = P["base"], P["e2"], P["R1"], P["R2"]
    red = 100 * (1 - r2["ratio"]) if r2["ratio"] is not None else None
    tb, te = b.get("wall_s"), e.get("wall_s")
    speed = (tb / te) if tb and te else None

    race = None
    if P["race_id"]:
        race = {}
        for arm in ("base", "e2"):
            p = Path(a.race_dir) / f"{arm}_{P['race_id']}.json"
            if p.exists():
                x = json.loads(p.read_text())
                race[arm] = {k: x.get(k) for k in ("n_tokens", "wall_s", "finish_reason", "correct", "matches_eval_text")}

    def verdict(r):
        return "PASS (in the edit's favour)" if r["pass_e2"] else "PASS (in the ORIGINAL's favour)" if r["pass_base"] else "FAIL"

    lines = [
        f"- R1 accuracy: original {r1['base_acc']:.0f}% -> edit {r1['e2_acc']:.0f}% ({r1['diff']:+.0f} points, 95% CI "
        f"[{r1['ci'][0]:+.0f}, {r1['ci'][1]:+.0f}]); fixed {r1['fixed']}, broken {r1['broken']}; exact McNemar p "
        f"{r1['p']:.3g}, Holm p {r1['p_holm']:.3g}: **{verdict(r1)}**.",
        f"- R2 tokens: original {r2['base_tokens']:,} -> edit {r2['e2_tokens']:,} (edit/original {r2['ratio']:.3f}, 95% CI "
        f"[{r2['ratio_ci'][0]:.3f}, {r2['ratio_ci'][1]:.3f}]); Wilcoxon signed-rank z {r2['z']:.2f}, p {r2['p']:.3g}, "
        f"Holm p {r2['p_holm']:.3g}: **{verdict(r2)}**.",
        f"- Wall clock of the batched test runs (load + generate): original {hms(tb)}, edit {hms(te)}"
        + (f" ({speed:.1f}x)." if speed else "."),
        f"- Answers that hit the 16,384-token limit: original {b['cap']}, edit {e['cap']}. Loop detector would have "
        f"fired: original {b['detector_would_fire']}, edit {e['detector_would_fire']}.",
        f"- Tokens per solved cube: original {('%.0f' % b['tokens_per_solved']) if b['tokens_per_solved'] else 'none solved'}, "
        f"edit {('%.0f' % e['tokens_per_solved']) if e['tokens_per_solved'] else 'none solved'}.",
        "- By depth (solved / tokens): " + "; ".join(
            f"{d} moves: original {b['by_depth'][d]['solved']}/{b['by_depth'][d]['n']} ({b['by_depth'][d]['tokens']:,}), "
            f"edit {e['by_depth'][d]['solved']}/{e['by_depth'][d]['n']} ({e['by_depth'][d]['tokens']:,})"
            for d in sorted(b["by_depth"], key=int)) + ".",
        f"- Race puzzle (pre-registered rule): {P['race_id'] or 'none (the edit solved no cube)'}"
        + (f"; single-stream: original {race['base']['n_tokens']:,} tokens in {race['base']['wall_s']:.0f} s "
           f"({race['base']['finish_reason']}, correct {race['base']['correct']}), edit {race['e2']['n_tokens']:,} tokens "
           f"in {race['e2']['wall_s']:.0f} s (correct {race['e2']['correct']}); recordings match the batched text: "
           f"original {race['base']['matches_eval_text']}, edit {race['e2']['matches_eval_text']}."
           if race and "base" in race and "e2" in race else "."),
        f"- Video: {'cube cut' if P['video_mode'] == 'cube' else 'Game of 24 cut (fallback rule in the addendum)'}; {a.video}.",
    ]
    claims = P["claims_allowed"]
    now = dt.datetime.now(IST).strftime("%d %b %Y, %H:%M IST")

    facts = [
        "# Post facts: Rubik's cube (Amendment 5)",
        "",
        f"Generated {now} by `a5_report.py` from `results/a5_primary.json`. Every number here comes from the",
        "pre-registered test runs (100 fresh scrambles, 2 to 6 moves, greedy, loop detector OFF, 16,384-token limit,",
        "DeepSeek-R1-Distill-Qwen-1.5B on an RTX 4050 laptop GPU). Original model vs the same model with 2 MLP neurons",
        "in layer 19 rewritten (9,215 weights), no training.",
        "",
        "## Headline",
        "",
        f"- Cubes solved: **{b['solved']} / 100 -> {e['solved']} / 100**",
        f"- Tokens over all 100 scrambles: **{b['total_tokens']:,} -> {e['total_tokens']:,}**"
        + (f" (**{red:.0f}% fewer**)" if red is not None and red > 0 else ""),
        f"- Time for all 100 on the laptop: **{hms(tb)} -> {hms(te)}**" + (f" (**{speed:.1f}x faster**)" if speed and speed > 1 else ""),
        "",
        "## What you may claim (pre-registered tests, Holm over 2)",
        "",
    ]
    if claims:
        facts += [f"- \"The edit {c}.\"" for c in claims]
    else:
        facts.append("- Neither primary test passed in the edit's favour: describe the numbers, do not claim an improvement on the cube.")
    for w in P["warnings"]:
        facts.append(f"- WARNING: {w}.")
    facts += ["", "## Full numbers", ""] + lines + [
        "",
        "## Say this too (honesty lines)",
        "",
        "- Fresh scrambles made after the edit was fixed; nobody edited or tuned anything for the cube.",
        "- The race in the video is one puzzle picked by a rule fixed before any run (the edit-solved puzzle with the "
        "median token saving), replayed from real per-token timestamps; the two models were recorded one after the other.",
        "- Time = this laptop (RTX 4050, 6 GB), batched runs; other hardware gives other absolute times.",
        "- Accuracy on hard maths (MATH, 1,200 fresh problems) did not change significantly (85.8% -> 86.4%, Amendment 4 P3).",
        "",
        "Repo: https://github.com/Quantum-Coded/open-weights",
    ]
    Path(a.facts).write_text("\n".join(facts) + "\n", encoding="utf-8")
    print(f"wrote {a.facts}")
    if a.no_repo_edits:
        return

    entry = [f"\n## {now}: Amendment 5 (Rubik's cube) done", "",
             "- Pipeline `ops/a5_full.sh`; analysis `amendment5_primary.py` -> `results/a5_primary.json`, log "
             "`logs_a5_primary.txt`.", *lines,
             f"- Claims allowed: {', '.join(claims) if claims else 'none'}."]
    with open(ROOT / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write("\n".join(entry) + "\n")

    w = ROOT / "WRITEUP.md"
    txt = w.read_text(encoding="utf-8")
    start, end = "<!-- A5 START -->", "<!-- A5 END -->"
    sec = "\n".join([start, "", "### 5.7 A second application nobody edited for: the Rubik's cube (Amendment 5)", "",
                     "100 fresh scrambles (2 to 6 face turns, 20 at each depth). The model must give a move sequence "
                     "that returns the cube to solved; a sticker-level simulator grades it (any valid solution counts). "
                     "Greedy, loop detector **off**, so tokens and time are what a user would pay. Pre-registered "
                     "tests R1 (accuracy, McNemar) and R2 (tokens, Wilcoxon), Holm over 2.", "", *lines, "",
                     f"Claims allowed: {', '.join(claims) if claims else 'none'}.", "", end])
    if start in txt:
        txt = txt[: txt.index(start)] + sec + txt[txt.index(end) + len(end):]
    else:
        anchor = "## 6. Methodology"
        txt = txt.replace(anchor, sec + "\n\n" + anchor, 1) if anchor in txt else txt + "\n" + sec + "\n"
    w.write_text(txt, encoding="utf-8")
    print("appended RUNLOG entry and WRITEUP section")


if __name__ == "__main__":
    main()
