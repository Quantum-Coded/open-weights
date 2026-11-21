"""Amendment 5 runner: the Rubik's cube. gen.py's generation, the cube grader, greedy, 16,384-token cap.

  python a5_eval.py --model work/base --name pilot --set a5:cube_pilot12              # loop kill on
  python a5_eval.py --model work/base --name base --set cube_100 --no-early-stop      # the test runs
  python a5_eval.py --model work/edit_r2 --name e2 --set cube_100 --no-early-stop

The test runs keep the loop detector OFF, so every answer runs until it stops or hits the cap: the token counts
and wall-clock times are what a user would see. The detector still records where it would have fired.
Output: results/<name>_<set>_greedy[_nokill].jsonl (+ .summary.json), the same fields as eval.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import RESULTS_DIR, SEALED_DIR, read_jsonl, write_jsonl
from cube import grade_cube
from eval import summarize
from gen import batch_invariant, free_llm, generate, load_llm

A5 = Path(__file__).resolve().parent / "a5_data"


def load_set(name: str) -> list[dict]:
    path = A5 / f"{name.split(':', 1)[1]}.jsonl" if name.startswith("a5:") else SEALED_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return read_jsonl(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--no-early-stop", action="store_true")
    a = ap.parse_args()

    rows = load_set(a.set)
    by_id = {r["id"]: r for r in rows}
    t0 = time.time()
    llm = load_llm(a.model, max_len=a.max_tokens + 1024)
    t_loaded = time.time()
    res = generate(llm, rows, max_tokens=a.max_tokens, early_stop=not a.no_early_stop,
                   temperature=0.0, top_p=1.0, n_samples=1)
    t_gen = time.time() - t_loaded
    free_llm(llm)
    wall = time.time() - t0

    res.sort(key=lambda r: (r["id"], r["sample"]))
    for r in res:
        r["correct"] = grade_cube(by_id[r["id"]], r["response"])
        r["depth"] = by_id[r["id"]]["depth"]
    tag = a.set.replace(":", "-") + "_greedy" + ("_nokill" if a.no_early_stop else "")
    out = RESULTS_DIR / f"{a.name}_{tag}.jsonl"
    write_jsonl(out, res)
    s = summarize(res)
    s["total_tokens"] = sum(r["n_tokens"] for r in res)
    s["detector_would_fire"] = sum(r["first_flag"] is not None for r in res)
    s["wall_s"] = round(wall, 1)
    s["generate_s"] = round(t_gen, 1)
    s["by_depth"] = {d: {"n": len(x), "correct": sum(r["correct"] for r in x),
                         "tokens": sum(r["n_tokens"] for r in x)}
                     for d in sorted({r["depth"] for r in res})
                     for x in [[r for r in res if r["depth"] == d]]}
    print(json.dumps(s, indent=2))
    Path(str(out) + ".summary.json").write_text(json.dumps(
        {**s, **vars(a), "batch_invariant": batch_invariant()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
