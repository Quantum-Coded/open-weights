"""Amendment 4 runner: eval.py's generation and loop kill, plus the Game of 24 grader and a 1-sample T=0.6 regime.

  python a4_eval.py --model work/base --name base --set game24_300 --regime greedy
  python a4_eval.py --model work/edit_r2 --name e2 --set math_fresh1200 --regime sampled1
  python a4_eval.py --model work/base --name pilot --set a4:game24_pilot20 --regime greedy

Regimes: greedy (T 0, 1 sample), sampled1 (T 0.6, top-p 0.95, 1 sample; DeepSeek's recommended settings).
Output: results/<name>_<set>_<regime>.jsonl (+ .summary.json), the same fields as eval.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import RESULTS_DIR, SEALED_DIR, grade, read_jsonl, write_jsonl
from eval import summarize
from gen import batch_invariant, free_llm, generate, load_llm
from game24 import grade_game24

REGIMES = {"greedy": dict(temperature=0.0, top_p=1.0, n_samples=1),
           "sampled1": dict(temperature=0.6, top_p=0.95, n_samples=1)}
A4 = Path(__file__).resolve().parent / "a4_data"


def load_set(name: str) -> list[dict]:
    path = A4 / f"{name.split(':', 1)[1]}.jsonl" if name.startswith("a4:") else SEALED_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return read_jsonl(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--regime", choices=REGIMES, default="greedy")
    ap.add_argument("--max-tokens", type=int, default=16384)
    a = ap.parse_args()

    rows = load_set(a.set)
    by_id = {r["id"]: r for r in rows}
    # Operational hold (13 Nov): ops/kv_verify.sh checks the KV-grouped generation path on the free
    # GPU before the long sampled MATH runs. Nothing is loaded on the GPU while holding.
    hold = Path(__file__).resolve().parent / "HOLD_A4"
    if a.set == "math_fresh1200" and hold.exists():
        print(f"HOLD {time.strftime('%FT%T')}", flush=True)
        while hold.exists():
            time.sleep(30)
        print(f"RELEASED {time.strftime('%FT%T')}", flush=True)
    t0 = time.time()
    llm = load_llm(a.model, max_len=a.max_tokens + 1024)
    res = generate(llm, rows, max_tokens=a.max_tokens, early_stop=True, **REGIMES[a.regime])
    free_llm(llm)
    wall = time.time() - t0

    res.sort(key=lambda r: (r["id"], r["sample"]))
    for r in res:
        row = by_id[r["id"]]
        r["correct"] = grade_game24(row, r["response"]) if row["kind"] == "game24" else grade(row, r["response"])
    tag = a.set.replace(":", "-")
    out = RESULTS_DIR / f"{a.name}_{tag}_{a.regime}.jsonl"
    write_jsonl(out, res)
    s = summarize(res)
    s["total_tokens"] = sum(r["n_tokens"] for r in res)
    s["wall_s"] = round(wall, 1)
    print(json.dumps(s, indent=2))
    Path(str(out) + ".summary.json").write_text(json.dumps(
        {**s, **vars(a), "batch_invariant": batch_invariant()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
