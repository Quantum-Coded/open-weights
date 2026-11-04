"""Run one model on one test set and save graded results.

  python eval.py --model work/base --name base --set math500 --regime greedy
  python eval.py --model work/edit --name edit --set math500 --regime greedy
  python eval.py --model work/base --name base_pp --set math500 --regime greedy --presence-penalty 1.5
  python eval.py --model work/base --name pilot --set practice:math_train --limit 20 --no-early-stop
  VLLM_BATCH_INVARIANT=0 python eval.py ... --name base_rerun   # noise floor: normal kernels

--set is a sealed set name (math500, gsm8k_test400, mmlupro300) or practice:<name>.
Regimes (pre-registered in eval_plan.md):
  greedy   temperature 0, 1 sample
  sampled  temperature 0.6, top-p 0.95, 2 samples (DeepSeek's recommended settings)
Output: results/<name>_<set>_<regime>.jsonl, one line per answer.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import PRACTICE_DIR, RESULTS_DIR, SEALED_DIR, grade, read_jsonl, write_jsonl
from gen import batch_invariant, free_llm, generate, load_llm

REGIMES = {"greedy": dict(temperature=0.0, top_p=1.0, n_samples=1),
           "sampled": dict(temperature=0.6, top_p=0.95, n_samples=2)}


def load_set(name: str) -> list[dict]:
    if name.startswith("practice:"):
        return read_jsonl(PRACTICE_DIR / f"{name.split(':', 1)[1]}.jsonl")
    path = SEALED_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Sealed sets are moved out while Claude edits; "
                         "put them back only after the patch is frozen.")
    return read_jsonl(path)


def summarize(res: list[dict]) -> dict:
    n = len(res)
    return {"n": n,
            "accuracy": sum(r["correct"] for r in res) / n,
            "loop_rate": sum(r["loop"] for r in res) / n,
            "cap_rate": sum(r["status"] == "cap" for r in res) / n,
            "mean_tokens_finished": (lambda f: sum(f) / len(f) if f else 0)(
                [r["n_tokens"] for r in res if r["status"] == "finished"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--regime", choices=REGIMES, default="greedy")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None, help="fixed random subset (seed 0)")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    a = ap.parse_args()

    rows = load_set(a.set)
    if a.limit:
        rows = sorted(random.Random(0).sample(rows, a.limit), key=lambda r: r["id"])
    by_id = {r["id"]: r for r in rows}

    llm = load_llm(a.model, max_len=a.max_tokens + 1024)
    res = generate(llm, rows, max_tokens=a.max_tokens, early_stop=not a.no_early_stop,
                   presence_penalty=a.presence_penalty, **REGIMES[a.regime])
    free_llm(llm)

    for r in res:
        r["correct"] = grade(by_id[r["id"]], r["response"])
    tag = a.set.replace(":", "-")
    out = RESULTS_DIR / f"{a.name}_{tag}_{a.regime}.jsonl"
    write_jsonl(out, res)

    s = summarize(res)
    print(json.dumps(s, indent=2))
    if a.no_early_stop:   # detector validation for the pilot
        flagged = [r for r in res if r["first_flag"] is not None]
        fp = [r for r in flagged if r["status"] == "finished" and r["correct"]]
        print(f"detector fired on {len(flagged)}/{len(res)}; of those {len(fp)} finished correctly anyway "
              f"(false-positive rate {len(fp) / max(len(flagged), 1):.1%})")
    Path(str(out) + ".summary.json").write_text(json.dumps(
        {**s, **vars(a), "batch_invariant": batch_invariant()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
