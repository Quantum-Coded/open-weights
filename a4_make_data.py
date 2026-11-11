"""Amendment 4 data (operator, 2026-11-11, built before any model runs on it).

  python a4_make_data.py pilot              # a4_data/game24_pilot20.jsonl (base-only pilot, never a test set)
  python a4_make_data.py test --n24 N       # sealed/game24_<N>.jsonl, sealed/math_fresh1200.jsonl,
                                            # a4_data/ft_problems1200.jsonl

Game of 24: all 1,362 solvable multisets of four numbers 1-13, shuffled with random.Random(20261111).
  Puzzles 0-19 of that order are the pilot; the next N are the sealed test set.
math_fresh1200: MATH test problems (EleutherAI/hendrycks_math, all 7 subjects, 5,000) that are not MATH-500
  problems and do not match an editor practice problem (Phase 5 rule, as in make_fresh.py). Shuffled with
  random.Random(20261112), the first 1,200 kept. Answer = the last \\boxed{} of the reference solution.
ft_problems1200: 600 math_train + 600 gsm8k_train practice problems (the editor's practice pools), drawn with
  random.Random(20261113). Training data for the LoRA fine-tune (fine-tuning sees practice data only).
"""
import difflib, hashlib, json, random, re, sys
from pathlib import Path
sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from common import PRACTICE_DIR, SEALED_DIR, last_boxed, read_jsonl, write_jsonl
from game24 import all_puzzles, make_row

A4 = Path("/home/dev_gaglani/wsp-r1/a4_data")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def game24_order():
    p = all_puzzles()
    assert len(p) == 1362
    random.Random(20261111).shuffle(p)
    return p


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    mode = sys.argv[1]
    order = game24_order()
    if mode == "pilot":
        out = A4 / "game24_pilot20.jsonl"
        write_jsonl(out, [make_row(t) for t in order[:20]])
        print("pilot", out, sha(out))
        return
    assert mode == "test" and sys.argv[2] == "--n24"
    n24 = int(sys.argv[3])
    g = sorted([make_row(t) for t in order[20:20 + n24]], key=lambda r: r["id"])
    out = SEALED_DIR / f"game24_{n24}.jsonl"
    write_jsonl(out, g)
    print(f"game24 test: {len(g)} puzzles -> {out}  sha256 {sha(out)}")

    # ---- fresh MATH
    from datasets import load_dataset
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    pool = []
    for s in subjects:
        for i, x in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="test")):
            pool.append((s, i, x))
    m500 = {norm(r["question"]) for r in read_jsonl(SEALED_DIR / "math500.jsonl")}
    pp = {}
    for f in ("math_train", "gsm8k_train"):
        for r in read_jsonl(PRACTICE_DIR / f"{f}.jsonl"):
            q = norm(r["question"])
            pp.setdefault(q[:40], []).append(q)

    def contaminated(q):
        q = norm(q)
        return any(x[:60] == q[:60] or difflib.SequenceMatcher(None, q, x).ratio() >= 0.80
                   for x in pp.get(q[:40], []))

    rng = random.Random(20261112)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    keep, skip500, skipprac, skipans = [], 0, 0, 0
    for k in idx:
        if len(keep) == 1200:
            break
        s, i, x = pool[k]
        q = norm(x["problem"])
        if q in m500:
            skip500 += 1; continue
        if contaminated(x["problem"]):
            skipprac += 1; continue
        ans = last_boxed(x["solution"])
        if not ans:
            skipans += 1; continue
        lvl = re.findall(r"\d", x.get("level", ""))
        keep.append({"id": f"mf-{s}-{i}", "kind": "math", "question": x["problem"], "answer": ans,
                     "subject": s, "level": int(lvl[0]) if lvl else None})
    keep.sort(key=lambda r: r["id"])
    out = SEALED_DIR / "math_fresh1200.jsonl"
    write_jsonl(out, keep)
    print(f"MATH test pool {len(pool)}; skipped {skip500} MATH-500, {skipprac} practice matches, "
          f"{skipans} without a boxed answer; kept {len(keep)} -> {out}  sha256 {sha(out)}")
    from collections import Counter
    print("  subjects", dict(Counter(r["subject"] for r in keep)), "levels", dict(Counter(r["level"] for r in keep)))

    # ---- fine-tune training problems (practice pools only)
    rng = random.Random(20261113)
    ft = []
    for f in ("math_train", "gsm8k_train"):
        rows = read_jsonl(PRACTICE_DIR / f"{f}.jsonl")
        ft += rng.sample(rows, 600)
    out = A4 / "ft_problems1200.jsonl"
    write_jsonl(out, ft)
    print(f"ft problems {len(ft)} -> {out}  sha256 {sha(out)}")
    # sanity: no test question in the training pool
    test_q = {norm(r["question"]) for f in SEALED_DIR.glob("*.jsonl") for r in read_jsonl(f) if "question" in r}
    assert not any(norm(r["question"]) in test_q for r in ft)
    print("ft pool has no exact overlap with any sealed set")


if __name__ == "__main__":
    main()
