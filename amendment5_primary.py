"""Amendment 5 analysis, as pre-registered in eval_plan_amendment5.md (+ eval_plan_amendment5_addendum.md).

  python amendment5_primary.py                                   # the test runs on cube_100
  python amendment5_primary.py --base results/dry_base_... --e2 results/dry_e2_... --out results/dry_a5_primary.json

Family R (Holm over 2, alpha 0.05):
  R1 accuracy: exact two-sided McNemar, paired by puzzle; 95% bootstrap CI of the difference (10,000 resamples).
  R2 tokens per puzzle: two-sided Wilcoxon signed-rank, paired by puzzle (zero differences dropped, normal
     approximation with tie correction, no continuity correction: scipy's defaults for n > 50); reported with the
     ratio of total tokens (e2 / base) and its 95% bootstrap CI.
Race puzzle (rule fixed before any test run): among puzzles e2 solves, the one whose token difference
(base - e2) is the median (the lower median when the count is even); ties go to the lowest id.
Video mode (addendum): "cube" if e2 solves at least one cube and R2 passes in e2's favour, else "game24".
Writes a JSON with everything the report and the video need.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def mcnemar(f, b):
    n, k = f + b, min(f, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank on paired differences d. Returns (W+, z, p, n_used)."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i, ties = 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return wp, 0.0, 1.0, n
    z = (wp - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return wp, z, min(1.0, p), n


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1) + 0.5)]


def arm_stats(X, summary_path):
    rs = list(X.values())
    s = json.loads(Path(str(summary_path) + ".summary.json").read_text()) if Path(str(summary_path) + ".summary.json").exists() else {}
    solved = sum(r["correct"] for r in rs)
    tot = sum(r["n_tokens"] for r in rs)
    by_depth = {}
    for r in rs:
        d = by_depth.setdefault(str(r["depth"]), {"n": 0, "solved": 0, "tokens": 0})
        d["n"] += 1
        d["solved"] += bool(r["correct"])
        d["tokens"] += r["n_tokens"]
    return {"n": len(rs), "solved": solved, "accuracy": solved / len(rs), "total_tokens": tot,
            "cap": sum(r["status"] == "cap" for r in rs), "finished": sum(r["status"] == "finished" for r in rs),
            "detector_would_fire": sum(r.get("first_flag") is not None for r in rs),
            "no_think_end": sum("</think>" not in r["response"] for r in rs),
            "tokens_per_solved": (tot / solved) if solved else None,
            "wall_s": s.get("wall_s"), "generate_s": s.get("generate_s"), "by_depth": by_depth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "results/base_cube_100_greedy_nokill.jsonl"))
    ap.add_argument("--e2", default=str(ROOT / "results/e2_cube_100_greedy_nokill.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/a5_primary.json"))
    a = ap.parse_args()

    B, E = load(a.base), load(a.e2)
    ids = sorted(B)
    assert sorted(E) == ids, "base and e2 do not pair up"
    rng = random.Random(0)

    # R1 accuracy
    fx = sum(1 for i in ids if E[i]["correct"] and not B[i]["correct"])
    br = sum(1 for i in ids if B[i]["correct"] and not E[i]["correct"])
    dacc = {i: int(E[i]["correct"]) - int(B[i]["correct"]) for i in ids}
    boots = sorted(100 * sum(dacc[rng.choice(ids)] for _ in ids) / len(ids) for _ in range(10000))
    acc_b = 100 * sum(B[i]["correct"] for i in ids) / len(ids)
    acc_e = 100 * sum(E[i]["correct"] for i in ids) / len(ids)
    r1 = {"base_acc": acc_b, "e2_acc": acc_e, "diff": acc_e - acc_b, "fixed": fx, "broken": br,
          "p": mcnemar(fx, br), "ci": [pct(boots, 0.025), pct(boots, 0.975)]}

    # R2 tokens
    d = [E[i]["n_tokens"] - B[i]["n_tokens"] for i in ids]
    wp, z, p2, nu = wilcoxon(d)
    tb = sum(B[i]["n_tokens"] for i in ids)
    te = sum(E[i]["n_tokens"] for i in ids)
    rb = []
    for _ in range(10000):
        s = [rng.choice(ids) for _ in ids]
        sb = sum(B[i]["n_tokens"] for i in s)
        rb.append(sum(E[i]["n_tokens"] for i in s) / sb if sb else float("nan"))
    r2 = {"base_tokens": tb, "e2_tokens": te, "ratio": te / tb if tb else None,
          "ratio_ci": [pct(rb, 0.025), pct(rb, 0.975)], "median_diff": sorted(d)[len(d) // 2],
          "w_plus": wp, "z": z, "p": p2, "n_nonzero": nu}

    h = holm([r1["p"], r2["p"]])
    r1["p_holm"], r2["p_holm"] = h
    r1["pass_e2"] = h[0] < 0.05 and r1["diff"] > 0
    r1["pass_base"] = h[0] < 0.05 and r1["diff"] < 0
    r2["pass_e2"] = h[1] < 0.05 and te < tb
    r2["pass_base"] = h[1] < 0.05 and te > tb

    claims = []
    if r1["pass_e2"]:
        claims.append("solves more cubes")
    if r2["pass_e2"]:
        claims.append("uses fewer tokens on the cube")
    warnings = []
    if r1["pass_base"]:
        warnings.append("the original solves MORE cubes than the edit (R1 in base's favour); report as prominently")
    if r2["pass_base"]:
        warnings.append("the edit uses MORE tokens than the original (R2 in base's favour)")

    # race puzzle by the pre-registered rule
    cand = [i for i in ids if E[i]["correct"]]
    race = None
    if cand:
        diffs = sorted((B[i]["n_tokens"] - E[i]["n_tokens"], i) for i in cand)
        med = diffs[(len(diffs) - 1) // 2][0]
        race = min(i for dd, i in diffs if dd == med)

    mode = "cube" if (cand and r2["pass_e2"]) else "game24"
    out = {"base_file": a.base, "e2_file": a.e2, "n": len(ids), "R1": r1, "R2": r2,
           "base": arm_stats(B, a.base), "e2": arm_stats(E, a.e2), "claims_allowed": claims, "warnings": warnings,
           "race_id": race, "race_candidates": len(cand), "video_mode": mode,
           "per_puzzle": [{"id": i, "depth": B[i]["depth"], "base_correct": bool(B[i]["correct"]),
                           "e2_correct": bool(E[i]["correct"]), "base_tokens": B[i]["n_tokens"],
                           "e2_tokens": E[i]["n_tokens"], "base_status": B[i]["status"],
                           "e2_status": E[i]["status"]} for i in ids]}
    Path(a.out).write_text(json.dumps(out, indent=2))

    pr = print
    pr(f"== Amendment 5, family R (Holm over 2), n {len(ids)} ==")
    pr(f"R1 accuracy: base {acc_b:.1f}% -> e2 {acc_e:.1f}% ({r1['diff']:+.1f} pts, 95% CI [{r1['ci'][0]:+.1f}, "
       f"{r1['ci'][1]:+.1f}]), fixed {fx}, broken {br}, McNemar p {r1['p']:.3g}, Holm p {h[0]:.3g} -> "
       f"{'PASS (e2)' if r1['pass_e2'] else 'PASS (base!)' if r1['pass_base'] else 'FAIL'}")
    pr(f"R2 tokens: base {tb:,} -> e2 {te:,} (ratio {r2['ratio']:.3f}, 95% CI [{r2['ratio_ci'][0]:.3f}, "
       f"{r2['ratio_ci'][1]:.3f}]), Wilcoxon z {z:.2f}, p {p2:.3g}, Holm p {h[1]:.3g} -> "
       f"{'PASS (e2)' if r2['pass_e2'] else 'PASS (base!)' if r2['pass_base'] else 'FAIL'}")
    pr(f"claims allowed: {claims or 'none'}; warnings: {warnings or 'none'}")
    for arm in ("base", "e2"):
        s = out[arm]
        pr(f"{arm}: solved {s['solved']}/{s['n']}, tokens {s['total_tokens']:,}, cap {s['cap']}, finished "
           f"{s['finished']}, detector would fire {s['detector_would_fire']}, wall {s['wall_s']} s, "
           f"tokens/solved {s['tokens_per_solved']}, by depth {s['by_depth']}")
    pr(f"race puzzle: {race} (from {len(cand)} puzzles e2 solves); video mode: {mode}")


if __name__ == "__main__":
    main()
