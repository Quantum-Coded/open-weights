"""Paired before/after statistics.

  python analyze.py --base results/base_math500_greedy.jsonl \
      --arm results/edit_math500_greedy.jsonl \
      --arm results/random_math500_greedy.jsonl \
      --arm results/base_pp_math500_greedy.jsonl

For every arm vs base, on the same problems:
  accuracy and loop rate, change with a 95% bootstrap CI (10,000 resamples of problems),
  problems fixed (wrong -> right) and broken (right -> wrong),
  exact McNemar p-value (greedy) or sign-flip permutation p-value (sampled),
  Holm-adjusted p-values across all arms.
Prints a markdown table you can paste into the write-up.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

from common import read_jsonl

B = 10_000


def per_problem(path: str) -> dict[str, dict]:
    acc, loop = defaultdict(list), defaultdict(list)
    for r in read_jsonl(Path(path)):
        acc[r["id"]].append(float(r["correct"]))
        loop[r["id"]].append(float(r["loop"]))
    return {k: {"acc": np.mean(acc[k]), "loop": np.mean(loop[k])} for k in acc}


def boot_ci(d: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def signflip_p(d: np.ndarray, rng) -> float:
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(B, len(d)))
    return float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    base = per_problem(a.base)
    rows, ps = [], []
    for arm_path in a.arm:
        arm = per_problem(arm_path)
        ids = sorted(set(base) & set(arm))
        if len(ids) < len(base):
            print(f"warning: {arm_path} covers {len(ids)}/{len(base)} base problems")
        b_acc = np.array([base[i]["acc"] for i in ids])
        a_acc = np.array([arm[i]["acc"] for i in ids])
        b_loop = np.array([base[i]["loop"] for i in ids])
        a_loop = np.array([arm[i]["loop"] for i in ids])
        d_acc, d_loop = a_acc - b_acc, a_loop - b_loop
        binary = set(np.unique(np.concatenate([b_acc, a_acc]))) <= {0.0, 1.0}
        fixed = int(((b_acc == 0) & (a_acc == 1)).sum()) if binary else int((d_acc > 0).sum())
        broken = int(((b_acc == 1) & (a_acc == 0)).sum()) if binary else int((d_acc < 0).sum())
        p = mcnemar_exact(fixed, broken) if binary else signflip_p(d_acc, rng)
        ps.append(p)
        rows.append(dict(arm=Path(arm_path).stem, n=len(ids),
                         base_acc=b_acc.mean(), arm_acc=a_acc.mean(), d_acc=d_acc.mean(),
                         ci_acc=boot_ci(d_acc, rng), base_loop=b_loop.mean(), arm_loop=a_loop.mean(),
                         d_loop=d_loop.mean(), ci_loop=boot_ci(d_loop, rng),
                         fixed=fixed, broken=broken, p=p, test="McNemar" if binary else "sign-flip"))
    for r, padj in zip(rows, holm(ps)):
        r["p_holm"] = padj

    pct = lambda x: f"{100 * x:.1f}%"
    pts = lambda x: f"{100 * x:+.1f}"
    print(f"\nBase: {Path(a.base).stem}\n")
    print("| Arm | n | Accuracy (base -> arm) | Change [95% CI] | Loop rate (base -> arm) | "
          "Change [95% CI] | Fixed / broken | p (Holm) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['arm']} | {r['n']} | {pct(r['base_acc'])} -> {pct(r['arm_acc'])} | "
              f"{pts(r['d_acc'])} [{pts(r['ci_acc'][0])}, {pts(r['ci_acc'][1])}] | "
              f"{pct(r['base_loop'])} -> {pct(r['arm_loop'])} | "
              f"{pts(r['d_loop'])} [{pts(r['ci_loop'][0])}, {pts(r['ci_loop'][1])}] | "
              f"{r['fixed']} / {r['broken']} | {r['p_holm']:.2g} ({r['test']}) |")


if __name__ == "__main__":
    main()
