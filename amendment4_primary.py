"""Amendment 4 analysis, exactly as pre-registered in eval_plan_amendment4.md. Missing runs print as 'pending'.

  python amendment4_primary.py
"""
import json, random
from math import comb
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(f):
    p = R / f
    if not p.exists():
        return None
    return {(r["id"], r["sample"]): r for r in map(json.loads, open(p, encoding="utf-8"))}


def mcnemar(f, b):
    n, k = f + b, min(f, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def compare(a_file, b_file, seed=0):
    """b vs a (b is the 'new' arm). Returns dict or None."""
    A, B = load(a_file), load(b_file)
    if A is None or B is None:
        return None
    keys = sorted(set(A) & set(B))
    assert len(keys) == len(A) == len(B), f"{a_file} and {b_file} do not pair up"
    fx = sum(1 for k in keys if not A[k]["correct"] and B[k]["correct"])
    br = sum(1 for k in keys if A[k]["correct"] and not B[k]["correct"])
    # bootstrap over problems (ids), all samples of a problem together
    ids = sorted({k[0] for k in keys})
    by = {i: [k for k in keys if k[0] == i] for i in ids}
    d = {i: sum(B[k]["correct"] - A[k]["correct"] for k in by[i]) for i in ids}
    m = {i: len(by[i]) for i in ids}
    rng = random.Random(seed)
    boots = []
    for _ in range(10000):
        s = [rng.choice(ids) for _ in ids]
        boots.append(100 * sum(d[i] for i in s) / sum(m[i] for i in s))
    boots.sort()
    acc = lambda X: 100 * sum(X[k]["correct"] for k in keys) / len(keys)
    return {"n": len(keys), "a_acc": acc(A), "b_acc": acc(B), "diff": acc(B) - acc(A), "fixed": fx, "broken": br,
            "p": mcnemar(fx, br), "ci": (boots[249], boots[9750])}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def fmt(c):
    return (f"n {c['n']}: {c['a_acc']:.1f}% -> {c['b_acc']:.1f}% ({c['diff']:+.1f} [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]), "
            f"fixed {c['fixed']} broken {c['broken']}, p {c['p']:.3g}")


def family(title, tests):
    print(f"\n== {title} ==")
    done = [(name, compare(a, b)) for name, a, b in tests]
    have = [(n, c) for n, c in done if c is not None]
    if len(have) < len(tests):
        print(f"  ({len(tests) - len(have)} of {len(tests)} pending; Holm needs the whole family, so p(Holm) is provisional)")
    adj = holm([c["p"] for _, c in have]) if have else []
    out = {}
    for (n, c), pa in zip(have, adj):
        print(f"  {n}: {fmt(c)}, p(Holm over the family) {pa:.3g}")
        out[n] = (c, pa)
    for n, c in done:
        if c is None:
            print(f"  {n}: pending")
    return out, len(have) == len(tests)


def summary(f):
    X = load(f)
    if X is None:
        return f"{f}: pending"
    rs = list(X.values())
    n = len(rs)
    return (f"{f}: n {n}, acc {100 * sum(r['correct'] for r in rs) / n:.1f}%, loops {100 * sum(r['loop'] for r in rs) / n:.1f}%, "
            f"cap {100 * sum(r['status'] == 'cap' for r in rs) / n:.1f}%, no </think> {sum('</think>' not in r['response'] for r in rs)}, "
            f"tokens {sum(r['n_tokens'] for r in rs):,}")


def main():
    P, p_complete = family("Family P (primary, Holm over 3): e2 vs base", [
        ("P1 game24_300 greedy", "base_game24_300_greedy.jsonl", "e2_game24_300_greedy.jsonl"),
        ("P2 game24_300 sampled1", "base_game24_300_sampled1.jsonl", "e2_game24_300_sampled1.jsonl"),
        ("P3 math_fresh1200 sampled1", "base_math_fresh1200_sampled1.jsonl", "e2_math_fresh1200_sampled1.jsonl"),
    ])
    for n, (c, pa) in P.items():
        verdict = "PASS" if pa < 0.05 and c["diff"] > 0 else "FAIL" if p_complete else "provisional"
        print(f"  {n}: {verdict}")

    H, h_complete = family("Family H (head-to-head, Holm over 4): e2 vs ft  (positive = edit ahead)", [
        ("H1 MATH-500/200 greedy", "ft_math500_greedy.jsonl", "edit_r2_math500_greedy.jsonl"),
        ("H2 gsm8k_fresh400 greedy", "ft_gsm8k_fresh400_greedy.jsonl", "e2_gsm8k_fresh400_greedy.jsonl"),
        ("H3 mmlupro_fresh300 greedy", "ft_mmlupro_fresh300_greedy.jsonl", "e2_mmlupro_fresh300_greedy.jsonl"),
        ("H4 game24_300 greedy", "ft_game24_300_greedy.jsonl", "e2_game24_300_greedy.jsonl"),
    ])
    words = {}
    for n, (c, pa) in H.items():
        if pa < 0.05:
            w = "edit beats the fine-tune" if c["diff"] > 0 else "fine-tune beats the edit"
        elif c["ci"][0] > -5:
            w = "edit matches the fine-tune"
        else:
            w = "inconclusive"
        words[n] = w
        print(f"  {n}: {w}{'' if h_complete else ' (provisional)'}")
    if h_complete:
        ok = all(words[k] in ("edit beats the fine-tune", "edit matches the fine-tune")
                 for k in ("H1 MATH-500/200 greedy", "H2 gsm8k_fresh400 greedy"))
        print(f"  'can stand in for a fine-tune for this fix': {'ALLOWED' if ok else 'NOT allowed'}")

    print("\n== Secondary: ft vs base ==")
    for name, a, b in [("MATH-500/200 greedy", "base_math500_greedy.jsonl", "ft_math500_greedy.jsonl"),
                       ("gsm8k_fresh400 greedy", "base_gsm8k_fresh400_greedy.jsonl", "ft_gsm8k_fresh400_greedy.jsonl"),
                       ("mmlupro_fresh300 greedy", "base_mmlupro_fresh300_greedy.jsonl", "ft_mmlupro_fresh300_greedy.jsonl"),
                       ("game24_300 greedy", "base_game24_300_greedy.jsonl", "ft_game24_300_greedy.jsonl"),
                       ("MATH-500/200 sampled, ft vs base", "base_math500_sampled.jsonl", "ft_math500_sampled.jsonl"),
                       ("MATH-500/200 sampled, e2 vs ft", "ft_math500_sampled.jsonl", "edit_r2_math500_sampled.jsonl")]:
        c = compare(a, b)
        print(f"  {name}: {fmt(c) if c else 'pending'}")

    print("\n== Secondary: per-run summaries ==")
    for s in ["game24_300_greedy", "game24_300_sampled1", "math_fresh1200_sampled1", "math500_greedy",
              "gsm8k_fresh400_greedy", "mmlupro_fresh300_greedy", "math500_sampled"]:
        for arm in ["base", "e2", "edit_r2", "ft"]:
            if (R / f"{arm}_{s}.jsonl").exists():
                print("  " + summary(f"{arm}_{s}.jsonl"))


if __name__ == "__main__":
    main()
