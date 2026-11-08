"""Amendment 2 primary family: e1, e1x1.5 (and e2 once run 2 exists) vs base, accuracy, per fresh set. Holm."""
import json
from math import comb
def mcn(x, y):
    n = x + y; k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
L = lambda f: {r["id"]: r for r in map(json.loads, open(f"results/{f}.jsonl"))}
tests = []
for S in ["gsm8k_fresh400", "mmlupro_fresh300"]:
    b = L(f"base_{S}_greedy"); rw = L(f"rand_w_{S}_greedy")
    for arm in ["e1", "e1x1.5"]:
        a = L(f"{arm}_{S}_greedy")
        x = sum(a[i]["correct"] and not b[i]["correct"] for i in b); y = sum(b[i]["correct"] and not a[i]["correct"] for i in b)
        tests.append((f"{arm} vs base, {S}", x, y, mcn(x, y)))
    a = L(f"e1_{S}_greedy")
    x = sum(a[i]["correct"] and not rw[i]["correct"] for i in rw); y = sum(rw[i]["correct"] and not a[i]["correct"] for i in rw)
    lx = sum(bool(b[i]["loop"]) and not a[i]["loop"] for i in b); ly = sum(bool(a[i]["loop"]) and not b[i]["loop"] for i in b)
    rlx = sum(bool(b[i]["loop"]) and not rw[i]["loop"] for i in b); rly = sum(bool(rw[i]["loop"]) and not b[i]["loop"] for i in b)
    print(f"[secondary] {S}: e1 vs rand_w {x}/{y} p={mcn(x,y):.3g}; loops removed/added e1 {lx}/{ly} p={mcn(lx,ly):.3g}, rand_w {rlx}/{rly} p={mcn(rlx,rly):.3g}")
def holm(ts, m):
    ts = sorted(ts, key=lambda t: t[3]); out = []; run = 0
    for k, t in enumerate(ts):
        run = max(run, min(1.0, (m - k) * t[3])); out.append((*t, run))
    return out
for m, label in [(4, "provisional, Holm over the 4 tests that exist"), (6, "conservative, Holm with m=6 as if both e2 tests had p=1")]:
    print(f"\n== {label}")
    for name, x, y, p, ph in holm(tests, m):
        print(f"{name:32s} fixed/broken {x:3d}/{y:3d}  p raw {p:.4f}  p Holm {ph:.4f}  {'PASS' if ph < 0.05 else 'fail'}")
