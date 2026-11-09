"""Amendment 3 primary: Holm over 2 (e3 vs base, e3 vs rand_e3), accuracy on mmlupro_fresh2_300. Secondaries."""
import json
from math import comb
from collections import defaultdict
def mcn(x, y):
    n = x + y; k = min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
L = lambda f: {r["id"]: r for r in map(json.loads, open(f"results/{f}.jsonl"))}
S = "mmlupro_fresh2_300"
b, e, r, pp = (L(f"{a}_{S}_greedy") for a in ["base", "e3", "rand_e3", "base_pp"])
def fb(a, c, key="correct"):
    return sum(bool(a[i][key]) and not c[i][key] for i in c), sum(bool(c[i][key]) and not a[i][key] for i in c)
t = [("e3 vs base", *fb(e, b)), ("e3 vs rand_e3", *fb(e, r))]
t = [(n, x, y, mcn(x, y)) for n, x, y in t]
ts = sorted(t, key=lambda z: z[3]); run = 0
print("== primary (Holm over 2)")
for k, (n, x, y, p) in enumerate(ts):
    run = max(run, min(1, (2 - k) * p)); print(f"{n:14s} better/worse {x}/{y}  p raw {p:.3f}  p Holm {run:.3f}  {'PASS' if run < .05 else 'fail'}")
print("== secondary")
for name, a in [("base", b), ("e3", e), ("rand_e3", r), ("base_pp", pp)]:
    n = len(a); print(f"{name:8s} acc {sum(x['correct'] for x in a.values())/n:.3f} loop {sum(bool(x['loop']) for x in a.values())/n:.3f} cap {sum(x['status']=='cap' for x in a.values())/n:.3f}")
lr, la = fb(b, e, "loop"); print(f"e3 loops removed/added vs base {lr}/{la} p {mcn(lr, la):.3f}")
x, y = fb(e, pp); print(f"e3 vs base_pp better/worse {x}/{y} p {mcn(x, y):.4f}")
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"sealed/{S}.jsonl")}
cat = defaultdict(lambda: [0, 0, 0])
for i in b:
    c = rows[i].get("category", "?"); cat[c][0] += 1; cat[c][1] += b[i]["correct"]; cat[c][2] += e[i]["correct"]
print("category n base e3:", {k: tuple(v) for k, v in sorted(cat.items())})
