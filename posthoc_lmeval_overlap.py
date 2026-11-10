"""POST HOC: is the smaller lm-eval gain about the problems or about the prompt/harness?
Matches lm-eval's first 400 GSM8K test problems to our gsm8k_test400 by question text and compares
both harnesses on the shared problems. Also counts loop/cap answers per harness."""
import json, glob, sys
from math import comb
R = sys.argv[1]

def norm(s):
    return " ".join(s.split())

def lm(arm):
    f = glob.glob(f"{R}/results/lmeval/{arm}/*/samples_*.jsonl")[0]
    return {norm(x["doc"]["question"]): x for x in map(json.loads, open(f, encoding="utf-8"))
            if x["filter"] == "flexible-extract"}

def ours(arm):
    out = {}
    for x in map(json.loads, open(f"{R}/results/{arm}_gsm8k_test400_greedy.jsonl", encoding="utf-8")):
        out[x["id"]] = x
    return out

def mcn(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

ob, oe = ours("base"), ours("edit_r2")
k0 = next(iter(ob.values()))
print("our result keys:", sorted(k0.keys()))
# question text for our rows
qkey = next((k for k in ("question", "problem", "prompt") if k in k0), None)
rows = {}
for f in glob.glob(f"{R}/sealed/*gsm8k*test*") + glob.glob(f"{R}/sealed/**/*gsm8k*test*", recursive=True):
    for l in open(f, encoding="utf-8"):
        try:
            x = json.loads(l)
        except Exception:
            continue
        q = x.get("question") or x.get("problem")
        if q and "id" in x:
            rows[x["id"]] = norm(q)
print("sealed rows found:", len(rows))
lb, le = lm("base"), lm("edit_r2")
shared = [i for i in ob if rows.get(i) in lb]
print("shared problems:", len(shared))

def correct(x):
    return bool(x.get("correct"))

for name, B, E in [("our harness", lambda i: correct(ob[i]), lambda i: correct(oe[i])),
                   ("lm-eval", lambda i: lb[rows[i]]["exact_match"] > 0, lambda i: le[rows[i]]["exact_match"] > 0)]:
    if not shared:
        break
    b = sum(1 for i in shared if E(i) and not B(i)); c = sum(1 for i in shared if B(i) and not E(i))
    print(f"{name} on shared: base {sum(map(B, shared))/len(shared):.4f} edit_r2 {sum(map(E, shared))/len(shared):.4f} "
          f"fixed {b} broken {c} p={mcn(b, c):.3g}")
for a, d in [("base", ob), ("edit_r2", oe)]:
    st = {}
    for x in d.values():
        st[x.get("status")] = st.get(x.get("status"), 0) + 1
    print("our harness status", a, st)
for a, d in [("base", lb), ("edit_r2", le)]:
    print("lm-eval loop/cap-length answers", a, sum(1 for x in d.values() if len(x["resps"][0][0]) > 20000))
