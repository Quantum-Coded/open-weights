"""POST HOC look at the lm-eval GSM8K cross-check (descriptive; the cross-check rule is 'direction and rough size')."""
import json, glob, re, sys
from math import comb
R = sys.argv[1]
BOX = re.compile(r"\\boxed\{([^{}]*)\}")

def load(arm):
    f = glob.glob(f"{R}/results/lmeval/{arm}/*/samples_*.jsonl")[0]
    return {x["doc_id"]: x for x in map(json.loads, open(f, encoding="utf-8")) if x["filter"] == "flexible-extract"}

def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace("$", ""))
    return float(m[-1]) if m else None

def boxed_ok(x):
    m = BOX.findall(x["resps"][0][0])
    return bool(m) and num(m[-1]) == float(x["target"].split("####")[-1].replace(",", "").strip())

def mcn(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

arms = [a for a in ["base", "edit_r2", "edit"] if glob.glob(f"{R}/results/lmeval/{a}/*/samples_*.jsonl")]
D = {a: load(a) for a in arms}
for a in arms:
    d = D[a]
    resp = [d[i]["resps"][0][0] for i in d]
    flex = sum(d[i]["exact_match"] for i in d) / len(d)
    box = sum(boxed_ok(d[i]) for i in d) / len(d)
    hasb = sum(1 for r in resp if BOX.search(r))
    longr = sum(1 for r in resp if len(r) > 20000)
    fw_br = sum(1 for i in d if d[i]["exact_match"] == 0 and boxed_ok(d[i]))
    print(f"{a}: n={len(d)} flexible-extract={flex:.4f} boxed-answer={box:.4f} has_boxed={hasb} "
          f"resp>20k chars (no </think>, i.e. loop/cap)={longr} flex-wrong-but-boxed-right={fw_br}")
for key, name in [(lambda x: x["exact_match"] > 0, "flexible-extract"), (boxed_ok, "boxed-answer")]:
    for a in arms:
        if a == "base":
            continue
        ids = sorted(D["base"])
        b = sum(1 for i in ids if key(D[a][i]) and not key(D["base"][i]))
        c = sum(1 for i in ids if not key(D[a][i]) and key(D["base"][i]))
        print(f"{a} vs base ({name}): fixed {b} broken {c} exact McNemar p={mcn(b, c):.3g}")
