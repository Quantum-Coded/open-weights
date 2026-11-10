# Run 2 copy of posthoc_mmlupro_lenient.py (same extractor), arms incl. e2 / edit_r2. POST HOC.
"""POST HOC, exploratory: MMLU-Pro rescoring with a lenient letter extractor (boxed first, else explicit answer
phrases in the text after </think>). Not pre-registered; the pre-registered grader stays the primary."""
import json, re, sys
from math import comb
sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from common import grade, answer_part
R = "/home/dev_gaglani/wsp-r1"   # test data restored after run 2
PAT = [r"\boxed\{\s*(?:\text\{)?\s*\(?([A-J])\b", r"[Aa]nswer(?: is)?\s*[:：]?\s*\**\s*\(?([A-J])\b(?![a-z])",
       r"(?:correct|best) (?:answer|option|choice|response) is\s*\**\s*\(?(?:option\s*)?([A-J])\b(?![a-z])",
       r"^\s*\**\(?([A-J])[\.\)]\s", r"[Oo]ption\s*\**\(?([A-J])\b(?![a-z])"]
def lenient(row, resp):
    if "</think>" not in resp: return False
    a = answer_part(resp)
    for p in PAT:
        m = re.findall(p, a, flags=re.M)
        if m: return m[-1] == row["answer"] if p != PAT[0] else m[-1] == row["answer"]
    return False
def mcn(x, y):
    n = x + y; k = min(x, y); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
for S, arms in [("mmlupro_fresh300", ["base", "e1", "e2", "rand_w"]), ("mmlupro300", ["base", "edit", "edit_r2"])]:
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/sealed/{S}.jsonl")}
    sc = {}
    for a in arms:
        res = {r["id"]: r for r in map(json.loads, open(f"{R}/results/{a}_{S}_greedy.jsonl"))}
        strict = {i: grade(rows[i], r["response"]) for i, r in res.items()}
        len_ = {i: strict[i] or lenient(rows[i], r["response"]) for i, r in res.items()}
        fin = sum("</think>" in r["response"] for r in res.values())
        boxed = sum("\boxed" in answer_part(r["response"]) for r in res.values() if "</think>" in r["response"])
        sc[a] = len_
        print(f"{S:20s} {a:8s} strict {sum(strict.values())/len(res):.3f}  lenient {sum(len_.values())/len(res):.3f}  finished {fin}  boxed {boxed}")
    b = sc["base"]
    for a in arms[1:]:
        x = sum(sc[a][i] and not b[i] for i in b); y = sum(b[i] and not sc[a][i] for i in b)
        print(f"   lenient {a} vs base {x}/{y} p {mcn(x, y):.3f}")
