"""Pre-registered example rule (eval_plan.md): for each MATH subject, the lowest-index problem the edit fixed. Fast-track 200."""
import sys; sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from common import read_jsonl
from pathlib import Path
H = Path("/home/dev_gaglani/wsp-r1")
S = {r["id"]: r for r in read_jsonl(H / "sealed/math500.jsonl")}
b = {r["id"]: r for r in read_jsonl(H / "results/base_math500_greedy.jsonl")}
e = {r["id"]: r for r in read_jsonl(H / "results/edit_math500_greedy.jsonl")}
fixed = sorted((i for i in b if not b[i]["correct"] and e[i]["correct"]), key=lambda i: int(i.split("-")[1]))
pick = {}
for i in fixed: pick.setdefault(S[i]["subject"], i)
for subj, i in sorted(pick.items(), key=lambda kv: int(kv[1].split("-")[1])):
    print(f"{i:9} {subj:24} L{S[i]['level']} base={b[i]['status']}/{b[i]['n_tokens']} first_flag={b[i]['first_flag']} edit={e[i]['status']}/{e[i]['n_tokens']}")
print("fixed total", len(fixed))
