"""Phase 8: hand-check 30 graded answers (15 base, 15 edit, random.Random(0)), MATH-500 fast-track 200."""
import json, random, sys
sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from common import read_jsonl, last_boxed
from pathlib import Path
R = Path("/home/dev_gaglani/wsp-r1/results")
G = {r["id"]: r["answer"] for r in read_jsonl(Path("/home/dev_gaglani/wsp-r1/sealed/math500.jsonl"))}
rng = random.Random(0)
for arm in ("base", "edit"):
    rows = read_jsonl(R / f"{arm}_math500_greedy.jsonl")
    for r in rng.sample(rows, 15):
        resp = r.get("response", "")
        tail = resp.split("</think>")[-1] if "</think>" in resp else resp[-300:]
        print(json.dumps({"arm": arm, "id": r["id"], "gold": G[r["id"]], "boxed": last_boxed(tail) if tail else None,
                          "graded": r["correct"], "status": r["status"], "loop": r["loop"], "n_tok": r["n_tokens"]}, ensure_ascii=False))

