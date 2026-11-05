"""Amendment 3 data (operator, 2026-11-06, before any model runs on it).
mmlupro_fresh2_300: 300 new MMLU-Pro test questions (sealed test set).
a3_practice_mmlupro128: 128 further MMLU-Pro questions used only to estimate the loop direction (practice).
Procedure: indices not in mmlupro300 / mmlupro_fresh300, shuffled with random.Random(20261106); walk in order;
skip a question that matches (Phase 5 rule) any editor practice problem, any sealed MMLU-Pro question, or any question
already taken for the other new set; first 300 kept -> test, next 128 -> practice. Both sorted by index."""
import json, random, re, difflib, hashlib, sys
sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from datasets import load_dataset
from common import write_jsonl, read_jsonl, SEALED_DIR, PRACTICE_DIR
def norm(s): return re.sub(r"\s+", " ", s).strip()
idx = {}
def add(q): q = norm(q); idx.setdefault(q[:40], []).append(q)
def hit(q):
    q = norm(q)
    return any(x[:60] == q[:60] or difflib.SequenceMatcher(None, q, x).ratio() >= 0.80 for x in idx.get(q[:40], []))
for f in ("math_train", "gsm8k_train"):
    for r in read_jsonl(PRACTICE_DIR / f"{f}.jsonl"): add(r["question"])
used = set()
for s in ("mmlupro300", "mmlupro_fresh300"):
    for r in read_jsonl(SEALED_DIR / f"{s}.jsonl"): add(r["question"]); used.add(int(r["id"].split("-")[1]))
mp = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
rest = [i for i in range(len(mp)) if i not in used]
random.Random(20261106).shuffle(rest)
test, prac, skip = [], [], 0
for i in rest:
    if len(prac) == 128: break
    q = mp[i]["question"]
    if hit(q): skip += 1; continue
    (test if len(test) < 300 else prac).append(i); add(q)
row = lambda i: {"id": f"mmlup-{i}", "kind": "mcq", "question": mp[i]["question"], "options": mp[i]["options"], "answer": mp[i]["answer"], "category": mp[i]["category"]}
write_jsonl(SEALED_DIR / "mmlupro_fresh2_300.jsonl", [row(i) for i in sorted(test)])
write_jsonl(SEALED_DIR / "a3_practice_mmlupro128.jsonl", [row(i) for i in sorted(prac)])
assert not (set(test) & set(prac)) and not ((set(test) | set(prac)) & used)
print(f"skipped {skip} near-duplicates; test {len(test)}, practice {len(prac)}")
from collections import Counter
print("test categories", dict(Counter(mp[i]["category"] for i in test)))
for f in ("mmlupro_fresh2_300", "a3_practice_mmlupro128"):
    print(f, "sha256", hashlib.sha256((SEALED_DIR / f"{f}.jsonl").read_bytes()).hexdigest())
