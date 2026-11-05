"""Amendment 2: fresh, untouched test sets (operator, 2026-11-05, before any model runs on them).
gsm8k_fresh400: 400 GSM8K test problems NOT in gsm8k_test400. mmlupro_fresh300: 300 MMLU-Pro test questions NOT in mmlupro300.
Procedure (fixed before building): shuffle the remaining indices with random.Random(20261105) (one rng, GSM8K first),
walk in that order, skip any question that matches a practice problem by the Phase 5 decontamination rule
(same first 40 normalized chars AND (difflib ratio >= 0.80 OR same first 60 chars)), keep the first 400 / 300; sort by index."""
import json, random, re, difflib, hashlib, sys
sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from datasets import load_dataset
from common import write_jsonl, read_jsonl, SEALED_DIR, PRACTICE_DIR
def norm(s): return re.sub(r"\s+", " ", s).strip()
pp = {}
for f in ("math_train", "gsm8k_train"):
    for r in read_jsonl(PRACTICE_DIR / f"{f}.jsonl"):
        q = norm(r["question"]); pp.setdefault(q[:40], []).append(q)
def contaminated(q):
    q = norm(q)
    return any(x[:60] == q[:60] or difflib.SequenceMatcher(None, q, x).ratio() >= 0.80 for x in pp.get(q[:40], []))
rng = random.Random(20261105)
used_g = {int(r["id"].split("-")[1]) for r in read_jsonl(SEALED_DIR / "gsm8k_test400.jsonl")}
used_m = {int(r["id"].split("-")[1]) for r in read_jsonl(SEALED_DIR / "mmlupro300.jsonl")}
gsm = list(load_dataset("openai/gsm8k", "main", split="test"))
rest = [i for i in range(len(gsm)) if i not in used_g]; rng.shuffle(rest)
keep, skip = [], 0
for i in rest:
    if len(keep) == 400: break
    if contaminated(gsm[i]["question"]): skip += 1; continue
    keep.append(i)
g = [{"id": f"gsm-{i}", "kind": "gsm8k", "question": gsm[i]["question"], "answer": gsm[i]["answer"].split("####")[-1].strip()} for i in sorted(keep)]
print(f"gsm8k test {len(gsm)}, old sealed {len(used_g)}, remaining {len(rest)}, skipped {skip} contaminated, kept {len(g)}")
mp = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
rest = [i for i in range(len(mp)) if i not in used_m]; rng.shuffle(rest)
keep, skip = [], 0
for i in rest:
    if len(keep) == 300: break
    if contaminated(mp[i]["question"]): skip += 1; continue
    keep.append(i)
m = [{"id": f"mmlup-{i}", "kind": "mcq", "question": mp[i]["question"], "options": mp[i]["options"], "answer": mp[i]["answer"], "category": mp[i]["category"]} for i in sorted(keep)]
print(f"mmlu-pro test {len(mp)}, old sealed {len(used_m)}, skipped {skip} contaminated, kept {len(m)}")
assert not ({r["id"] for r in g} & {f"gsm-{i}" for i in used_g}) and not ({r["id"] for r in m} & {f"mmlup-{i}" for i in used_m})
write_jsonl(SEALED_DIR / "gsm8k_fresh400.jsonl", g)
write_jsonl(SEALED_DIR / "mmlupro_fresh300.jsonl", m)
from collections import Counter
print("mmlu fresh categories:", dict(Counter(r["category"] for r in m)))
for f in ("gsm8k_fresh400", "mmlupro_fresh300"):
    print(f, "sha256", hashlib.sha256((SEALED_DIR / f"{f}.jsonl").read_bytes()).hexdigest())
