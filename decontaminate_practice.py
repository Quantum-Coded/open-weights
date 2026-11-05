"""Pre-handoff decontamination (operator, before any edit exists).
Rule, fixed before applying: drop every practice problem that shares its first 40 normalized characters with a
sealed test question AND (has difflib ratio >= 0.80 with it OR shares its first 60 normalized characters). Originals kept in work/practice_orig/. Run once, right after prepare_data.py."""
import json, re, difflib, shutil, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
S = ROOT / "sealed"
P = ROOT / "editor" / "data" / "practice"
O = ROOT / "work" / "practice_orig"
def norm(s): return re.sub(r"\s+", " ", s).strip()
O.mkdir(parents=True, exist_ok=True)
for f in ("math_train.jsonl", "gsm8k_train.jsonl"):
    if not (O / f).exists():
        shutil.copy2(P / f, O / f)
tests = []
for s in ("math500", "gsm8k_test400", "mmlupro300"):
    for l in open(S / f"{s}.jsonl"):
        r = json.loads(l); tests.append((s, r["id"], norm(r["question"])))
by_pref = {}
for s, i, q in tests:
    by_pref.setdefault(q[:40], []).append((s, i, q))
removed = []
for f in ("math_train.jsonl", "gsm8k_train.jsonl"):
    keep = []
    lines = [l for l in (O / f).read_text(encoding="utf-8").split("\n") if l.strip()]
    for l in lines:
        r = json.loads(l); q = norm(r["question"])
        m = [(round(difflib.SequenceMatcher(None, q, tq).ratio(), 3), s, i) for s, i, tq in by_pref.get(q[:40], [])]
        m = [x for x, (s, i, tq) in zip(m, by_pref.get(q[:40], [])) if x[0] >= 0.80 or tq[:60] == q[:60]]
        if m:
            removed.append({"practice_id": r["id"], "file": f, "matches": m})
        else:
            keep.append(l)
    (P / f).write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f, len(lines), "->", len(keep))
(ROOT / "work" / "practice_removed.json").write_text(json.dumps(removed, indent=1))
print("removed", len(removed), "practice problems; tests touched:", len({(m[1], m[2]) for r in removed for m in r["matches"]}))
for f in ("math_train.jsonl", "gsm8k_train.jsonl"):
    print(f, "sha256", hashlib.sha256((P / f).read_bytes()).hexdigest()[:16])
