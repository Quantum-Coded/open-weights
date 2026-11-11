"""Amendment 4, fine-tune baseline step 1: training data by rejection sampling (practice problems only).

Base model, a4_data/ft_problems1200.jsonl, T 0.6, top-p 0.95, 1 sample, 3,072-token cap, loop kill on.
Keep answers that finished (EOS) and are graded correct. Writes a4_data/ft_train.jsonl.
The 3,072 cap is a memory limit of the 6 GB GPU for training (sequence <= prompt + 3,072), fixed in advance.
"""
import json, time
from pathlib import Path
from common import grade, read_jsonl, write_jsonl
from gen import free_llm, generate, load_llm

A4 = Path(__file__).resolve().parent / "a4_data"
rows = read_jsonl(A4 / "ft_problems1200.jsonl")
by_id = {r["id"]: r for r in rows}
t0 = time.time()
llm = load_llm("work/base", max_len=3072 + 1024)
res = generate(llm, rows, max_tokens=3072, temperature=0.6, top_p=0.95, n_samples=1, early_stop=True)
free_llm(llm)
keep = []
for r in res:
    row = by_id[r["id"]]
    ok = r["status"] == "finished" and grade(row, r["response"])
    if ok:
        keep.append({**row, "response": r["response"], "n_tokens": r["n_tokens"]})
write_jsonl(A4 / "ft_train.jsonl", keep)
stat = {s: sum(r["status"] == s for r in res) for s in ("finished", "loop", "cap")}
print(json.dumps({"problems": len(rows), "status": stat, "kept": len(keep),
                  "kept_math": sum(r["kind"] == "math" for r in keep),
                  "kept_gsm8k": sum(r["kind"] == "gsm8k" for r in keep),
                  "kept_tokens": sum(r["n_tokens"] for r in keep), "wall_s": round(time.time() - t0)}, indent=2))
