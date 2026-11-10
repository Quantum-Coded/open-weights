"""Race-demo recording (POST HOC, descriptive; RUNLOG 10-11 Nov).

Streams one model on chosen problems, ONE request at a time (single stream, batch size 1), greedy, no loop
kill, up to the 16,384-token cap, and records a timestamp for every engine step. The video replays the arms
side by side on a shared clock; each arm is recorded separately on this laptop, which the video discloses.

The text is checked against the arm's batched no-loop-kill eval run (batch-invariant kernels should make it
identical); the result says whether it matched.

usage: python record_race.py --model work/edit_r2 --arm e2 --ref results/e2_nokill_gsm8k_fresh400_greedy.jsonl \
           --ids gsm-665,gsm-794 --out results/race
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

from common import build_prompt_ids  # noqa: E402
from gen import load_llm, free_llm  # noqa: E402

MAX_TOKENS = 16384


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--set", default="sealed/gsm8k_fresh400.jsonl")
    ap.add_argument("--out", default="results/race")
    a = ap.parse_args()

    from vllm import SamplingParams
    rows = {r["id"]: r for r in map(json.loads, open(a.set, encoding="utf-8"))}
    ref = {r["id"]: r for r in map(json.loads, open(a.ref, encoding="utf-8"))}
    os.makedirs(a.out, exist_ok=True)

    llm = load_llm(a.model, 1024 + MAX_TOKENS)
    tok = llm.get_tokenizer()
    eng = llm.llm_engine
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, skip_special_tokens=False)

    # Warm-up on an unrelated short prompt so start-up costs do not land in the timed run.
    warm = tok("Count from 1 to 50.", add_special_tokens=False)["input_ids"]
    eng.add_request("warmup", {"prompt_token_ids": warm}, SamplingParams(max_tokens=256, temperature=0.0))
    while eng.has_unfinished_requests():
        eng.step()

    for pid in a.ids.split(","):
        out_path = f"{a.out}/{a.arm}_{pid}.json"
        if os.path.exists(out_path):
            print("SKIP", out_path)
            continue
        prompt = build_prompt_ids(tok, rows[pid])
        steps, ids, finish = [], [], None
        t0 = time.perf_counter()
        eng.add_request(f"race-{pid}", {"prompt_token_ids": prompt}, params)
        while eng.has_unfinished_requests():
            for o in eng.step():
                c = o.outputs[0]
                ids = list(c.token_ids)
                steps.append([len(ids), round(time.perf_counter() - t0, 4)])
                if o.finished:
                    finish = c.finish_reason
        text = tok.decode(ids, skip_special_tokens=True)
        r = ref.get(pid, {})
        rec = {
            "arm": a.arm, "model": a.model, "id": pid, "question": rows[pid]["question"],
            "answer": rows[pid].get("answer"), "prompt_tokens": len(prompt), "n_tokens": len(ids),
            "finish_reason": finish, "wall_s": steps[-1][1] if steps else 0.0,
            "tok_per_s": len(ids) / steps[-1][1] if steps and steps[-1][1] else None,
            "ref_status": r.get("status"), "ref_correct": r.get("correct"), "ref_n_tokens": r.get("n_tokens"),
            "matches_eval_text": text == r.get("response"),
            "steps": steps, "token_ids": ids, "text": text,
        }
        json.dump(rec, open(out_path, "w", encoding="utf-8"))
        print(f"{a.arm} {pid}: {len(ids)} tokens in {rec['wall_s']:.1f}s ({rec['tok_per_s']:.1f} tok/s), "
              f"finish={finish}, matches eval text={rec['matches_eval_text']} (ref {r.get('status')}, "
              f"correct={r.get('correct')}, {r.get('n_tokens')} tokens)", flush=True)
    free_llm(llm)


if __name__ == "__main__":
    main()
