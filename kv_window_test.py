"""Check gen.generate_windowed (14 Nov, operational) on practice data before kv_window.on is created.

Greedy text is batch-invariant (kv_group_test.py), so with greedy decoding the windowed path must give exactly the
same tokens as llm.generate; this checks that requests are admitted, collected and mapped back correctly,
including when they finish out of order. A sampled case checks that the path runs and returns sane outputs.
Exit code 0 only if every check passes.
"""
import json, sys, time
from pathlib import Path
from common import read_jsonl
from gen import CHUNK, generate_windowed, kv_capacity, load_llm, free_llm, window_size
from kv_group_test import make_gens, params_for, MAX_TOKENS

ROOT = Path(__file__).resolve().parent


def main():
    llm = load_llm("work/base", max_len=MAX_TOKENS + 1024)
    tok = llm.get_tokenizer()
    print("kv capacity tokens:", kv_capacity(llm), flush=True)
    rows = read_jsonl(ROOT / "a4_data" / "ft_problems1200.jsonl")[:20]
    thoughts = [r["response"].split("</think>")[0] for r in read_jsonl(ROOT / "a4_data" / "ft_train.jsonl")]
    ok_all, report = True, []

    def check(name, gens, t, p, k_expected):
        nonlocal ok_all
        params = params_for(gens, t, p)
        prompts = [{"prompt_token_ids": g.prompt_ids + g.out_ids} for g in gens]
        k = window_size(llm, gens, params, force=True)
        t0 = time.time(); w = generate_windowed(llm, prompts, params, k); tw = time.time() - t0
        t0 = time.time(); ref = llm.generate(prompts, params, use_tqdm=False); tr = time.time() - t0
        same = [list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids) for a, b in zip(w, ref)]
        sane = all(o is not None and 0 < len(o.outputs[0].token_ids) <= pr.max_tokens for o, pr in zip(w, params))
        r = {"case": name, "n": len(gens), "k": k, "k_expected": k_expected, "wall_window_s": round(tw),
             "wall_plain_s": round(tr), "identical": same, "sane": sane,
             "lens": [len(o.outputs[0].token_ids) for o in w], "finish": [o.outputs[0].finish_reason for o in w]}
        print(json.dumps(r), flush=True)
        report.append(r)
        good = sane and k == k_expected and (t > 0 or all(same))
        ok_all &= good

    # 1) last round of a 16k run, greedy: window 1
    check("round8_greedy", make_gens(tok, rows, thoughts, 3, 7 * CHUNK), 0.0, 1.0, 1)
    # 2) round 5, greedy, mixed lengths so some finish early and out of order: window 2
    gens = make_gens(tok, rows, thoughts, 6, 4 * CHUNK)
    for i, g in enumerate(gens):
        g.out_ids = g.out_ids[: 4 * CHUNK - 600 * i]
    check("round5_greedy_mixed", gens, 0.0, 1.0, 2)
    # 3) round 4, sampled: window 3
    check("round4_sampled", make_gens(tok, rows, thoughts, 5, 3 * CHUNK), 0.6, 0.95, 3)

    free_llm(llm)
    (ROOT / "results" / "kv_window_test.json").write_text(json.dumps(report, indent=2))
    print("WINDOW OK" if ok_all else "WINDOW CHECK FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
