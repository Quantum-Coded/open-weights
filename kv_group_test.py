"""Check that KV-grouped generation (gen._groups) gives exactly the same tokens as one llm.generate call.

Operational check for Amendment 4 (13 Nov), run by ops/kv_verify.sh before the flag file kv_group.on is
created. Uses practice data only: contexts are the base model's own practice reasoning
(a4_data/ft_train.jsonl, thinking part only), packed to the lengths of the last chunk rounds of a
16k-token run. For each case the same requests are generated
  OLD:  one llm.generate call over all of them (the path used so far: vLLM queues and preempts),
  NEW:  gen._groups split, forced on,
  NEW2: gen._groups split on a reordered list (different batch neighbours),
and the token ids of every request must be identical across the three. Exit code 0 only if all match.
"""
import json, sys, time
from pathlib import Path
from common import build_prompt_ids, read_jsonl, text_hash
from gen import CHUNK, Gen, _groups, free_llm, kv_capacity, load_llm

ROOT = Path(__file__).resolve().parent
MAX_TOKENS = 16384


def make_gens(tok, rows, thoughts, n, ctx_out):
    """n requests whose generated-so-far part is ctx_out tokens of varied practice reasoning."""
    gens = []
    for i in range(n):
        row = rows[i]
        ids = []
        j = i * 7
        while len(ids) < ctx_out:
            ids += tok(thoughts[j % len(thoughts)] + "\n\nHmm, let me look at this from another angle.\n\n",
                       add_special_tokens=False)["input_ids"]
            j += 1
        gens.append(Gen(row=row, sample=0, prompt_ids=build_prompt_ids(tok, row), out_ids=ids[:ctx_out]))
    return gens


def params_for(gens, temperature, top_p):
    from vllm import SamplingParams
    out = []
    for g in gens:
        s = int(text_hash(f"{g.row['id']}|{g.sample}"), 16) % 1_000_003 + len(g.out_ids)
        out.append(SamplingParams(max_tokens=min(CHUNK, MAX_TOKENS - len(g.out_ids)), temperature=temperature,
                                  top_p=top_p if temperature > 0 else 1.0,
                                  seed=None if temperature == 0 else s, skip_special_tokens=False))
    return out


def run(llm, gens, params, force, order):
    gs = [gens[i] for i in order]
    ps = [params[i] for i in order]
    prompts = [{"prompt_token_ids": g.prompt_ids + g.out_ids} for g in gs]
    groups = _groups(llm, gs, ps, force=force)
    t0 = time.time()
    outs = []
    for lo, hi in groups:
        outs.extend(llm.generate(prompts[lo:hi], ps[lo:hi], use_tqdm=False))
    wall = time.time() - t0
    toks = {}
    for i, o in zip(order, outs):
        toks[i] = (list(o.outputs[0].token_ids), o.outputs[0].finish_reason)
    return toks, [hi - lo for lo, hi in groups], wall


def main():
    llm = load_llm("work/base", max_len=MAX_TOKENS + 1024)
    tok = llm.get_tokenizer()
    cap = kv_capacity(llm)
    print("kv capacity tokens:", cap, flush=True)
    if not cap:
        print("FAIL: KV capacity unknown, grouping would never switch on")
        return 1
    rows = read_jsonl(ROOT / "a4_data" / "ft_problems1200.jsonl")[:20]
    thoughts = [r["response"].split("</think>")[0] for r in read_jsonl(ROOT / "a4_data" / "ft_train.jsonl")]
    cases = [  # (name, n, generated-so-far tokens = round, temperature, top_p)
        ("round8_sampled", 3, 7 * CHUNK, 0.6, 0.95),   # last round of a 16k run: one answer fits
        ("round6_sampled", 6, 5 * CHUNK, 0.6, 0.95),   # two fit
        ("round7_greedy", 3, 6 * CHUNK, 0.0, 1.0),
    ]
    report, ok_all = [], True
    for name, n, ctx, t, p in cases:
        gens = make_gens(tok, rows, thoughts, n, ctx)
        params = params_for(gens, t, p)
        fwd = list(range(n))
        alt = fwd[1::2] + fwd[0::2]
        old, g_old, w_old = run(llm, gens, params, False, fwd)
        new, g_new, w_new = run(llm, gens, params, True, fwd)
        new2, g_new2, w_new2 = run(llm, gens, params, True, alt)
        same = [old[i] == new[i] == new2[i] for i in fwd]
        ntok = sum(len(old[i][0]) for i in fwd)
        first_diff = []
        for i in fwd:
            a, b = old[i][0], new[i][0]
            d = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None if len(a) == len(b) else min(len(a), len(b)))
            first_diff.append(d)
        r = {"case": name, "n": n, "ctx_out": ctx, "groups_old": g_old, "groups_new": g_new, "groups_new2": g_new2,
             "wall_old_s": round(w_old), "wall_new_s": round(w_new), "wall_new2_s": round(w_new2),
             "tokens_compared": ntok, "identical": all(same), "per_request": same, "first_diff": first_diff,
             "finish": [old[i][1] for i in fwd]}
        print(json.dumps(r), flush=True)
        report.append(r)
        ok_all &= all(same) and len(g_new) > 1
    free_llm(llm)
    (ROOT / "results" / "kv_group_test.json").write_text(json.dumps(report, indent=2))
    print("ALL IDENTICAL" if ok_all else "MISMATCH OR NO SPLIT")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
