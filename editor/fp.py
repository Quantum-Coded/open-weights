"""The budget gate: the ONLY way the editor may run the model.

Every call costs 1 forward pass out of the budget in state.json and is logged.

Generate pass (practice problems only, up to 32, cap up to 8,192 tokens):
  python fp.py generate --source math_train --n 32 --seed 7
  python fp.py generate --source gsm8k_train --ids my_ids.txt --max-tokens 8192

Read pass (prefill only, up to 256 texts of up to 1,024 tokens, returns activations):
  python fp.py read --texts texts.json --layers 10,15,19 --module resid --positions last:64
  python fp.py read --texts texts.json --layers 15 --module mlp_act --positions mean --chat

  --module resid    output of decoder layer L (residual stream)
  --module mlp_act  input to layer L's down_proj (the MLP neurons, d=8960)
  --chat            wrap each text as a user message with the chat template
                    (default: tokenize the text as-is, e.g. a saved model answer)

Outputs go to logs/fp/pass_<k>.* ; a summary prints to the terminal.
Both passes run the CURRENT model: base + edits/patch.safetensors.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import PRACTICE_DIR, WORK_DIR, build_prompt_ids, grade, read_jsonl, write_jsonl  # noqa: E402

STATE = HERE / "state.json"
LOGDIR = HERE / "logs" / "fp"
PATCH = HERE / "edits" / "patch.safetensors"
MAX_PROBLEMS, MAX_GEN_TOKENS = 32, 8192
MAX_TEXTS, MAX_TEXT_TOKENS = 256, 1024


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"budget": 50, "used": 0, "passes": []}


def charge(state: dict, kind: str, args: dict) -> int:
    if state["used"] >= state["budget"]:
        raise SystemExit(f"Budget exhausted ({state['used']}/{state['budget']}). No more model runs.")
    state["used"] += 1
    k = state["used"]
    state["passes"].append({"k": k, "kind": kind, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": args})
    STATE.write_text(json.dumps(state, indent=2))
    return k


def current_model_dir() -> Path:
    from apply_patch import build
    out = WORK_DIR / "fp_current"
    info = build(str(PATCH) if PATCH.exists() else None, out)
    print(f"model: base + patch ({info.get('changed_weights', 0):,} weights changed)")
    return out


def cmd_generate(a, state):
    if a.max_tokens > MAX_GEN_TOKENS:
        raise SystemExit(f"--max-tokens is capped at {MAX_GEN_TOKENS}")
    path = PRACTICE_DIR / f"{a.source}.jsonl"
    rows = read_jsonl(path)          # practice sets only: nothing else is loadable here
    if a.ids:
        want = set(Path(a.ids).read_text().split())
        rows = [r for r in rows if r["id"] in want]
    else:
        rows = random.Random(a.seed).sample(rows, a.n)
    if not 0 < len(rows) <= MAX_PROBLEMS:
        raise SystemExit(f"need 1..{MAX_PROBLEMS} problems, got {len(rows)}")

    k = charge(state, "generate", vars(a))
    from gen import free_llm, generate, load_llm
    llm = load_llm(str(current_model_dir()), max_len=a.max_tokens + 1024)
    res = generate(llm, rows, max_tokens=a.max_tokens, temperature=a.temperature,
                   top_p=0.95, early_stop=not a.no_early_stop)
    free_llm(llm)
    by_id = {r["id"]: r for r in rows}
    for r in res:
        r["correct"] = grade(by_id[r["id"]], r["response"])
        r["question"] = by_id[r["id"]]["question"]
    out = LOGDIR / f"pass_{k}.jsonl"
    write_jsonl(out, res)
    n = len(res)
    print(f"pass {k}/{state['budget']}: {n} problems | accuracy {sum(r['correct'] for r in res)}/{n} | "
          f"loop {sum(r['loop'] for r in res)} | cap {sum(r['status'] == 'cap' for r in res)} | "
          f"finished {sum(r['status'] == 'finished' for r in res)}")
    print(f"answers: {out}")


def cmd_read(a, state):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = json.loads(Path(a.texts).read_text())
    if not 0 < len(texts) <= MAX_TEXTS:
        raise SystemExit(f"need 1..{MAX_TEXTS} texts, got {len(texts)}")
    layers = [int(x) for x in a.layers.split(",")]
    k = charge(state, "read", vars(a))

    mdir = current_model_dir()
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.bfloat16).cuda().eval()
    torch.set_grad_enabled(False)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    if a.module == "mlp_act":
        for L in layers:
            mod = model.model.layers[L].mlp.down_proj
            hooks.append(mod.register_forward_hook(
                lambda m, inp, out, L=L: captured.__setitem__(L, inp[0][0].float().cpu())))

    last_k = int(a.positions.split(":")[1]) if a.positions.startswith("last:") else None
    arrays = {L: [] for L in layers}
    token_tails = []
    for t in texts:
        if a.chat:
            ids = build_prompt_ids(tok, {"kind": "math", "question": t})
        else:
            ids = tok(t, add_special_tokens=True)["input_ids"]
        ids = ids[-MAX_TEXT_TOKENS:]
        x = torch.tensor([ids]).cuda()
        out = model(x, output_hidden_states=(a.module == "resid"))
        for L in layers:
            h = out.hidden_states[L + 1][0].float().cpu() if a.module == "resid" else captured[L]
            if last_k:
                h = h[-last_k:]
                if h.shape[0] < last_k:   # pad short texts at the front with NaN
                    h = torch.cat([torch.full((last_k - h.shape[0], h.shape[1]), float("nan")), h])
            else:
                h = h.mean(0, keepdim=True)
            arrays[L].append(h.numpy().astype(np.float16))
        token_tails.append(ids[-last_k:] if last_k else [])
    for hk in hooks:
        hk.remove()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    out = LOGDIR / f"pass_{k}.npz"
    np.savez_compressed(out, **{f"L{L}_{a.module}": np.stack(arrays[L]) for L in layers})
    (LOGDIR / f"pass_{k}_tokens.json").write_text(json.dumps(token_tails))
    shape = np.stack(arrays[layers[0]]).shape
    print(f"pass {k}/{state['budget']}: read {len(texts)} texts, layers {layers}, {a.module}, "
          f"array shape {shape} per layer -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--source", choices=["math_train", "gsm8k_train"], required=True)
    g.add_argument("--n", type=int, default=32)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--ids", default=None, help="file of problem ids, whitespace separated")
    g.add_argument("--max-tokens", type=int, default=MAX_GEN_TOKENS)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--no-early-stop", action="store_true")
    r = sub.add_parser("read")
    r.add_argument("--texts", required=True, help="JSON list of strings")
    r.add_argument("--layers", required=True)
    r.add_argument("--module", choices=["resid", "mlp_act"], default="resid")
    r.add_argument("--positions", default="last:64", help="last:K or mean")
    r.add_argument("--chat", action="store_true")
    a = ap.parse_args()

    state = load_state()
    print(f"budget: {state['used']}/{state['budget']} used")
    (cmd_generate if a.cmd == "generate" else cmd_read)(a, state)


if __name__ == "__main__":
    main()
