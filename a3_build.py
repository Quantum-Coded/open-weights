"""Amendment 3: re-apply editing run 1's recipe (editor/work/steer.py + make_h4.py) to MMLU-Pro loops.

  python a3_build.py --results results/a3prac_a3_practice_mmlupro128_greedy.jsonl \
      --rows sealed/a3_practice_mmlupro128.jsonl --out work/a3

Recipe (unchanged from run 1; only the practice transcripts differ):
  1. Read the layer-14 residual stream (output of decoder layer 14) of the base model over 1,000-token windows of
     its own greedy practice transcripts. Windows start at response offsets 0, 1000, 2000, ...; the window at
     offset 0 is read with the prompt in front, later windows on their own (BOS + window), as in run 1. The first
     300 positions of every later window (100 of the first) are skipped as warm-up.
  2. d = mean(h | failed runs, response position >= 2000) - mean(h | healthy class).
     Healthy class, in this fixed order of preference (first with >= 2,000 positions is used):
       A. correct runs, response position >= 1000, before </think>   (run 1's definition)
       B. finished runs (any grade), position >= 1000, before </think>
       C. finished runs (any grade), position >= 300, before </think>
  3. Target -0.5 * d, realized as a constant attention output through v_proj.bias of layers 10-14 with
     one ridge least-squares solve, lambda = 1e-2 * trace(A^T A) / n (make_h4.py).
No gradients, no evaluation inside this script, no tuning. Writes patch_e3.safetensors and a3_build.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from apply_patch import base_dir, load_base_tensors
from common import build_prompt_ids, read_jsonl

THINK_END = 151649          # </think>
W, L_READ, ALPHA, LAYERS = 1000, 14, 0.5, [10, 11, 12, 13, 14]
MIN_POS = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = {r["id"]: r for r in read_jsonl(Path(a.rows))}
    res = read_jsonl(Path(a.results))
    src = base_dir()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16).cuda().eval()
    torch.set_grad_enabled(False)
    grab = {}
    model.model.layers[L_READ].register_forward_hook(lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))

    D = 1536
    acc = {k: [torch.zeros(D, dtype=torch.float64, device="cuda"), 0] for k in ("loop", "A", "B", "C")}
    def add(k, h):
        if h.shape[0]:
            acc[k][0] += h.double().sum(0); acc[k][1] += h.shape[0]
    n_win = 0
    for r in res:
        p = build_prompt_ids(tok, rows[r["id"]])
        resp = tok(r["response"], add_special_tokens=False)["input_ids"]
        end = resp.index(THINK_END) if THINK_END in resp else len(resp)
        finished = r["status"] == "finished"
        for s in range(0, len(resp), W):
            chunk = resp[s:s + W]
            if s == 0:
                ids, off, lo = p + chunk, len(p), max(0, 100 - len(p))
            else:
                ids, off, lo = [tok.bos_token_id] + chunk, 1, 300
            model(torch.tensor([ids], device="cuda")); n_win += 1
            h = grab["h"][off:]                                  # one row per response token in the window
            gp = s + torch.arange(h.shape[0], device="cuda")
            keep = torch.arange(h.shape[0], device="cuda") >= lo
            pre = gp < end
            if not r["correct"]:
                add("loop", h[keep & (gp >= 2000)])
            if r["correct"]:
                add("A", h[keep & pre & (gp >= 1000)])
            if finished:
                add("B", h[keep & pre & (gp >= 1000)])
                add("C", h[keep & pre & (gp >= 300)])
    counts = {k: v[1] for k, v in acc.items()}
    healthy = next((k for k in ("A", "B", "C") if counts[k] >= MIN_POS), None)
    if healthy is None or counts["loop"] < MIN_POS:
        raise SystemExit(f"not enough positions: {counts}")
    mean = lambda k: (acc[k][0] / acc[k][1]).cpu().numpy()
    d = mean("loop") - mean(healthy)
    del model; torch.cuda.empty_cache()

    # make_h4.py, unchanged
    Wt = load_base_tensors(src)
    E = np.zeros((1536, 256), dtype=np.float32)
    for hd in range(12):
        g = hd // 6
        E[hd * 128:(hd + 1) * 128, g * 128:(g + 1) * 128] = np.eye(128)
    A = np.concatenate([Wt[f"model.layers.{l}.self_attn.o_proj.weight"].float().numpy() @ E for l in LAYERS], 1)
    target = -ALPHA * d
    lam = 1e-2 * np.trace(A.T @ A) / A.shape[1]
    x = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ target)
    realized = 1 - np.linalg.norm(A @ x - target) ** 2 / np.linalg.norm(target) ** 2
    patch = {}
    for k, l in enumerate(LAYERS):
        name = f"model.layers.{l}.self_attn.v_proj.bias"
        b = Wt[name]
        patch[name] = (b.float() + torch.tensor(x[k * 256:(k + 1) * 256], dtype=torch.float32)).to(b.dtype).contiguous()
    save_file(patch, str(out / "patch_e3.safetensors"))
    np.save(out / "d_e3_L14.npy", d)
    d1 = np.load("editor/work/d_loop_L14.npy")
    info = {"windows_read": n_win, "positions": counts, "healthy_class": healthy, "norm_d": float(np.linalg.norm(d)),
            "cos_with_run1_direction": float(d @ d1 / np.linalg.norm(d) / np.linalg.norm(d1)),
            "realized_fraction": float(realized), "norm_target": float(np.linalg.norm(target)),
            "transcripts": len(res), "failed": sum(not r["correct"] for r in res),
            "finished": sum(r["status"] == "finished" for r in res)}
    (out / "a3_build.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
