"""Helpers for the editor (Claude). Import from a script or a python session:

    from wsp import load_base, load_current, save_patch, config, note
    W = load_base()                       # dict name -> fp32 tensor (CPU), ~7 GB RAM
    cfg = config()                        # model config (layers, dims ...)
    w = W["model.layers.15.mlp.down_proj.weight"].clone()
    ...                                   # one-shot linear algebra only
    save_patch({"model.layers.15.mlp.down_proj.weight": w})
    note("Removed direction v from 64 columns of L15 down_proj because ...")

Reading and editing weights is free. Running the model costs budget: use fp.py.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from apply_patch import base_dir, load_base_tensors, patch_stats  # noqa: E402

PATCH = HERE / "edits" / "patch.safetensors"
NOTEBOOK = HERE / "logs" / "notebook.md"

torch.set_grad_enabled(False)   # rule 1: no gradients, ever


def config() -> dict:
    return json.loads((base_dir() / "config.json").read_text())


def load_base(dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Base weights, read-only reference. Cast to fp32 for precise linear algebra."""
    return {k: v.to(dtype) for k, v in load_base_tensors(base_dir()).items()}


def load_current(dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Base weights with the current patch applied (what fp.py runs)."""
    W = load_base(dtype)
    if PATCH.exists():
        from safetensors.torch import load_file
        for k, v in load_file(str(PATCH)).items():
            W[k] = v.to(dtype)
    return W


def save_patch(modified: dict[str, torch.Tensor], replace: bool = False) -> dict:
    """Write modified tensors to edits/patch.safetensors.

    Merges with the existing patch unless replace=True. A tensor equal to the
    base is dropped. Values are stored in the base dtype (bf16), so tiny
    changes below bf16 precision vanish: check the reported count.
    """
    from safetensors.torch import load_file, save_file
    base = load_base_tensors(base_dir())
    patch = {} if (replace or not PATCH.exists()) else load_file(str(PATCH))
    for k, v in modified.items():
        if k not in base:
            raise KeyError(f"unknown tensor {k}")
        patch[k] = v.to(base[k].dtype).contiguous()
    patch = {k: v for k, v in patch.items() if not torch.equal(v, base[k])}
    PATCH.parent.mkdir(exist_ok=True)
    if patch:
        save_file(patch, str(PATCH))
    elif PATCH.exists():
        PATCH.unlink()
    s = patch_stats(base, patch)
    print(f"patch: {s['changed_weights']:,} weights changed ({s['changed_pct']:.4f}%) in {len(patch)} tensors")
    return s


def reset_patch() -> None:
    if PATCH.exists():
        PATCH.unlink()
    print("patch cleared: model is back to base")


def note(text: str) -> None:
    """Append a timestamped entry to logs/notebook.md (hypotheses, findings, decisions)."""
    NOTEBOOK.parent.mkdir(exist_ok=True)
    with open(NOTEBOOK, "a", encoding="utf-8") as f:
        f.write(f"\n### {dt.datetime.now():%Y-%m-%d %H:%M}\n{text.strip()}\n")
