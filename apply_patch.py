"""Build a runnable model folder: base weights + a patch (optionally scaled).

  python apply_patch.py --out work/base                         # plain base model
  python apply_patch.py --patch editor/edits/patch.safetensors --out work/edit
  python apply_patch.py --patch editor/edits/patch.safetensors --scale 0.5 --out work/edit_x0.5
  python apply_patch.py --patch work/random_control.safetensors --out work/random

A patch is a safetensors file holding the FULL new value of every tensor the
editor modified (same names, shapes and dtype as the base). Tensors not in the
patch are copied unchanged. --scale s gives base + s * (patch - base).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from common import BASE_MODEL


def base_dir() -> Path:
    return Path(snapshot_download(BASE_MODEL))


def load_base_tensors(src: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(src.glob("model*.safetensors")):   # only the model's own shards
        tensors.update(load_file(str(f)))
    return tensors


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_stats(base: dict[str, torch.Tensor], patch: dict[str, torch.Tensor]) -> dict:
    total = sum(t.numel() for t in base.values())
    per, changed = {}, 0
    for name, new in patch.items():
        old = base[name]
        if old.shape != new.shape:
            raise ValueError(f"{name}: shape {tuple(new.shape)} != base {tuple(old.shape)}")
        n = int((new.to(old.dtype) != old).sum())
        per[name] = n
        changed += n
    return {"changed_weights": changed, "total_weights": total,
            "changed_pct": 100 * changed / total, "per_tensor": per}


def build(patch_path: str | None, out: Path, scale: float = 1.0) -> dict:
    src = base_dir()
    base = load_base_tensors(src)
    merged = dict(base)
    info: dict = {"base": BASE_MODEL, "patch": patch_path, "scale": scale}
    if patch_path:
        patch = load_file(patch_path)
        for name, new in patch.items():
            old = base[name]
            val = old.float() + scale * (new.float() - old.float())
            merged[name] = val.to(old.dtype).contiguous()
        info["patch_sha256"] = sha256(Path(patch_path))
        info.update(patch_stats(base, {k: merged[k] for k in patch}))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():   # config, tokenizer, generation config ...
        if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json" and f.is_file():
            shutil.copy(f, out / f.name)
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "patch_info.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = build(a.patch, Path(a.out), a.scale)
    if a.patch:
        print(f"changed {info['changed_weights']:,} of {info['total_weights']:,} weights "
              f"({info['changed_pct']:.4f}%)  sha256={info['patch_sha256'][:12]}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
