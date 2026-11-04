"""Random-edit controls matched to Claude's patch.

  python make_random_control.py --patch editor/edits/patch.safetensors --mode same-weights --out work/random_same_weights.safetensors
  python make_random_control.py --patch editor/edits/patch.safetensors --mode same-size    --out work/random_same_size.safetensors

same-weights: random change on exactly the weights Claude changed, same norm per tensor.
same-size:    random change on the same NUMBER of weights in the same tensors, at random
              positions, same norm per tensor.
If either control helps as much as Claude's edit, the edit's direction didn't matter.
"""
import argparse

import torch
from safetensors.torch import load_file, save_file

from apply_patch import base_dir, load_base_tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--mode", choices=["same-weights", "same-size"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = torch.Generator().manual_seed(a.seed)
    base = load_base_tensors(base_dir())
    patch = load_file(a.patch)
    out = {}
    for name, new in patch.items():
        old = base[name].float()
        delta = new.float() - old
        mask = delta != 0
        nnz = int(mask.sum())
        if nnz == 0:
            out[name] = base[name]
            continue
        if a.mode == "same-size":
            flat = torch.zeros(mask.numel(), dtype=torch.bool)
            flat[torch.randperm(mask.numel(), generator=g)[:nnz]] = True
            mask = flat.view_as(mask)
        rnd = torch.randn(delta.shape, generator=g) * mask
        rnd *= delta.norm() / rnd.norm()
        out[name] = (old + rnd).to(base[name].dtype).contiguous()
        print(f"{name}: {nnz:,} weights, norm {delta.norm():.4f}")
    save_file(out, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
