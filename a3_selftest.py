"""Check: a3_build.py's patch step, fed run 1's direction, reproduces run 1's frozen patch."""
import sys, ast, numpy as np, torch
sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
src = open("/home/dev_gaglani/wsp-r1/a3_build.py").read(); ast.parse(src)
from apply_patch import base_dir, load_base_tensors
from safetensors.torch import load_file
d = np.load("/home/dev_gaglani/wsp-r1/editor/work/d_loop_L14.npy")
Wt = load_base_tensors(base_dir()); LAYERS = [10, 11, 12, 13, 14]
E = np.zeros((1536, 256), dtype=np.float32)
for hd in range(12):
    g = hd // 6; E[hd*128:(hd+1)*128, g*128:(g+1)*128] = np.eye(128)
A = np.concatenate([Wt[f"model.layers.{l}.self_attn.o_proj.weight"].float().numpy() @ E for l in LAYERS], 1)
target = -0.5 * d; lam = 1e-2 * np.trace(A.T @ A) / A.shape[1]
x = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ target)
P = load_file("/home/dev_gaglani/wsp-r1/work/frozen_patch_run1.safetensors")
tot = diff = 0
for k, l in enumerate(LAYERS):
    name = f"model.layers.{l}.self_attn.v_proj.bias"; b = Wt[name]
    mine = (b.float() + torch.tensor(x[k*256:(k+1)*256], dtype=torch.float32)).to(b.dtype)
    tot += mine.numel(); diff += int((mine != P[name]).sum())
print(f"a3_build parses; patch step vs run-1 frozen patch: {diff} of {tot} values differ")
