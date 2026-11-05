import sys,json,numpy as np,torch
sys.path.insert(0,".")
from wsp import load_base_tensors, base_dir, save_patch, note
# usage: make_patch.py '{"14190":1.0,"13824":1.0}'
bias={int(k):float(v) for k,v in json.loads(sys.argv[1]).items()}
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().clone(); del W
u=torch.tensor(np.load('work/u_const.npy'),dtype=torch.float32)
for t,b in bias.items(): lm[t]-=b*u
s=save_patch({'lm_head.weight':lm},replace=True)
