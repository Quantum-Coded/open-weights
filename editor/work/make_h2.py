import sys,json,numpy as np,torch; sys.path.insert(0,'.')
from wsp import load_base_tensors, base_dir, save_patch
# args: B_commit  wait_bias
B=float(sys.argv[1]); wb=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().clone(); del W
w=np.load('work/w_commit.npy'); u=torch.tensor(np.load('work/u_const.npy'),dtype=torch.float32)
lm[334]+=B*(torch.tensor(w[:1536],dtype=torch.float32)+float(w[1536])*u)
if wb:
    for t in (14190,13824): lm[t]-=wb*u
save_patch({'lm_head.weight':lm},replace=True)
