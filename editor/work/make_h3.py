import sys,numpy as np,torch; sys.path.insert(0,'.')
from wsp import load_base_tensors, base_dir, save_patch
B=float(sys.argv[1])
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().clone(); del W
w=np.load('work/w3.npy'); u=torch.tensor(np.load('work/u_const.npy'),dtype=torch.float32)
dw=B*(torch.tensor(w[:1536],dtype=torch.float32)+float(w[1536])*u)
lm[334]+=dw
print('row delta norm',float(dw.norm()),'row norm',float(lm[334].norm()))
save_patch({'lm_head.weight':lm},replace=True)
