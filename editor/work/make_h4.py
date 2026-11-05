import sys,numpy as np,torch; sys.path.insert(0,'.')
from wsp import load_base_tensors, base_dir, save_patch
alpha=float(sys.argv[1]); Ls=[10,11,12,13,14]
W=load_base_tensors(base_dir())
d=np.load(sys.argv[2] if len(sys.argv)>2 else 'work/d_loop_L14.npy'); target=-alpha*d
cols=[]
E=np.zeros((1536,256),dtype=np.float32)
for hd in range(12): g=hd//6; E[hd*128:(hd+1)*128, g*128:(g+1)*128]=np.eye(128)
for l in Ls: cols.append(W[f'model.layers.{l}.self_attn.o_proj.weight'].float().numpy()@E)
A=np.concatenate(cols,1)
lam=1e-2*np.trace(A.T@A)/A.shape[1]
x=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@target)
print('realized fraction %.3f  |realized| %.2f |target| %.2f'%(1-np.linalg.norm(A@x-target)**2/np.linalg.norm(target)**2,np.linalg.norm(A@x),np.linalg.norm(target)))
mod={}
for k,l in enumerate(Ls):
    name=f'model.layers.{l}.self_attn.v_proj.bias'; b=W[name].float().clone()
    db=torch.tensor(x[k*256:(k+1)*256],dtype=torch.float32)
    print(name,'|b| %.2f |db| %.2f'%(b.norm(),db.norm()))
    mod[name]=b+db
save_patch(mod,replace=True)
