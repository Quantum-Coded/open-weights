import numpy as np,json,torch,sys; sys.path.insert(0,'..')
from apply_patch import base_dir, load_base_tensors
z=np.load('logs/fp/pass_3.npz'); H=z['L14_resid']; toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
loop=[];heal=[];heal_late=[];fail_early=[]
for i,m in enumerate(meta):
    n=len(toks[i]); lo=300 if m['start']>0 else 100
    h=H[i][-n:][lo:].astype(np.float32); gp=m['start']+np.arange(lo,n)
    if 151649 in toks[i]: 
        e=toks[i].index(151649); h=h[:max(0,e-lo)]; gp=gp[:len(h)]
    if m['correct']:
        heal.append(h); heal_late.append(h[gp>=1000])
    else:
        loop.append(h[gp>=2000]); fail_early.append(h[gp<1500])
L=np.concatenate(loop);Hh=np.concatenate(heal);HL=np.concatenate(heal_late);FE=np.concatenate(fail_early)
d=L.mean(0)-Hh.mean(0); d2=L.mean(0)-HL.mean(0); d3=FE.mean(0)-Hh.mean(0)
cos=lambda a,b: a@b/np.linalg.norm(a)/np.linalg.norm(b)
print('n',len(L),len(Hh),len(HL),len(FE))
print('|loop-heal| %.1f |loop-heal_late| %.1f |failearly-heal| %.1f  mean|h| %.1f'%(np.linalg.norm(d),np.linalg.norm(d2),np.linalg.norm(d3),np.linalg.norm(Hh,axis=1).mean()))
print('cos(d,d2) %.2f cos(d,d3) %.2f'%(cos(d,d2),cos(d2,d3)))
# std along d2 in each class
u=d2/np.linalg.norm(d2)
print('proj on d2: loop %.1f+-%.1f heal %.1f+-%.1f heal_late %.1f+-%.1f'%((L@u).mean(),(L@u).std(),(Hh@u).mean(),(Hh@u).std(),(HL@u).mean(),(HL@u).std()))
np.save('work/d_loop_L14.npy',d2)
W=load_base_tensors(base_dir())
for Ls in [[14],[12,13,14],[10,11,12,13,14]]:
    cols=[]
    for l in Ls:
        O=W[f'model.layers.{l}.self_attn.o_proj.weight'].float().numpy()  # 1536 x 1536
        E=np.zeros((1536,256),dtype=np.float32)
        for hd in range(12):
            g=hd//6
            E[hd*128:(hd+1)*128, g*128:(g+1)*128]=np.eye(128)
        cols.append(O@E)
    A=np.concatenate(cols,1)
    x,res,rk,sv=np.linalg.lstsq(A,d2,rcond=None)
    print('layers',Ls,'fraction of d2 realizable %.3f'%(1-np.linalg.norm(A@x-d2)**2/np.linalg.norm(d2)**2),'|db| %.2f'%np.linalg.norm(x))
