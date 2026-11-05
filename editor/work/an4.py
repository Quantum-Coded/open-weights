import json,numpy as np
z=np.load('logs/fp/pass_3.npz'); H=z['L27_resid']
toks=json.load(open('logs/fp/pass_3_tokens.json'))
rng=np.random.default_rng(1)
X=[];G=[]
for i in range(len(toks)):
    n=len(toks[i]); h=H[i][-n:].astype(np.float32)
    idx=rng.choice(n,size=min(200,n),replace=False); X.append(h[idx]); G+= [i]*len(idx)
X=np.concatenate(X); G=np.array(G)
tr=G%2==0; te=~tr
mu=X[tr].mean(0)
u_mean=mu/(mu@mu)
for lam in [0, 1e-2,1e-1,1, 10]:
    A=X[tr]; u=np.linalg.solve(A.T@A+lam*len(A)*np.eye(A.shape[1]), A.T@np.ones(len(A)))
    p=X[te]@u; print('lam',lam,'test mean %.3f std %.3f  p5 %.3f p95 %.3f  |u| %.3f'%(p.mean(),p.std(),*np.percentile(p,[5,95]),np.linalg.norm(u)))
p=X[te]@u_mean; print('mean-dir: mean %.3f std %.3f p5 %.3f p95 %.3f'%(p.mean(),p.std(),*np.percentile(p,[5,95])))
np.save('work/u_const.npy', np.linalg.solve(X.T@X+1e-2*len(X)*np.eye(1536), X.T@np.ones(len(X))))
print('top dims of mu', np.argsort(-np.abs(mu))[:5], np.sort(-np.abs(mu))[:5])
