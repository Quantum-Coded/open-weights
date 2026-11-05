import torch,numpy as np,json,sys
recs=torch.load('work/ds.pt',weights_only=False)
seen=set(); R=[]
for r in recs:
    k=(r['key'],r['gp'])
    if k in seen: continue
    seen.add(k); R.append(r)
commit={}
for r in R:
    if r['correct'] and r['para'] and r['nextstar']:
        commit[r['key']]=min(commit.get(r['key'],1e9),r['gp'])
def label(r):
    if r['correct']:
        cp=commit.get(r['key'])
        if cp is None: return np.nan
        if r['gp']>cp: return np.nan
        if r['para'] and r['gp']==cp: return 1.0
        if r['para'] and cp-r['gp']<150: return np.nan
        return 0.0
    else:
        if not r['para']: return 0.0
        return 1.0 if r['gp']>=2000 else np.nan
for r in R: r['y']=label(r)
X=np.array([r['h'] for r in R],dtype=np.float32); y=np.array([r['y'] for r in R]); 
pids=np.array([r['pid'] for r in R]); para=np.array([r['para'] for r in R]); corr=np.array([r['correct'] for r in R])
gap=np.array([r['gap'] for r in R]); gp=np.array([r['gp'] for r in R]); keys=[r['key'] for r in R]
iscommit=np.array([r['correct'] and r['para'] and r['nextstar'] and commit.get(r['key'])==r['gp'] for r in R])
print('records',len(R),'pos',np.nansum(y==1),'neg',np.nansum(y==0),'commits',iscommit.sum())
up=np.unique(pids); rng=np.random.default_rng(0); rng.shuffle(up); F=5; fold={p:i%F for i,p in enumerate(up)}
fo=np.array([fold[p] for p in pids])
def fit(m,lam):
    A=np.c_[X[m],np.ones(m.sum())]; yy=y[m]
    # weight positives/negatives equally per class
    wpos=0.5/ (yy==1).sum(); wneg=0.5/(yy==0).sum(); sw=np.where(yy==1,wpos,wneg)
    G=(A*sw[:,None]).T@A; G[:1536,:1536]+=lam*np.eye(1536)*np.trace(G)/1537
    return np.linalg.solve(G,(A*sw[:,None]).T@yy)
for lam in [1e-3,1e-2,1e-1]:
    s=np.zeros(len(R))
    for f in range(F):
        tr=(fo!=f)&~np.isnan(y); w=fit(tr,lam); te=fo==f; s[te]=np.c_[X[te],np.ones(te.sum())]@w
    print(f'lam {lam}: mean s  pos {s[y==1].mean():.2f} neg-para {s[(y==0)&para].mean():.2f} neg-random {s[(y==0)&~para].mean():.2f} commits {s[iscommit].mean():.2f}')
    for B in [8,12,16,20]:
        flip=(gap-B*s)<0
        fk=set(k for k,f,c,ic in zip(keys,flip,corr,iscommit) if f and not c)
        allf=set(k for k,c in zip(keys,corr) if not c)
        early=set(k for k,f,c,g,p in zip(keys,flip,corr,gp,para) if f and c and p and commit.get(k) is not None and commit[k]-g>300)
        allh=set(k for k in keys if k in commit)
        inline=(flip&~para&~np.isnan(y)).sum()
        kept=(flip&iscommit).mean()/iscommit.mean() if iscommit.any() else 0
        print(f'   B {B}: failed runs w/ commit {len(fk)}/{len(allf)}  healthy early>300 {len(early)}/{len(allh)}  inline flips {inline}/{(~para).sum()}')
    np.save(f'work/s3_lam{lam}.npy',s)
w=fit(~np.isnan(y),1e-2); np.save('work/w3.npy',w)
