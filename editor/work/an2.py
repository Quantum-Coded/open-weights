import sys,json,numpy as np,torch; sys.path.insert(0,'..')
z=np.load('logs/fp/pass_3.npz')
toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
rng=np.random.default_rng(0)
for L in [14,27]:
    H=z[f'L{L}_resid']
    X=[];y=[];g=[]
    for i,m in enumerate(meta):
        n=len(toks[i]); h=H[i][-n:].astype(np.float32)
        if m['correct']: lab=0
        elif m['start']>=3000: lab=1
        else: continue
        idx=rng.choice(np.arange(100,n),size=min(150,n-100),replace=False) if n>200 else []
        for j in idx: X.append(h[j]); y.append(lab); g.append(m['id'])
    X=np.array(X);y=np.array(y);g=np.array(g)
    ids=np.unique(g); rng.shuffle(ids); half=set(ids[:len(ids)//2])
    tr=np.array([x in half for x in g]); te=~tr
    def lda(X,y):
        m0=X[y==0].mean(0);m1=X[y==1].mean(0)
        C=np.cov((X-np.where(y[:,None]==1,m1,m0)).T); lam=0.1*np.trace(C)/C.shape[0]
        w=np.linalg.solve(C+lam*np.eye(C.shape[0]),m1-m0); return w/np.linalg.norm(w),m0,m1
    w,m0,m1=lda(X[tr],y[tr])
    s=X@w; thr=(s[tr&(y==0)].mean()+s[tr&(y==1)].mean())/2
    rankdata=lambda s: np.argsort(np.argsort(s))+1.0
    def roc_auc_score(y,s):
        r=rankdata(s); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n0*n1)
    print(L,'n',len(y),'pos',y.sum(),'train auc',roc_auc_score(y[tr],s[tr]).round(3),'test auc',roc_auc_score(y[te],s[te]).round(3))
    dm=(m1-m0)/np.linalg.norm(m1-m0); s2=X@dm
    print('   diff-of-means test auc',roc_auc_score(y[te],s2[te]).round(3), 'norm diff',np.linalg.norm(m1-m0).round(2),'mean norm',np.linalg.norm(X,axis=1).mean().round(1))
