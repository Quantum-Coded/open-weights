import torch,numpy as np,json,sys
z=np.load('logs/fp/pass_3.npz'); H=z['L27_resid']
P=torch.load('work/parastarts.pt'); toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
STAR=334; THINK=151649
rows=[] # (pid, label, h, gap, is_commit, tokens_before_commit, failed, globalpos)
commit_at={}
for k,i,pos,lg in P:
    t=toks[i]; m=meta[i]
    for a,j in enumerate(pos):
        if THINK in t[:j+1]: continue
        if t[j+1]==STAR and m['correct']: commit_at.setdefault(m['id'],m['start']+j)
for k,i,pos,lg in P:
    t=toks[i]; m=meta[i]; lg=lg.float(); n=len(t); h=H[i][-n:].astype(np.float32)
    for a,j in enumerate(pos):
        if THINK in t[:j+1]: continue
        if m['start']>0 and j<300: continue
        gp=m['start']+j
        gap=(lg[a].max()-lg[a,STAR]).item()
        if m['correct']:
            cp=commit_at.get(m['id']); 
            if cp is None or gp>cp: continue
            lab=0.0
        else:
            lab=1.0 if gp>=2000 else np.nan
        rows.append(dict(pid=m['id'],y=lab,h=h[j],gap=gap,commit=(m['correct'] and gp==commit_at.get(m['id'])),tbc=(commit_at.get(m['id'],0)-gp) if m['correct'] else None,failed=not m['correct'],gp=gp))
# extra negatives: random non-para positions from all runs (target 0)
rng=np.random.default_rng(0); extra=[]
for i in range(len(toks)):
    n=len(toks[i]); h=H[i][-n:].astype(np.float32)
    ar=np.arange(300 if meta[i]["start"]>0 else 0,n)
    if len(ar)<20: continue
    for j in rng.choice(ar,20,replace=False): extra.append((meta[i]['id'],h[j]))
pids=sorted(set(r['pid'] for r in rows)); rng.shuffle(pids)
def run(trainset,testset,lam=1.0):
    X=[r['h'] for r in rows if r['pid'] in trainset and not np.isnan(r['y'])]+[h for p,h in extra if p in trainset]
    y=[r['y'] for r in rows if r['pid'] in trainset and not np.isnan(r['y'])]+[0.0]*sum(p in trainset for p,h in extra)
    X=np.array(X);y=np.array(y)
    Xa=np.c_[X,np.ones(len(X))]
    w=np.linalg.solve(Xa.T@Xa+lam*len(X)*np.diag(np.r_[np.ones(1536),0]),Xa.T@y)
    return w
folds=[set(pids[f::4]) for f in range(4)]
out=[]
for lam in [1e-3,1e-2,1e-1]:
  res=[]
  for f in range(4):
    te=folds[f]; tr=set(pids)-te
    w=run(tr,te,lam)
    for r in rows:
        if r['pid'] in te: res.append((r,float(np.r_[r['h'],1]@w)))
  for B in [6,10,14,18]:
    prem=0; prem_far=0; nh=0; fail_commit=set(); fail_runs=set(); healthy_runs=set(); healthy_bad=set(); commit_kept=0; ncommit=0
    for r,s in res:
        flips = (r['gap'] - B*s) < 0 if not r['commit'] else None
        if r['failed']:
            fail_runs.add(r['pid'])
            if flips: fail_commit.add(r['pid'])
        else:
            healthy_runs.add(r['pid'])
            if r['commit']: continue
            nh+=1
            if flips:
                prem+=1
                if r['tbc']>300: healthy_bad.add(r['pid'])
    print(f'lam {lam} B {B}: failed runs with >=1 commit {len(fail_commit)}/{len(fail_runs)} | healthy para flips {prem}/{nh} | healthy runs w/ commit >300tok early {len(healthy_bad)}/{len(healthy_runs)}')
  sc=np.array([s for r,s in res]); yy=np.array([r['failed'] for r,s in res])
  print('   mean score healthy %.3f failed %.3f'%(sc[~yy].mean(),sc[yy].mean()))
print('----')
w=run(set(pids),set(),1e-2); np.save('work/w_commit.npy',w)
res=[]
for f in range(4):
    te=folds[f]; w2=run(set(pids)-te,te,1e-2)
    res+=[(r,float(np.r_[r['h'],1]@w2)) for r in rows if r['pid'] in te]
src={}
for f in ['logs/fp/pass_1.jsonl','logs/fp/pass_2.jsonl']:
    for x in map(json.loads,open(f)): src[x['id']]=x
gold={}
for f in ['data/practice/math_train.jsonl','data/practice/gsm8k_train.jsonl']:
    for l in open(f):
        x=json.loads(l)
        if x['id'] in src: gold[x['id']]=x['answer']
for B in [6,10]:
    print('B',B)
    for pid in sorted(set(r['pid'] for r,s in res if r['failed'])):
        fl=sorted(r['gp'] for r,s in res if r['pid']==pid and r['gap']-B*s<0)
        print('  ',pid[:28].ljust(28),src[pid]['status'],src[pid]['n_tokens'],'gold',gold[pid],'first commit tok',fl[:3])
