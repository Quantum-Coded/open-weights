import numpy as np,json
sums={'loop':[np.zeros(1536),0],'heal':[np.zeros(1536),0]}
per_src={}
def add(H,toks,meta,tag):
    s={'loop':[np.zeros(1536),0],'heal':[np.zeros(1536),0]}
    for i,m in enumerate(meta):
        n=len(toks[i]); lo=300 if m['start']>0 else 100
        h=H[i][-n:].astype(np.float32); gp=m['start']+np.arange(n)
        keep=np.arange(n)>=lo
        if 151649 in toks[i]: keep&=np.arange(n)<toks[i].index(151649)
        if m['correct']: keep&=gp>=1000; k='heal'
        else: keep&=gp>=2000; k='loop'
        if keep.sum()==0: continue
        s[k][0]+=h[keep].sum(0); s[k][1]+=keep.sum()
    per_src[tag]=s
    for k in s: sums[k][0]+=s[k][0]; sums[k][1]+=s[k][1]
z=np.load('logs/fp/pass_3.npz'); add(z['L14_resid'],json.load(open('logs/fp/pass_3_tokens.json')),json.load(open('work/read1_meta.json')),'s7')
for p,part in [(25,0),(26,1)]:
    z=np.load(f'logs/fp/pass_{p}.npz'); add(z['L14_resid'],json.load(open(f'logs/fp/pass_{p}_tokens.json')),json.load(open(f'work/read5_{part}_meta.json')),f'p{p}')
d_all=sums['loop'][0]/sums['loop'][1]-sums['heal'][0]/sums['heal'][1]
d7=np.load('work/d_loop_L14.npy')
new={k:v for k,v in per_src.items() if k!='s7'}
L=sum(v['loop'][0] for v in new.values())/sum(v['loop'][1] for v in new.values()); Hh=sum(v['heal'][0] for v in new.values())/sum(v['heal'][1] for v in new.values())
d_new=L-Hh
cos=lambda a,b:a@b/np.linalg.norm(a)/np.linalg.norm(b)
print('counts loop',sums['loop'][1],'heal',sums['heal'][1])
print('|d7| %.2f |d_new(seeds11,13,21)| %.2f |d_all| %.2f'%(np.linalg.norm(d7),np.linalg.norm(d_new),np.linalg.norm(d_all)))
print('cos(d7,d_new) %.3f cos(d7,d_all) %.3f'%(cos(d7,d_new),cos(d7,d_all)))
np.save('work/d_loop_L14_all.npy',d_all)
