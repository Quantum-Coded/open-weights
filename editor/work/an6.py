import torch,numpy as np,json,sys; sys.path.insert(0,'..')
from apply_patch import base_dir
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
P=torch.load('work/parastarts.pt'); toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
STAR=334; THINK=151649
commit=[];loopg=[];healthy_other=[]
for k,i,pos,lg in P:
    lg=lg.float(); t=toks[i]
    # is there </think> after this window position? find think end index in this window
    tend=t.index(THINK) if THINK in t else None
    for a,j in enumerate(pos):
        if tend is not None and j>tend: continue   # skip post-think summary
        top=lg[a].max().item(); s=lg[a,STAR].item()
        second=lg[a].topk(2).values[1].item()
        if t[j+1]==STAR and k=='c': commit.append(top-second if lg[a].argmax()==STAR else s-top)
        elif k=='c': healthy_other.append(top-s)
        else: loopg.append(top-s)
commit=np.array(commit);loopg=np.array(loopg);ho=np.array(healthy_other)
print('commit points (c):',len(commit),'margin of ** over runner-up', np.round(commit,2))
print('healthy non-commit para starts: gap top-** pct5/10/25/50', np.percentile(ho,[1,5,10,25,50]).round(2), len(ho))
print('failed para starts gap pct', np.percentile(loopg,[1,5,10,25,50]).round(2), len(loopg))
for b in [2,3,4,5,6]:
    print('boost',b,' healthy-premature frac %.4f'%(ho<b).mean(),' failed would-commit frac %.4f'%(loopg<b).mean())
