import torch,numpy as np,json,sys; sys.path.insert(0,'..')
from apply_patch import base_dir
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
P=torch.load('work/parastarts.pt'); toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
STAR=334; THINK=151649
# global token offset of commit per problem
res={}
for k,i,pos,lg in P:
    if k!='c': continue
    t=toks[i]; m=meta[i]; off=m['start']+ (1000-len(t)) if False else m['start']
    lg=lg.float()
    for a,j in enumerate(pos):
        gap=(lg[a].max()-lg[a,STAR]).item()
        res.setdefault(m['id'],[]).append((off+j,gap,t[j+1]==STAR, THINK in t[:j]))
for pid,v in list(res.items())[:60]:
    commit=[p for p,g,c,after in v if c and not after]
    if not commit: continue
    cp=commit[0]
    near=[(cp-p) for p,g,c,after in v if g<4 and not c and p<cp]
    if near: print(pid[:26].ljust(26),'commit@',cp,' near(gap<4) tokens-before-commit:',sorted(near)[:12])
