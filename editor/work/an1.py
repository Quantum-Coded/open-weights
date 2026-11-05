import sys,json,numpy as np,torch; sys.path.insert(0,'..')
from apply_patch import base_dir, load_base_tensors
W=load_base_tensors(base_dir())
lm=W['lm_head.weight'].float(); fn=W['model.norm.weight'].float()
del W
z=np.load('logs/fp/pass_3.npz'); H=z['L27_resid']
toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
WAIT=[14190,13824]; 
res=[]
agree=tot=0
rows=[]
for i in range(len(toks)):
    t=toks[i]; n=len(t); h=torch.tensor(H[i][-n:].astype(np.float32))
    ok=~torch.isnan(h).any(1); 
    logits=h@lm.T
    nxt=torch.tensor(t[1:]); pred=logits[:-1].argmax(1)
    agree+=(pred==nxt).sum().item(); tot+=len(nxt)
    top2=logits[:-1].topk(2,dim=1)
    for j in range(n-1):
        if pred[j].item() in WAIT:
            rows.append((i,j,meta[i]['correct'],meta[i]['status'],(top2.values[j,0]-top2.values[j,1]).item(),top2.indices[j,1].item()))
print('greedy agreement',agree/tot)
torch.save(rows,'work/waitrows.pt')
import collections
for c in [True,False]:
    m=np.array([r[4] for r in rows if r[2]==c])
    print('correct' if c else 'failed', len(m), 'margin pct', np.percentile(m,[10,25,50,75,90]).round(2))
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
print(collections.Counter(tok.decode([r[5]]) for r in rows).most_common(15))
