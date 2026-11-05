import sys,json,numpy as np,torch,collections; sys.path.insert(0,'..')
from apply_patch import base_dir, load_base_tensors
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().cuda(); del W
z=np.load('logs/fp/pass_3.npz'); H=z['L27_resid']
toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
vocab=[tok.decode([i]) for i in range(len(tok))] if False else None
endsnn=set(i for i in range(151643) if tok.decode([i]).endswith('\n\n'))
WAIT=14190; STAR=334; BUT=3983
out={'c':collections.Counter(),'f':collections.Counter()}
gap={'c':[], 'f':[]}
P=[]  # store logits at para starts: (label, top token, logit vec for selected tokens)
sel=None
for i in range(len(toks)):
    t=toks[i]; n=len(t)
    h=torch.tensor(H[i][-n:].astype(np.float32)).cuda()
    pos=[j for j in range(n-1) if t[j] in endsnn and j>20]
    if not pos: continue
    lg=(h[pos]@lm.T)
    k='c' if meta[i]['correct'] else 'f'
    top=lg.argmax(1).cpu().numpy()
    for a,j in enumerate(pos):
        out[k][vocab_t if False else top[a]]+=1
        gap[k].append((lg[a].max()-lg[a,STAR]).item())
    P.append((k,i,pos,lg.cpu().half()))
for k in 'cf':
    tot=sum(out[k].values()); print(k,'para starts',tot)
    print('  ',[(tok.decode([a]),round(b/tot,3)) for a,b in out[k].most_common(14)])
    print('   gap top-"**" pct',np.percentile(gap[k],[5,10,25,50]).round(2))
torch.save(P,'work/parastarts.pt')
