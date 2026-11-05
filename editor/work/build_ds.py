import torch,numpy as np,json,sys; sys.path.insert(0,'..')
from apply_patch import base_dir, load_base_tensors
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
endsnn=torch.zeros(151936,dtype=torch.bool)
for i in range(151643):
    if tok.decode([i]).endswith('\n\n'): endsnn[i]=True
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().cuda(); del W
STAR=334; THINK=151649
src=[('logs/fp/pass_3.npz','logs/fp/pass_3_tokens.json','work/read1_meta.json'),
     ('logs/fp/pass_12.npz','logs/fp/pass_12_tokens.json','work/read3_0_meta.json'),
     ('logs/fp/pass_13.npz','logs/fp/pass_13_tokens.json','work/read3_1_meta.json')]
# pass1: commit global positions for healthy runs (first ** at para start before </think>)
recs=[]; rng=np.random.default_rng(0)
for npz,tj,mj in src:
    H=np.load(npz)['L27_resid']; toks=json.load(open(tj)); meta=json.load(open(mj))
    for i in range(len(toks)):
        t=toks[i]; n=len(t); m=meta[i]; key=(m.get('pass','s7'),m['id'])
        lo=300 if m['start']>0 else 30
        h=torch.tensor(H[i][-n:].astype(np.float32)).cuda()
        lg=h@lm.T
        gap=(lg.max(1).values-lg[:,STAR]).cpu().numpy()
        tt=torch.tensor(t)
        para=endsnn[tt].numpy()
        tend=t.index(THINK) if THINK in t else n
        hc=h.cpu().numpy().astype(np.float16)
        for j in range(lo,min(n-1,tend)):
            isp=bool(para[j]); 
            if not isp and rng.random()>0.04: continue
            recs.append(dict(key=key,pid=m['id'],gp=m['start']+j,para=isp,nextstar=(t[j+1]==STAR),gap=float(gap[j]),correct=m['correct'],h=hc[j]))
print(len(recs))
torch.save(recs,'work/ds.pt')
