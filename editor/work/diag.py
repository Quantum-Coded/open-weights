import torch,numpy as np,json,sys; sys.path.insert(0,'..')
from apply_patch import base_dir, load_base_tensors
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
endsnn=torch.zeros(151936,dtype=torch.bool)
for i in range(151643):
    if tok.decode([i]).endswith('\n\n'): endsnn[i]=True
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().cuda(); del W
w=np.load('work/w3.npy'); wt=torch.tensor(w[:1536],dtype=torch.float32).cuda(); b=float(w[1536])
u=torch.tensor(np.load('work/u_const.npy'),dtype=torch.float32).cuda()
H=np.load('logs/fp/pass_18.npz')['L27_resid']; toks=json.load(open('logs/fp/pass_18_tokens.json')); meta=json.load(open('work/read4_meta.json'))
per={}
for i in range(len(toks)):
    t=toks[i]; n=len(t); m=meta[i]; lo=300 if m['start']>0 else 30
    h=torch.tensor(H[i][-n:].astype(np.float32)).cuda()
    lg=h@lm.T   # base logits
    s=h@wt + b*(h@u)
    gap=lg.max(1).values-lg[:,334]
    para=endsnn[torch.tensor(t)].cuda()
    idx=torch.arange(n).cuda(); sel=para&(idx>=lo)&(idx<n-1)
    d=per.setdefault(m['id'],{'s':[],'gap':[],'gp':[]})
    d['s']+=s[sel].tolist(); d['gap']+=gap[sel].tolist(); d['gp']+=(m['start']+idx[sel]).tolist()
for k,d in per.items():
    s=np.array(d['s']);g=np.array(d['gap']);gp=np.array(d['gp'])
    late=gp>=2000
    need=(g/np.maximum(s,1e-3))[late]
    print(k[:30].ljust(30),'para',len(s),'late s mean %.2f p90 %.2f max %.2f | gap median %.1f min %.1f | min B needed %.1f'%(s[late].mean(),np.percentile(s[late],90),s[late].max(),np.median(g[late]),g[late].min(),need[need>0].min() if (need>0).any() else -1))
