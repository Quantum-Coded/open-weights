import torch,numpy as np,json,sys; sys.path.insert(0,'..')
from apply_patch import base_dir, load_base_tensors
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().cuda(); del W
w=np.load('work/w3.npy'); wt=torch.tensor(w[:1536],dtype=torch.float32).cuda(); b=float(w[1536])
u=torch.tensor(np.load('work/u_const.npy'),dtype=torch.float32).cuda()
FINAL=19357
srcs=[('logs/fp/pass_18.npz','logs/fp/pass_18_tokens.json','work/read4_meta.json'),('logs/fp/pass_3.npz','logs/fp/pass_3_tokens.json','work/read1_meta.json'),('logs/fp/pass_12.npz','logs/fp/pass_12_tokens.json','work/read3_0_meta.json'),('logs/fp/pass_13.npz','logs/fp/pass_13_tokens.json','work/read3_1_meta.json')]
rows=[]
for npz,tj,mj in srcs:
    H=np.load(npz)['L27_resid']; toks=json.load(open(tj)); meta=json.load(open(mj))
    for i in range(len(toks)):
        t=toks[i]; n=len(t)
        js=[j for j in range(1,n-1) if t[j]==334 and tok.decode([t[j-1]]).endswith('\n\n') and 151649 not in t[:j]]
        if not js: continue
        h=torch.tensor(H[i][-n:].astype(np.float32)).cuda()[js]
        lg=h@lm.T; s=(h@wt+b*(h@u)).tolist(); gap=(lg.max(1).values-lg[:,FINAL]).tolist()
        for a,j in enumerate(js): rows.append((npz[-6:-4],meta[i]['correct'],tok.decode([t[j+1]]),s[a],gap[a]))
import collections
for c in [True,False]:
    R=[r for r in rows if r[1]==c]
    hdr=[r for r in R if r[2]!='Final']; fin=[r for r in R if r[2]=='Final']
    print('correct' if c else 'failed','| after **: Final',len(fin),'header',len(hdr))
    if hdr: print('   header: s mean %.2f, Final-gap median %.1f p10 %.1f p90 %.1f'%(np.mean([r[3] for r in hdr]),*np.percentile([r[4] for r in hdr],[50,10,90])))
    if fin: print('   Final : s mean %.2f'%np.mean([r[3] for r in fin]))
    print('  ',collections.Counter(r[2] for r in hdr).most_common(8))
