# full-position side-effect check for H2 commit-gate edit (B=12), using full-data w
import torch,numpy as np,json,sys; sys.path.insert(0,'..')
from apply_patch import base_dir, load_base_tensors
W=load_base_tensors(base_dir()); lm=W['lm_head.weight'].float().cuda(); del W
w=np.load('work/w_commit.npy'); u=np.load('work/u_const.npy'); B=12.0
dw=torch.tensor(B*(w[:1536]+w[1536]*u),dtype=torch.float32).cuda()
z=np.load('logs/fp/pass_3.npz'); H=z['L27_resid']
toks=json.load(open('logs/fp/pass_3_tokens.json')); meta=json.load(open('work/read1_meta.json'))
STAR=334; THINK=151649
stats={True:[0,0],False:[0,0]}
shift=[]
for i in range(len(toks)):
    t=toks[i]; n=len(t); h=torch.tensor(H[i][-n:].astype(np.float32)).cuda()
    lg=h@lm.T; d=h@dw; shift.append(d.cpu().numpy()[300 if meta[i]['start']>0 else 0:])
    newtop=(lg[:,STAR]+d)>lg.max(1).values
    was=lg.argmax(1)==STAR
    lo=300 if meta[i]['start']>0 else 0
    c=meta[i]['correct']
    stats[c][0]+=int((newtop&~was)[lo:].sum()); stats[c][1]+=n-lo
print('new ** argmax positions: healthy %d/%d  failed %d/%d'%(stats[True][0],stats[True][1],stats[False][0],stats[False][1]))
s=np.concatenate(shift); print('logit shift on ** pct',np.percentile(s,[1,10,50,90,99]).round(2))
