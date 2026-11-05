import torch,numpy as np,collections,sys; sys.path.insert(0,'..')
from apply_patch import base_dir
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base_dir())
P=torch.load('work/parastarts.pt')
groups={'wait':[14190,13824],'hmm':[80022,88190],'but':[3983],'alt':[92014]}
for w in ['Hmm',' Hmm','Alternatively']: print(repr(w),tok(w,add_special_tokens=False)['input_ids'])
def rates(bias):
    r={'c':collections.Counter(),'f':collections.Counter()};N={'c':0,'f':0}
    for k,i,pos,lg in P:
        lg=lg.float().clone()
        for tid,b in bias.items(): lg[:,tid]-=b
        top=lg.argmax(1)
        N[k]+=len(top)
        for t in top.tolist(): r[k][t]+=1
    return r,N
for b in [0,0.5,1,1.5,2,3]:
    bias={t:b for t in groups['wait']}
    r,N=rates(bias)
    print('b=%.1f'%b, ' '.join(f"{k}: Wait {r[k][14190]/N[k]:.3f} But {r[k][3983]/N[k]:.3f} So {r[k][4416]/N[k]:.3f} Alt {r[k][92014]/N[k]:.3f}" for k in 'cf'))
