import torch,numpy as np,json,sys,re; sys.path.insert(0,'..')
exec(open('work/fit3.py').read().split("up=np.unique")[0])
from apply_patch import base_dir
from transformers import AutoTokenizer
from common import build_prompt_ids
tok=AutoTokenizer.from_pretrained(base_dir())
s=np.load('work/s3_lam0.01.npy')
runs={}
P={'s7':['logs/fp/pass_1.jsonl','logs/fp/pass_2.jsonl']}
for k in ['8','9','10','11']: P[k]=[f'logs/fp/pass_{k}.jsonl']
kind=lambda pid:'math' if pid.startswith('mtrain') else 'gsm8k'
for k,fs in P.items():
    for f in fs:
        for r in map(json.loads,open(f)): runs[(k,r['id'])]=r
def boxed(t):
    i=t.rfind('\\boxed{'); 
    if i<0: return None
    j=i+7; d=1; 
    while j<len(t) and d: d+= (t[j]=='{')-(t[j]=='}'); j+=1
    return t[i+7:j-1]
cache={}
def prefix_text(key,gp):
    r=runs[key]
    if key not in cache:
        p=build_prompt_ids(tok,{"kind":kind(key[1]),"question":r['question']}); ids=p+tok(r['response'],add_special_tokens=False)['input_ids']; cache[key]=(len(p),ids)
    lp,ids=cache[key]; return tok.decode(ids[lp:gp+1])
for B in [8,12,16]:
    flip=(gap-B*s)<0
    first={}
    for i,(k,f,c,p) in enumerate(zip(keys,flip,corr,para)):
        if f and p and (k in commit) and commit[k]-gp[i]>0 and not iscommit[i]:
            first[k]=min(first.get(k,1e9),gp[i])
    ok=bad=0; badl=[]
    for k,g in first.items():
        if commit[k]-g<=300: continue
        fa=boxed(runs[k]['response']); pre=prefix_text(k,int(g)).replace(' ','')
        if fa and fa.replace(' ','') in pre: ok+=1
        else: bad+=1; badl.append((k[1],fa))
    # failed runs: first flip; has gold appeared before?
    print(f'B {B}: healthy early(>300) commits: final answer already present {ok}, absent {bad}')
gold={}
for f in ['data/practice/math_train.jsonl','data/practice/gsm8k_train.jsonl']:
    for l in open(f):
        x=json.loads(l); gold[x['id']]=x['answer']
def has_gold(txt,g):
    g=g.replace(' ','').replace(',',''); t=txt.replace(' ','').replace(',','')
    if re.fullmatch(r'-?\d+(\.\d+)?',g): return re.search(r'(?<![\d.])'+re.escape(g)+r'(?![\d])',t) is not None and len(re.findall(r'(?<![\d.])'+re.escape(g)+r'(?![\d])',t))>=2
    return g in t
for B in [6,8,10,12,16]:
    flip=(gap-B*s)<0
    first={}
    for i,(k,f,c,p) in enumerate(zip(keys,flip,corr,para)):
        if f and p and not c: first[k]=min(first.get(k,1e9),gp[i])
    good=sum(has_gold(prefix_text(k,int(g)),gold[k[1]]) for k,g in first.items())
    # losses: healthy early commits with final answer absent
    firsth={}
    for i,(k,f,c,p) in enumerate(zip(keys,flip,corr,para)):
        if f and p and (k in commit) and commit[k]-gp[i]>0 and not iscommit[i]: firsth[k]=min(firsth.get(k,1e9),gp[i])
    loss=0
    for k,g in firsth.items():
        fa=boxed(runs[k]['response']); 
        if not (fa and fa.replace(' ','') in prefix_text(k,int(g)).replace(' ','')): loss+=1
    print(f'B {B}: failed runs committing {len(first)}/55, with gold already present {good}  | healthy commits w/o final answer present {loss}/121  => net est {good-loss:+d}')
