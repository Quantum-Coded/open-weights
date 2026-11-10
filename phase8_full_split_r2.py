import json
from math import comb
def binomtest(x,n):
    k=min(x,n-x); p=2*sum(comb(n,i) for i in range(k+1))/2**n
    class R: pass
    r=R(); r.pvalue=min(1.0,p); return r
L=lambda f:{r["id"]:r for r in map(json.loads,open(f"results/{f}.jsonl"))}
b,e=L("base_full_math500_greedy"),L("edit_full_r2_math500_greedy")
fast=set(L("base_math500_greedy"))
def mc(ids,key):
    x=sum(1 for i in ids if e[i][key] and not b[i][key]); y=sum(1 for i in ids if b[i][key] and not e[i][key])
    p=binomtest(x,x+y).pvalue if x+y else 1
    return x,y,p
for name,ids in [("all500",list(b)),("fast200",[i for i in b if i in fast]),("rest300",[i for i in b if i not in fast])]:
    n=len(ids); ab=sum(b[i]["correct"] for i in ids)/n; ae=sum(e[i]["correct"] for i in ids)/n
    lb=sum(bool(b[i]["loop"]) for i in ids)/n; le=sum(bool(e[i]["loop"]) for i in ids)/n
    print(name,n,f"acc {ab:.3f}->{ae:.3f}","fixed/broken/p",mc(ids,"correct"),f"loop {lb:.3f}->{le:.3f}","loops added/removed/p",mc(ids,"loop"))
# consistency: fast200 subset in full run equals earlier fast-track run?
fb=L("base_math500_greedy"); fe=L("edit_r2_math500_greedy")
print("fast-track identical text base:",sum(fb[i]["response"]==b[i]["response"] for i in fb),"edit:",sum(fe[i]["response"]==e[i]["response"] for i in fe))
