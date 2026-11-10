"""Race-demo example for run 2, by the rule fixed on 10 Nov before looking (RUNLOG):
among gsm8k_fresh400 problems where base loops and e2 finishes correct, the shortest e2 answer, ties by id."""
import json, sys
R = sys.argv[1]
def load(arm):
    return {x["id"]: x for x in map(json.loads, open(f"{R}/results/{arm}_gsm8k_fresh400_greedy.jsonl", encoding="utf-8"))}
b, e2, e1 = load("base"), load("e2"), load("e1")
c = [i for i in b if b[i]["loop"] and e2[i]["status"] == "finished" and e2[i]["correct"]]
c.sort(key=lambda i: (e2[i]["n_tokens"], i))
print("candidates:", len(c))
for i in c[:5]:
    print(i, "e2 tokens", e2[i]["n_tokens"], "| e1", e1[i]["status"], e1[i]["correct"], e1[i]["n_tokens"], "| base flag at", b[i]["first_flag"])
print("PICK", c[0])
# how many problems went each way between base and e2 (for the captions)
fixed = sum(1 for i in b if e2[i]["correct"] and not b[i]["correct"])
broke = sum(1 for i in b if b[i]["correct"] and not e2[i]["correct"])
rescued = sum(1 for i in b if b[i]["loop"] and e2[i]["correct"])
newloop = sum(1 for i in b if e2[i]["loop"] and not b[i]["loop"])
print(f"e2 vs base: fixed {fixed}, broke {broke}; base loops rescued to correct {rescued}; new loops {newloop}; base loops {sum(b[i]['loop'] for i in b)}")
