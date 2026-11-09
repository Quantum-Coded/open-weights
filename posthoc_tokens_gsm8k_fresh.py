"""POST HOC (not pre-registered), descriptive: tokens spent per arm on gsm8k_fresh400.
Results live in ~/wsp_sealed/results while editing run 2 runs; pass that dir as argv[1].
Loops were killed by the detector, so "nokill" counts a loop as the full 16,384-token cap."""
import json, statistics as st, sys
R = sys.argv[1]
CAP = 16384
def load(a):
    return {d["id"]: d for d in map(json.loads, open(f"{R}/{a}_gsm8k_fresh400_greedy.jsonl", encoding="utf-8"))}
arms = {a: load(a) for a in ["base", "e1", "e1x1.5", "rand_w"]}
ids = sorted(arms["base"])
print("arm      acc   loop  measured_tok_total  nokill_tok_total(loops->16384)  median_tok  mean_tok_correct")
for a, D in arms.items():
    meas = sum(D[i]["n_tokens"] for i in ids)
    nok = sum(CAP if D[i]["loop"] else D[i]["n_tokens"] for i in ids)
    med = st.median(CAP if D[i]["loop"] else D[i]["n_tokens"] for i in ids)
    cor = [D[i]["n_tokens"] for i in ids if D[i]["correct"]]
    print(f"{a:7s} {sum(D[i]['correct'] for i in ids)/len(ids):.3f} {sum(D[i]['loop'] for i in ids)/len(ids):.3f} {meas:12,d} {nok:14,d} {med:10.0f} {st.mean(cor):8.0f}")
B, E = arms["base"], arms["e1"]
rescued = [i for i in ids if B[i]["loop"] and E[i]["correct"] and not E[i]["loop"]]
broke = [i for i in ids if E[i]["loop"] and B[i]["correct"] and not B[i]["loop"]]
print("base loops -> e1 correct finish:", len(rescued), " reverse:", len(broke))
print("e1 tokens on rescued: median", st.median(E[i]["n_tokens"] for i in rescued), "min", min(E[i]["n_tokens"] for i in rescued))
print("short rescues (<1500 tok):", sorted(((E[i]["n_tokens"], B[i]["first_flag"], i) for i in rescued))[:12])
nb = sum(CAP if B[i]["loop"] else B[i]["n_tokens"] for i in ids); ne = sum(CAP if E[i]["loop"] else E[i]["n_tokens"] for i in ids)
print(f"no-kill compute saving e1 vs base: {1-ne/nb:.1%}")
