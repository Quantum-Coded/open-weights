"""POST HOC: what do the wrong lm-eval answers look like? (descriptive)"""
import json, glob, sys, re
R = sys.argv[1]
BOX = re.compile(r"\\boxed\{([^{}]*)\}")
for arm in ["base", "edit_r2"]:
    f = glob.glob(f"{R}/results/lmeval/{arm}/*/samples_*.jsonl")[0]
    xs = [x for x in map(json.loads, open(f, encoding="utf-8")) if x["filter"] == "flexible-extract"]
    wrong = [x for x in xs if x["exact_match"] == 0]
    cats = {}
    for x in wrong:
        t = x["resps"][0][0]
        if len(t) > 20000:
            c = "loop/cap (no </think>)"
        elif t.lstrip().startswith(("Okay", "Alright", "Hmm", "First,", "So,")) or "Wait," in t[:3000]:
            c = "looks like unclosed thinking (stopped early?)"
        elif BOX.search(t):
            c = "finished, boxed wrong"
        else:
            c = "finished, no box"
        cats[c] = cats.get(c, 0) + 1
    print(arm, "wrong:", len(wrong), cats)
    # examples of unclosed thinking
    ex = [x for x in wrong if len(x["resps"][0][0]) <= 20000 and (x["resps"][0][0].lstrip().startswith(("Okay", "Alright", "Hmm")) or "Wait," in x["resps"][0][0][:3000])][:2]
    for x in ex:
        t = x["resps"][0][0]
        print("  e.g. doc", x["doc_id"], len(t), "chars; tail:", repr(t[-200:]))
