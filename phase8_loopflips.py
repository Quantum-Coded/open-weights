import sys; sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from analyze import per_problem, mcnemar_exact
R = "/home/dev_gaglani/wsp-r1/results/"
B = per_problem(R + "base_math500_greedy.jsonl")
for a in ("edit", "rand_w", "rand_s"):
    A = per_problem(R + a + "_math500_greedy.jsonl")
    rem = sum(1 for i in B if B[i]["loop"] and not A[i]["loop"]); add = sum(1 for i in B if not B[i]["loop"] and A[i]["loop"])
    print(a, "loops removed", rem, "added", add, "p", round(mcnemar_exact(add, rem), 4))
