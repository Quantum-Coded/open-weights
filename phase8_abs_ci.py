import sys; sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
import numpy as np
from analyze import per_problem, boot_ci
R = "/home/dev_gaglani/wsp-r1/results/"
for f in ["base_math500_greedy", "edit_math500_greedy", "rand_w_math500_greedy", "rand_s_math500_greedy", "base_pp_math500_greedy", "edit_pp_math500_greedy",
          "base_gsm8k_test400_greedy", "edit_gsm8k_test400_greedy", "base_mmlupro300_greedy", "edit_mmlupro300_greedy"]:
    P = per_problem(R + f + ".jsonl")
    a = np.array([v["acc"] for v in P.values()]); l = np.array([v["loop"] for v in P.values()])
    ca = boot_ci(a, np.random.default_rng(0)); cl = boot_ci(l, np.random.default_rng(0))
    print(f"{f:28} n={len(a)} acc {100*a.mean():.1f}% [{100*ca[0]:.1f}, {100*ca[1]:.1f}]  loop {100*l.mean():.1f}% [{100*cl[0]:.1f}, {100*cl[1]:.1f}]")
