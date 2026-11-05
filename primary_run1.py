"""Pre-registered primary family (eval_plan.md), Holm over 4 tests, run 1."""
import sys; sys.path.insert(0, "/home/dev_gaglani/wsp-r1")
from analyze import per_problem, mcnemar_exact, holm
R = "/home/dev_gaglani/wsp-r1/results/"
def pair(a, b, key):
    A, B = per_problem(R + a), per_problem(R + b)
    ids = sorted(set(A) & set(B))
    x = sum(1 for i in ids if not A[i][key] and B[i][key]); y = sum(1 for i in ids if A[i][key] and not B[i][key])
    return len(ids), x, y
tests = []
n, f, b = pair("base_math500_greedy.jsonl", "edit_math500_greedy.jsonl", "acc"); tests.append(("MATH-500 acc, edit vs base", n, f, b, mcnemar_exact(f, b)))
n, on, off = pair("base_math500_greedy.jsonl", "edit_math500_greedy.jsonl", "loop"); tests.append(("MATH-500 loop rate, edit vs base (new loops / loops removed)", n, on, off, mcnemar_exact(on, off)))
n, f, b = pair("base_gsm8k_test400_greedy.jsonl", "edit_gsm8k_test400_greedy.jsonl", "acc"); tests.append(("GSM8K acc, edit vs base", n, f, b, mcnemar_exact(f, b)))
n, f, b = pair("rand_w_math500_greedy.jsonl", "edit_math500_greedy.jsonl", "acc"); tests.append(("MATH-500 acc, edit vs rand_w", n, f, b, mcnemar_exact(f, b)))
for t, ph in zip(tests, holm([t[4] for t in tests])):
    print(f"{t[0]}: n={t[1]} {t[2]}/{t[3]} p_raw={t[4]:.2g} p_holm={ph:.2g} {'PASS' if ph < 0.05 else 'fail'}")
