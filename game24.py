"""Amendment 4: the Game of 24, an application nobody edited for.

Puzzle: four numbers from 1-13 (a multiset). Use each exactly once with + - * / and parentheses to make 24.
The model must give the EXPRESSION in \\boxed{}; the grader parses it, checks the numbers used are exactly the
puzzle's numbers (as a multiset), and evaluates it with exact fractions. No LLM judge.
"""
from __future__ import annotations

import ast
import itertools
import re
from fractions import Fraction

from common import answer_part, last_boxed

TARGET = Fraction(24)


# ------------------------------------------------------------------ solver
def solvable(nums) -> bool:
    """Exhaustive search over all orders and operator trees (exact fractions)."""
    def rec(vals):
        if len(vals) == 1:
            return vals[0] == TARGET
        for i, j in itertools.permutations(range(len(vals)), 2):
            a, b = vals[i], vals[j]
            rest = [vals[k] for k in range(len(vals)) if k not in (i, j)]
            outs = [a + b, a - b, a * b]
            if b != 0:
                outs.append(a / b)
            if any(rec(rest + [o]) for o in outs):
                return True
        return False
    return rec([Fraction(n) for n in nums])


def all_puzzles() -> list[tuple[int, ...]]:
    """Every multiset of four numbers 1-13 that can make 24, in sorted order."""
    return [c for c in itertools.combinations_with_replacement(range(1, 14), 4) if solvable(c)]


def make_row(nums: tuple[int, ...]) -> dict:
    s = ", ".join(map(str, nums))
    q = (f"Game of 24. Use the four numbers {s} exactly once each, together with +, -, ×, ÷ and parentheses, "
         f"to make an expression equal to 24. Give the expression itself (not just the number 24) as your "
         f"final answer.")
    return {"id": "g24-" + "-".join(map(str, nums)), "kind": "game24", "numbers": list(nums), "question": q}


# ------------------------------------------------------------------ grader
_LATEX = [
    (r"\\left|\\right|\\!|\\,|\\;|\\:", ""),
    (r"\\times|\\cdot|×|·|∗", "*"),
    (r"\\div|÷", "/"),
    (r"−|–|—", "-"),
    (r"\\\{|\\\}|\[|\]", lambda m: "(" if m.group(0) in ("\\{", "[") else ")"),
]
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def normalize(expr: str) -> str:
    s = expr.strip().strip("$").strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    for _ in range(4):                       # nested \frac, innermost first
        s = _FRAC.sub(r"((\1)/(\2))", s)
    for pat, rep in _LATEX:
        s = re.sub(pat, rep, s)
    s = s.replace("{", "(").replace("}", ")")
    # drop a trailing "= 24" (or a leading "24 =")
    if s.count("=") == 1:
        left, right = (x.strip() for x in s.split("="))
        s = left if right == "24" else right if left == "24" else s
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)     # 3(4+4) -> 3*(4+4)
    s = re.sub(r"\)\s*(\d|\()", r")*\1", s)  # (1+2)(3+5) -> (1+2)*(3+5)
    return s.strip()


def _eval(node) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        raise ValueError("unary minus is not allowed")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if b == 0:
            raise ZeroDivisionError
        return a / b
    raise ValueError(f"not allowed: {ast.dump(node)[:40]}")


def _literals(node) -> list[int]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)]


def check_expression(expr: str, nums) -> bool:
    try:
        tree = ast.parse(normalize(expr), mode="eval")
        return sorted(_literals(tree)) == sorted(nums) and _eval(tree) == TARGET
    except Exception:
        return False


def grade_game24(row: dict, response: str) -> bool:
    """Correct only if the model finished thinking and its last \\boxed{} holds a valid expression."""
    if "</think>" not in response:
        return False
    b = last_boxed(answer_part(response))
    return b is not None and check_expression(b, row["numbers"])


def _selftest():
    assert solvable((3, 3, 8, 8)) and solvable((1, 1, 4, 6)) and not solvable((1, 1, 1, 1))
    good = [("8/(3-8/3)", (3, 3, 8, 8)), (r"\frac{8}{3 - \frac{8}{3}}", (3, 3, 8, 8)),
            (r"(6 \times 4) \times (1 \times 1)", (1, 1, 4, 6)), ("(13-1)×(1+1)", (1, 1, 13, 1)),
            (r"6 \div (1 - \frac{3}{4})", (1, 3, 4, 6)), ("4(6) (1)(1) = 24", (1, 1, 4, 6)),
            ("(10 - 4) \\times (2 + 2)", (2, 2, 4, 10))]
    bad = [("24", (3, 3, 8, 8)), ("8*3", (3, 3, 8, 8)), ("8/(3-8/3) + 0", (3, 3, 8, 8)),
           ("6*4", (1, 1, 4, 6)), ("(6+4+1+1)*2", (1, 1, 4, 6)), ("-(1-25)", (1, 25, 0, 0))]
    for e, n in good:
        assert check_expression(e, n), (e, normalize(e))
    for e, n in bad:
        assert not check_expression(e, n), (e, normalize(e))
    assert grade_game24({"numbers": [3, 3, 8, 8]}, "think...</think> The answer is \\boxed{8 \\div (3 - 8 \\div 3)}.")
    assert not grade_game24({"numbers": [3, 3, 8, 8]}, "no think end \\boxed{8/(3-8/3)}")
    print("game24 selftest ok")


if __name__ == "__main__":
    _selftest()
    p = all_puzzles()
    print(f"{len(p)} solvable puzzles out of 1820 multisets")
