"""Amendment 5: the Rubik's cube, a second application nobody edited for.

Puzzle: a solved 3x3 cube is scrambled by a short sequence of face turns, given in standard notation. The model
must give, in \\boxed{}, a move sequence that returns the cube to solved. The grader applies the scramble and then
the answer to a sticker-level simulator and checks every face is one colour. Any valid solution counts, not only
the inverse of the scramble. No LLM judge.

Simulator: each of the 54 stickers is a (position, outward normal) pair on the 3x3x3 grid, coordinates in
{-1, 0, 1}. A face turn rotates every sticker in that face's layer by 90 degrees clockwise as seen looking at the
face from outside. Solved = on each of the six faces, all nine stickers carry the same colour.
"""
from __future__ import annotations

import random
import re

from common import answer_part, last_boxed

# outward normal of each face: x = right, y = up, z = front
FACES = {"U": (0, 1, 0), "D": (0, -1, 0), "R": (1, 0, 0), "L": (-1, 0, 0), "F": (0, 0, 1), "B": (0, 0, -1)}
AXIS = {"U": 1, "D": 1, "R": 0, "L": 0, "F": 2, "B": 2}


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rot_cw(v, n):
    """Rotate integer vector v by -90 degrees about unit axis n (clockwise seen from the +n side)."""
    c = _cross(n, v)
    d = sum(x * y for x, y in zip(n, v))
    return tuple(-c[i] + n[i] * d for i in range(3))


def solved_state() -> dict:
    """{(pos, normal): colour}; the colour of a sticker is the face it starts on."""
    st = {}
    for f, n in FACES.items():
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                pos = [0, 0, 0]
                ax = AXIS[f]
                others = [i for i in range(3) if i != ax]
                pos[ax] = n[ax]
                pos[others[0]], pos[others[1]] = a, b
                st[(tuple(pos), n)] = f
    assert len(st) == 54
    return st


def turn(st: dict, face: str, quarter: int) -> dict:
    """quarter = 1 (clockwise), 2 (half turn) or 3 (counter-clockwise)."""
    n = FACES[face]
    ax = AXIS[face]
    out = {}
    for (pos, nor), col in st.items():
        if pos[ax] == n[ax]:
            for _ in range(quarter):
                pos, nor = _rot_cw(pos, n), _rot_cw(nor, n)
        out[(pos, nor)] = col
    return out


def is_solved(st: dict) -> bool:
    faces = {}
    for (pos, nor), col in st.items():
        faces.setdefault(nor, set()).add(col)
    return len(faces) == 6 and all(len(c) == 1 for c in faces.values())


# ------------------------------------------------------------------ notation
_MOVE = re.compile(r"([UDLRFB])(2|3)?('|i)?$")


def parse_moves(text: str) -> list[tuple[str, int]] | None:
    """'R U2 F'' -> [('R',1),('U',2),('F',3)]. Only the 18 face turns are accepted; None if anything else."""
    s = text.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{?\\prime\}?|\\prime|’|′|‘|`", "'", s)
    s = re.sub(r"\^\{?2\}?", "2", s)
    s = re.sub(r"\\[,;:! ]|\\quad|~", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    toks = [t for t in re.split(r"[\s,;→>\-]+", s) if t]
    moves = []
    for t in toks:
        m = _MOVE.match(t)
        if not m:
            return None
        q = 2 if m.group(2) == "2" else 3 if m.group(2) == "3" else 1
        if m.group(3):
            q = (4 - q) % 4 or 4
        q %= 4
        if q:
            moves.append((m.group(1), q))
    return moves


def apply_moves(st: dict, moves) -> dict:
    for f, q in moves:
        st = turn(st, f, q)
    return st


def fmt(moves) -> str:
    return " ".join(f + {1: "", 2: "2", 3: "'"}[q] for f, q in moves)


# ------------------------------------------------------------------ puzzles
def scramble(rng: random.Random, depth: int) -> list[tuple[str, int]]:
    """No two turns of the same face in a row, and no three turns on one axis in a row (e.g. R L R)."""
    out = []
    while len(out) < depth:
        f = rng.choice("UDLRFB")
        if out and out[-1][0] == f:
            continue
        if len(out) >= 2 and AXIS[out[-1][0]] == AXIS[out[-2][0]] == AXIS[f]:
            continue
        out.append((f, rng.choice((1, 2, 3))))
    return out


def make_row(idx: int, depth: int, moves) -> dict:
    s = fmt(moves)
    q = ("Rubik's cube. A standard 3x3 Rubik's cube starts solved. It is scrambled with these moves, in standard "
         "notation: " + s + ". (U, D, L, R, F, B turn the up, down, left, right, front and back face 90 degrees "
         "clockwise as seen looking straight at that face; a ' after a letter means counter-clockwise and a 2 "
         "means a half turn.) Give a sequence of moves, in the same notation, that returns the cube to solved. "
         "Give the move sequence itself as your final answer.")
    return {"id": f"cube-d{depth}-{idx:03d}", "kind": "cube", "depth": depth, "scramble": s, "question": q}


# ------------------------------------------------------------------ grader
def grade_cube(row: dict, response: str) -> bool:
    """Correct only if the model finished thinking and the last \\boxed{} of the answer is a face-turn sequence
    that, applied after the scramble, leaves the cube solved."""
    if "</think>" not in response:
        return False
    b = last_boxed(answer_part(response))
    if b is None:
        return False
    moves = parse_moves(b)
    if moves is None:
        return False
    st = apply_moves(solved_state(), parse_moves(row["scramble"]))
    return is_solved(apply_moves(st, moves))


if __name__ == "__main__":  # self-test
    s0 = solved_state()
    assert is_solved(s0)
    for f in FACES:
        assert not is_solved(turn(s0, f, 1))
        assert apply_moves(s0, [(f, 1)] * 4) == s0
        assert apply_moves(s0, [(f, 1), (f, 3)]) == s0
        assert apply_moves(s0, [(f, 2), (f, 2)]) == s0
    sexy = parse_moves("R U R' U'")
    st = s0
    for i in range(6):
        st = apply_moves(st, sexy)
        assert is_solved(st) == (i == 5)
    # orientation check: R moves the front-right column's front stickers up (to U)
    st = turn(s0, "R", 1)
    assert st[((1, 1, 1), (0, 1, 0))] == "F" and st[((1, 1, 1), (1, 0, 0))] == "R"
    # U moves the front top row to the left (its front stickers go to L)
    st = turn(s0, "U", 1)
    assert st[((-1, 1, 1), (-1, 0, 0))] == "F"
    # F moves the up face's front row to the right face
    st = turn(s0, "F", 1)
    assert st[((1, 1, 1), (1, 0, 0))] == "U"
    rng = random.Random(1)
    for d in range(1, 9):
        for _ in range(50):
            m = scramble(rng, d)
            inv = [(f, 4 - q) for f, q in reversed(m)]
            assert is_solved(apply_moves(apply_moves(s0, m), inv))
            assert not is_solved(apply_moves(s0, m)) or d == 0
    row = make_row(0, 3, parse_moves("R U F'"))
    ok = [r"</think> \boxed{F R' ... }", r"</think> \boxed{F U' R'}", r"</think> $\boxed{F, U^{\prime}, R'}$",
          r"</think> \boxed{F \ U' \ R'}", r"</think> \boxed{F U3 R3}", r"</think> \boxed{F U' R' U U'}",
          r"</think> \boxed{F → U' → R'}"]
    want = [False, True, True, True, True, True, True]
    for r, w in zip(ok, want):
        assert grade_cube(row, r) == w, (r, grade_cube(row, r))
    assert not grade_cube(row, r"\boxed{F U' R'}")              # never finished thinking
    assert not grade_cube(row, r"</think> \boxed{R U F'}")     # the scramble itself
    assert not grade_cube(row, r"</think> \boxed{F' U' R'}")
    print("cube self-test ok")
