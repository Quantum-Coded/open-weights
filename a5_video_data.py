"""Amendment 5 video data: writes window.CUBE = {...} for the cube cut of the race video (descriptive only).

  python a5_video_data.py --primary results/a5_primary.json --race-dir results/race --out <video>/assets/cube-data.js
  python a5_video_data.py --placeholder --out <video>/assets/cube-data.js      # layout only, marked PLACEHOLDER

Race lanes replay the single-stream recordings (results/race/<arm>_<id>.json): the token curve, each sentence with
the time it was finished and whether it was said before in the same answer, and the final answer's moves.
The boards show all 100 test puzzles: scramble and whether each arm solved it. Totals come from the batched test
runs. A placeholder file is for checking the layout and must never be published.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import answer_part, last_boxed  # noqa: E402
from cube import parse_moves  # noqa: E402

CAP = 16384


def stream_pieces(tok, ids):
    pieces, buf = [], []
    for i in ids:
        buf.append(i)
        s = tok.decode(buf, skip_special_tokens=True)
        if "�" in s and len(buf) < 8:
            pieces.append("")
            continue
        pieces.append(s)
        buf = []
    return pieces


def sentences(text):
    spans, start = [], 0
    for i, ch in enumerate(text):
        end = None
        if ch == "\n":
            end = i + 1
        elif ch in ".!?" and (i + 1 == len(text) or text[i + 1] in " \n"):
            end = i + 1
        if end is not None and text[start:end].strip():
            spans.append((start, end))
            start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def clip(s, n=110):
    s = re.sub(r"\\boxed\{(.*)\}", r"Answer: \1", s.strip())
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]|\$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def lane(rec, tok_ids, times, text, finish, correct):
    """times[k] = seconds when token k+1 existed."""
    n = len(tok_ids)
    # char offset at the end of each token
    ends, acc = [], 0
    for p in rec["_pieces"]:
        acc += len(p)
        ends.append(acc)
    def t_at_char(c):
        lo, hi = 0, n - 1
        while lo < hi:
            m = (lo + hi) // 2
            if ends[m] >= c:
                hi = m
            else:
                lo = m + 1
        return times[lo] if n else 0.0
    seen, ticker = set(), []
    think_end = text.find("</think>")
    for a, b in sentences(text):
        s = text[a:b]
        if not s.strip() or "</think>" in s:
            continue
        k = norm(s)
        rep = k in seen and len(k) > 12
        seen.add(k)
        ticker.append([round(t_at_char(b), 3), clip(s), int(rep)])
    ans = last_boxed(answer_part(text)) if "</think>" in text else None
    moves = parse_moves(ans) if ans else None
    step = max(1, n // 400)
    curve = [[round(times[k], 3), k + 1] for k in range(0, n, step)]
    if n:
        curve.append([round(times[-1], 3), n])
    return {"tokens": n, "wall_s": round(times[-1], 3) if n else 0.0, "finish": finish, "correct": bool(correct),
            "answer_text": ans, "answer": [[f, q] for f, q in moves] if moves else None,
            "think_end_s": round(t_at_char(think_end), 3) if think_end >= 0 else None,
            "curve": curve, "ticker": ticker}


def from_recording(tok, path):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = r["token_ids"]
    # steps: [n_tokens_so_far, t]; fill a time for every token
    times, j = [], 0
    for k in range(1, len(ids) + 1):
        while j < len(r["steps"]) - 1 and r["steps"][j][0] < k:
            j += 1
        times.append(r["steps"][j][1])
    r["_pieces"] = stream_pieces(tok, ids)
    text = "".join(r["_pieces"])
    return lane(r, ids, times, text, "cap" if r["finish_reason"] == "length" else "stop", r.get("correct")), r


def from_text(tok, text, tok_s):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    rec = {"_pieces": stream_pieces(tok, ids)}
    times = [(k + 1) / tok_s for k in range(len(ids))]
    return lane(rec, ids, times, "".join(rec["_pieces"]), "cap" if len(ids) >= CAP else "stop", None)


def board(rows, per):
    out = []
    for p in per:
        r = rows[p["id"]]
        out.append({"id": p["id"], "depth": p["depth"], "scramble": [[f, q] for f, q in parse_moves(r["scramble"])],
                    "base": int(p["base_correct"]), "e2": int(p["e2_correct"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=str(ROOT / "results/a5_primary.json"))
    ap.add_argument("--race-dir", default=str(ROOT / "results/race"))
    ap.add_argument("--set", default=str(ROOT / "sealed/cube_100.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--placeholder", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "work/base"))

    if a.placeholder:
        # Layout only. Scramble and the looping text come from the PILOT (never a test puzzle); the edit's lane
        # and every number are made up and the video shows a PLACEHOLDER tag.
        pilot = [json.loads(l) for l in open(ROOT / "a5_data/cube_pilot12_v2.jsonl", encoding="utf-8")]
        res = {r["id"]: r for r in map(json.loads, open(ROOT / "results/pilot2_a5-cube_pilot12_v2_greedy.jsonl", encoding="utf-8"))}
        row = next(r for r in pilot if r["depth"] == 4)
        loop_text = res[row["id"]]["response"]
        base_text = loop_text
        while len(tok(base_text, add_special_tokens=False)["input_ids"]) < CAP:
            base_text += "\n" + loop_text[len(loop_text) // 2:]
        base_ids = tok(base_text, add_special_tokens=False)["input_ids"][:CAP]
        base_lane = from_text(tok, tok.decode(base_ids), 34.0)
        mv = parse_moves(row["scramble"])
        inv = " ".join(f + {1: "'", 2: "2", 3: ""}[q] for f, q in reversed(mv))
        e2_text = ("Okay, so the cube was scrambled with " + row["scramble"] + ". To undo a sequence, I reverse the order "
                   "and invert each move. PLACEHOLDER: the edited model's real reasoning appears here. ") * 18
        e2_text += "\n</think>\n\nUndo each move in reverse order.\n\n\\boxed{" + inv + "}"
        e2_lane = from_text(tok, e2_text, 43.0)
        e2_lane["correct"] = True
        import random
        rng = random.Random(7)
        from cube import scramble
        brd = []
        for d in range(2, 7):
            for i in range(20):
                brd.append({"id": f"ph-{d}-{i}", "depth": d, "scramble": [[f, q] for f, q in scramble(rng, d)],
                            "base": 0, "e2": int(rng.random() < 0.55 - 0.08 * d)})
        data = {"placeholder": True, "mode": "cube",
                "race": {"id": row["id"], "depth": row["depth"], "scramble": [[f, q] for f, q in mv],
                         "arms": {"base": base_lane, "e2": e2_lane}},
                "totals": {"base": {"solved": 0, "tokens": 1638400, "wall_s": 21000, "cap": 100},
                           "e2": {"solved": sum(b["e2"] for b in brd), "tokens": 412000, "wall_s": 5400, "cap": 12}},
                "board": brd}
    else:
        P = json.loads(Path(a.primary).read_text())
        rows = {r["id"]: r for r in map(json.loads, open(a.set, encoding="utf-8"))}
        data = {"placeholder": False, "mode": P["video_mode"], "race": None,
                "totals": {arm: {"solved": P[arm]["solved"], "tokens": P[arm]["total_tokens"],
                                 "wall_s": P[arm]["wall_s"], "cap": P[arm]["cap"]} for arm in ("base", "e2")},
                "board": board(rows, P["per_puzzle"])}
        rid = P["race_id"]
        if rid:
            arms = {}
            for arm in ("base", "e2"):
                lanes, rec = from_recording(tok, Path(a.race_dir) / f"{arm}_{rid}.json")
                lanes["matches_eval_text"] = rec.get("matches_eval_text")
                arms[arm] = lanes
            data["race"] = {"id": rid, "depth": rows[rid]["depth"],
                            "scramble": [[f, q] for f, q in parse_moves(rows[rid]["scramble"])], "arms": arms}
    Path(a.out).write_text("// Generated by ~/wsp-r1/a5_video_data.py. Do not edit by hand.\nwindow.CUBE = "
                           + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    r = data["race"]
    print(f"wrote {a.out}: mode {data['mode']}, placeholder {data['placeholder']}, race "
          f"{r['id'] if r else None}" + (f", base {r['arms']['base']['tokens']} tok {r['arms']['base']['wall_s']} s, "
          f"e2 {r['arms']['e2']['tokens']} tok {r['arms']['e2']['wall_s']} s" if r else ""))


if __name__ == "__main__":
    main()
