"""Side-by-side demo video: original model vs edited model on one problem.

  # 30 s LinkedIn cut (loops seamlessly)
  python make_demo_video.py --left results/base_math500_greedy.jsonl \
      --right results/edit_math500_greedy.jsonl --set math500 --id m500-17 \
      --patch-info work/edit/patch_info.json --out demo/m500-17.mp4

  # 60 s explainer for the write-up / README (adds "what is WSP" and random-control cards)
  python make_demo_video.py ... --variant explainer \
      --control "Random (same weights)=results/rand_w_math500_greedy.jsonl" \
      --control "Random (same size)=results/rand_s_math500_greedy.jsonl"

Honesty rules this script enforces:
  * every character on screen is the logged model response, decoded exactly; nothing is
    shortened, rewritten or extrapolated. Fast-forwarding skips time, never text, and the
    pane shows a speed tag with the real factor whenever it runs faster than normal;
  * token counters run to the logged n_tokens and stop there. "Never finishes" is only
    shown when the logged run hit the token cap; a run killed by the loop detector says so;
  * end-card numbers are computed from the two results files given (all shared problems);
  * the footer names the problem id and both results files.
Output: 1080x1350, 30 fps, H.264 (-preset veryfast), no audio. Pillow frames piped to ffmpeg.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import BASE_MODEL, PRACTICE_DIR, SEALED_DIR, last_boxed, read_jsonl

# ----------------------------------------------------------------------------- canvas
W, H, FPS = 1080, 1350, 30
MARGIN, GAP = 24, 24
PANE_W = (W - 2 * MARGIN - GAP) // 2          # 504
PANE_PAD = 16
CAPTION_Y, PROBLEM_Y, LABEL_Y, PANE_Y = 20, 122, 268, 310
PANE_H = 840
BAR_Y = PANE_Y + PANE_H + 12                  # 1162
FOOTER_Y = 1282
MONO_PX, LINE_H = 28, 37
COLS = 28                                     # max characters per pane line

BG = (13, 17, 23)
PANE_BG = (22, 27, 34)
BORDER = (48, 54, 61)
TEXT = (201, 209, 217)
DIM = (125, 133, 144)
WHITE = (240, 246, 252)
AMBER = (240, 165, 0)
AMBER_DIM_BG = (92, 62, 6)
AMBER_TEXT = (255, 214, 140)
GREEN = (63, 185, 80)
GREEN_BG = (35, 134, 54)
BLUE = (47, 129, 247)
DARK = (16, 16, 16)

# ----------------------------------------------------------------------------- timing
NORMAL_RATE = 12.0     # tokens / second at "1x"
T_FF = 6.0             # shared fast-forward may start here if the loop is far away
T_LOOP = 10.0          # left pane reaches its first repeat
T_LEFT_FAST = 18.0
T_RIGHT_DONE = 19.0
T_LEFT_DONE = 22.5
T_CARD = 23.0
T_MAIN_END = 28.0      # end card held until here
T_XFADE = 2.0          # short cut: cross-fade back to frame 0
MILESTONES = (8000, 12000, 16000)

# explainer variant (60 s)
EX_INTRO, EX_CONTROL, EX_METHOD, EX_CLOSE = 9.0, 9.0, 8.0, 6.0

LEFT_LABEL = "DeepSeek-R1 1.5B (original)"
RIGHT_LABEL = "After Claude's weight edit"


# ----------------------------------------------------------------------------- fonts
def _find(cands: list[str]) -> str | None:
    for c in cands:
        if Path(c).exists():
            return c
    return None


DJ = "/usr/share/fonts/truetype/dejavu/"
WINF = "/mnt/c/Windows/Fonts/"
FONT_MONO = _find([DJ + "DejaVuSansMono.ttf", WINF + "consola.ttf"])
FONT_MONO_B = _find([DJ + "DejaVuSansMono-Bold.ttf", WINF + "consolab.ttf"])
FONT_SANS = _find([DJ + "DejaVuSans.ttf", WINF + "segoeui.ttf"])
FONT_SANS_B = _find([DJ + "DejaVuSans-Bold.ttf", WINF + "segoeuib.ttf"])
FONT_CJK = _find(["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", WINF + "msyh.ttc"])

_cache: dict = {}


def font(path: str | None, px: int):
    key = (path, px)
    if key not in _cache:
        if path is None:
            raise SystemExit("no usable font found (install fonts-dejavu-core)")
        _cache[key] = ImageFont.truetype(path, px)
    return _cache[key]


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


# ----------------------------------------------------------------------------- data
def load_problem(set_name: str, pid: str) -> dict:
    if set_name.startswith("practice:"):
        path = PRACTICE_DIR / f"{set_name.split(':', 1)[1]}.jsonl"
    else:
        path = SEALED_DIR / f"{set_name}.jsonl"
    for r in read_jsonl(path):
        if r["id"] == pid:
            return r
    raise SystemExit(f"{pid} not in {path}")


def load_result(path: str, pid: str) -> dict:
    rows = [r for r in read_jsonl(Path(path)) if r["id"] == pid]
    if not rows:
        raise SystemExit(f"{pid} not in {path}")
    return min(rows, key=lambda r: r.get("sample", 0))   # greedy: one sample


def rates(path: str, ids: set[str] | None = None) -> dict:
    rows = [r for r in read_jsonl(Path(path)) if ids is None or r["id"] in ids]
    n = len(rows)
    return {"n": n, "acc": sum(r["correct"] for r in rows) / n,
            "loop": sum(r["loop"] for r in rows) / n, "ids": {r["id"] for r in rows}}


def clean(text: str) -> str:
    """Length-preserving: control characters (other than newline) become spaces."""
    return "".join(c if (c == "\n" or ord(c) >= 32) else " " for c in text)


class Stream:
    """One logged response: exact text, token -> char mapping, wrapped layout."""

    def __init__(self, tok, res: dict):
        self.res = res
        self.text = clean(res["response"])
        enc = tok(res["response"], add_special_tokens=False, return_offsets_mapping=True)
        self.ids = enc["input_ids"]
        ends, m = [], 0
        for _, e in enc["offset_mapping"]:
            m = max(m, e)
            ends.append(m)
        self.char_end = ends                   # chars visible after k+1 tokens
        self.n_retok = len(self.ids)
        self.n_logged = int(res["n_tokens"])
        self.lines = wrap(self.text, COLS)
        self.starts = [s for s, _ in self.lines]

    def chars_at(self, k: float) -> int:
        k = int(k)
        if k <= 0:
            return 0
        return self.char_end[min(k, self.n_retok) - 1]

    def counter_at(self, k: float) -> int:
        """Logged token count corresponding to retokenized position k (exact at the end)."""
        return round(min(k, self.n_retok) * self.n_logged / max(self.n_retok, 1))


def wrap(text: str, cols: int) -> list[tuple[int, int]]:
    """Greedy word wrap that keeps absolute char offsets. Wide (CJK) chars count as 2."""
    lines: list[tuple[int, int]] = []
    pos = 0
    for para in text.split("\n"):
        p0, p1 = pos, pos + len(para)
        ls, col, last_space = p0, 0, -1
        i = p0
        while i < p1:
            ch = text[i]
            col += 2 if is_wide(ch) else 1
            if ch == " ":
                last_space = i
            if col > cols:
                if last_space > ls:
                    lines.append((ls, last_space))
                    ls = last_space + 1
                else:
                    lines.append((ls, i))
                    ls = i
                col = sum(2 if is_wide(c) else 1 for c in text[ls:i + 1])
                last_space = -1
            i += 1
        lines.append((ls, p1))
        pos = p1 + 1
    return lines


def find_loop(ids: list[int], n: int = 20, window: int = 2048) -> tuple[int, int] | None:
    """(start, period) of the verbatim-periodic tail, or None.

    Period = most common distance between repeated 20-grams in the last `window` tokens;
    start = earliest token from which ids[i] == ids[i + period] holds to the end
    (tolerating runs of <= 3 mismatches from retokenization at copy boundaries).
    """
    if len(ids) < 4 * n:
        return None
    last: dict[tuple, int] = {}
    dists: Counter = Counter()
    lo = max(0, len(ids) - window)
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if g in last and i >= lo:
            dists[i - last[g]] += 1
        last[g] = i
    if not dists:
        return None
    P = dists.most_common(1)[0][0]
    S, miss = len(ids) - P, 0
    for i in range(len(ids) - P - 1, -1, -1):
        if ids[i] == ids[i + P]:
            S, miss = i, 0
        else:
            miss += 1
            if miss > 3:
                break
    if len(ids) - S < 2 * P:
        return None
    return S, P


# ----------------------------------------------------------------------------- schedules
def interp(keys: list[tuple[float, float]], t: float) -> float:
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / max(t1 - t0, 1e-9)
    return keys[-1][1]


def slope(keys: list[tuple[float, float]], t: float) -> float:
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            return (v1 - v0) / max(t1 - t0, 1e-9)
    return 0.0


def monotone(keys):
    out, m = [], 0.0
    for t, v in keys:
        m = max(m, v)
        out.append((t, m))
    return out


def plan_schedules(L: Stream, R: Stream, loop: tuple[int, int]) -> dict:
    S, P = loop
    onset = S + min(P, 200)                    # first repeat (capped for very long periods)
    nL, nR = L.n_retok, R.n_retok
    s0 = int(max(8, min(60, 0.3 * onset, 0.3 * nR)))
    if s0 + NORMAL_RATE * T_LOOP >= onset:
        shared = [(0.0, s0), (T_LOOP, onset)]
    else:
        shared = [(0.0, s0), (T_FF, s0 + NORMAL_RATE * T_FF), (T_LOOP, onset)]
    coupled = onset <= 0.8 * nR
    if coupled:
        right = shared + [(T_RIGHT_DONE, nR)]
    else:   # loop starts too late: right pane has to be slowed relative to the left after T_FF
        right = [(0.0, s0), (T_FF, min(s0 + NORMAL_RATE * T_FF, 0.4 * nR)),
                 (T_LOOP, 0.6 * nR), (T_RIGHT_DONE, nR)]
    mid = min(nL, onset + max(6 * P, NORMAL_RATE * 8 * 6))
    left = shared + [(T_LEFT_FAST, mid), (T_LEFT_DONE, nL)]
    left, right = monotone(left), monotone(right)

    def max_x(keys):
        return max((v1 - v0) / (t1 - t0) for (t0, v0), (t1, v1) in zip(keys, keys[1:])) / NORMAL_RATE

    weak = []
    if onset > 3000:
        weak.append(f"loop starts late (first repeat at token {onset})")
    if nR > 3500:
        weak.append(f"edited answer is long ({R.n_logged} tokens)")
    if not coupled:
        weak.append("edited answer ends before the loop starts; panes cannot share speed")
    if max(max_x(left), max_x(right)) > 80:
        weak.append(f"needs {max(max_x(left), max_x(right)):.0f}x fast-forward")
    return {"left": left, "right": right, "onset": onset, "loop_start": S, "period": P,
            "coupled": coupled, "max_x_left": max_x(left), "max_x_right": max_x(right),
            "weak": weak}


# ----------------------------------------------------------------------------- drawing
def text_w(f, s: str) -> float:
    return f.getlength(s)


def wrap_px(s: str, f, width: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        cand = (cur + " " + w_).strip()
        if text_w(f, cand) <= width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def fit(s: str, path: str, px: int, width: int):
    while px > 14 and text_w(font(path, px), s) > width:
        px -= 1
    return font(path, px)


def pill(d: ImageDraw.ImageDraw, x: int, y: int, label: str, bg, fg, px: int = 24, anchor_right=False):
    f = font(FONT_SANS_B, px)
    tw = int(text_w(f, label))
    w_, h_ = tw + 22, px + 14
    if anchor_right:
        x -= w_
    d.rounded_rectangle([x, y, x + w_, y + h_], radius=h_ // 2, fill=bg)
    d.text((x + 11, y + 6), label, font=f, fill=fg)


def blend(c1, c2, a: float):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


class Renderer:
    def __init__(self, a, L: Stream, R: Stream, problem: dict, sched: dict, stats: dict, pct: str):
        self.a, self.L, self.R, self.sched, self.stats = a, L, R, sched, stats
        self.cw = font(FONT_MONO, MONO_PX).getlength("M")
        self.n_lines = (PANE_H - 2 * PANE_PAD) // LINE_H
        self.loop_char = L.chars_at(sched["loop_start"])
        rt = R.text
        b = rt.rfind("\\boxed{")
        self.box = None
        if b >= 0 and last_boxed(rt) is not None:
            j, depth = rt.index("{", b) + 1, 1
            while j < len(rt) and depth:
                depth += {"{": 1, "}": -1}.get(rt[j], 0)
                j += 1
            self.box = (b, j)
        self.left_done_t = T_LEFT_DONE
        self.right_done_t = next((t / 100 for t in range(0, int(T_RIGHT_DONE * 100) + 1)
                                  if interp(sched["right"], t / 100) >= R.n_retok), T_RIGHT_DONE)
        self.static = self._static(problem, pct)
        self.watermark = self._watermark() if a.test else None

    # -- static layer: captions, problem, labels, pane frames, footer
    def _static(self, problem: dict, pct: str) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        cap1 = "Same model. Same question."
        cap2 = self.a.caption2 or f"One has {pct} of its weights edited."
        for i, s in enumerate((cap1, cap2)):
            f = fit(s, FONT_SANS_B, 40, W - 2 * MARGIN)
            d.text((W // 2, CAPTION_Y + i * 50), s, font=f, fill=WHITE, anchor="ma")
        # pinned problem
        d.rounded_rectangle([MARGIN, PROBLEM_Y, W - MARGIN, LABEL_Y - 14], radius=14, fill=PANE_BG,
                            outline=BORDER, width=2)
        fq = font(FONT_SANS, 28)
        q = " ".join(problem["question"].split())
        lines = wrap_px("Q: " + q, fq, W - 2 * MARGIN - 36)
        if len(lines) > 3:
            lines = lines[:3]
            while text_w(fq, lines[2] + " …") > W - 2 * MARGIN - 36:
                lines[2] = lines[2].rsplit(" ", 1)[0]
            lines[2] += " …"
        for i, s in enumerate(lines):
            d.text((MARGIN + 18, PROBLEM_Y + 12 + i * 36), s, font=fq, fill=TEXT)
        # labels + pane frames
        for k, label in enumerate((self.a.left_label, self.a.right_label)):
            x = MARGIN + k * (PANE_W + GAP)
            f = fit(label, FONT_SANS_B, 28, PANE_W - 8)
            d.text((x + PANE_W // 2, LABEL_Y), label, font=f, fill=WHITE, anchor="ma")
            d.rounded_rectangle([x, PANE_Y, x + PANE_W, PANE_Y + PANE_H], radius=14, fill=PANE_BG,
                                outline=BORDER, width=2)
        # footer
        ff = font(FONT_SANS, 18)
        foot1 = f"Problem {problem['id']} · greedy decoding · every character is the logged model output"
        foot2 = f"Left: {Path(self.a.left).name}   Right: {Path(self.a.right).name}"
        if self.a.test:
            foot1 = "TEST RENDER on practice data · " + foot1
        for i, s in enumerate((foot1, foot2)):
            f = fit(s, FONT_SANS, 18, W - 2 * MARGIN)
            d.text((W // 2, FOOTER_Y + i * 26), s, font=f, fill=DIM, anchor="ma")
        return im

    def _watermark(self) -> Image.Image:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        txt = Image.new("RGBA", (1400, 140), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((700, 70), "TEST RENDER · PRACTICE DATA", font=font(FONT_SANS_B, 72),
                                 fill=(255, 255, 255, 46), anchor="mm")
        txt = txt.rotate(51, expand=True)
        layer.alpha_composite(txt, ((W - txt.width) // 2, (H - txt.height) // 2))
        return layer

    # -- one pane
    def _pane(self, d, x0: int, s: Stream, k: float, t: float, side: str):
        n_chars = s.chars_at(k)
        streaming = k < s.n_retok
        if n_chars == 0:
            return
        li = bisect.bisect_right(s.starts, n_chars) - 1
        # skip a trailing empty line created right at a newline we haven't "typed" past
        first = max(0, li - self.n_lines + 1)
        mono, mono_b = font(FONT_MONO, MONO_PX), font(FONT_MONO_B, MONO_PX)
        loop_on = side == "L" and t >= T_LOOP
        flash = 0.5 + 0.5 * math.cos(2 * math.pi * 1.5 * (t - T_LOOP)) if T_LOOP <= t < T_LEFT_FAST else 0.0
        y = PANE_Y + PANE_PAD
        cur_xy = None
        for j in range(first, li + 1):
            a_, b_ = s.lines[j]
            b_ = min(b_, n_chars)
            x = x0 + PANE_PAD
            col = 0
            i = a_
            while i < b_:
                # group a run of same style
                st = self._style(side, i, loop_on, flash)
                wide = is_wide(s.text[i])
                j2 = i + 1
                while j2 < b_ and self._style(side, j2, loop_on, flash) == st and is_wide(s.text[j2]) == wide:
                    j2 += 1
                run = s.text[i:j2]
                cols = sum(2 if is_wide(c) else 1 for c in run)
                fg, bg, bold = st
                xx = x + col * self.cw
                if bg is not None:
                    d.rectangle([xx, y - 3, xx + cols * self.cw, y + LINE_H - 5], fill=bg)
                if wide and FONT_CJK:
                    fc = font(FONT_CJK, MONO_PX)
                    for q, ch in enumerate(run):
                        d.text((xx + 2 * q * self.cw, y), ch, font=fc, fill=fg)
                else:
                    d.text((xx, y), run, font=mono_b if bold else mono, fill=fg)
                col += cols
                i = j2
            cur_xy = (x + col * self.cw, y)
            y += LINE_H
        if streaming and cur_xy and int(t * 4) % 2 == 0:
            cx, cy = cur_xy
            d.rectangle([cx + 2, cy + 2, cx + self.cw - 2, cy + LINE_H - 8], fill=DIM)

    def _style(self, side: str, i: int, loop_on: bool, flash: float):
        if side == "L" and loop_on and i >= self.loop_char:
            if flash > 0:
                return (blend(AMBER_TEXT, DARK, flash), blend(AMBER_DIM_BG, AMBER, flash), True)
            return (AMBER_TEXT, AMBER_DIM_BG, False)
        if side == "R" and self.box and self.box[0] <= i < self.box[1]:
            return (WHITE, GREEN_BG, True)
        return (TEXT, None, False)

    # -- bottom bar
    def _bar(self, d, t: float, kL: float, kR: float):
        L, R, st = self.L, self.R, self.stats
        for side, x, s, k in (("L", MARGIN, L, kL), ("R", MARGIN + PANE_W + GAP, R, kR)):
            n = s.counter_at(k)
            color = WHITE
            if side == "L":
                crossed = [m for m in MILESTONES if m <= n]
                if crossed:
                    # time since crossing the latest milestone -> brief amber flash
                    tc = self._cross_time(crossed[-1])
                    if tc is not None and 0 <= t - tc < 0.6:
                        color = AMBER
            d.text((x + 4, BAR_Y), f"{n:,}", font=font(FONT_MONO_B, 46), fill=color)
            d.text((x + 4 + font(FONT_MONO_B, 46).getlength(f"{n:,}") + 10, BAR_Y + 18), "tokens",
                   font=font(FONT_SANS, 24), fill=DIM)
            sy = BAR_Y + 60
            if side == "L":
                # milestone chips for milestones the logged run actually passed
                cx = x + PANE_W
                for m in reversed([m for m in MILESTONES if m <= L.n_logged]):
                    lab = f"{m // 1000}k"
                    on = n >= m
                    f = font(FONT_SANS_B, 20)
                    w_ = int(text_w(f, lab)) + 16
                    cx -= w_ + 6
                    d.rounded_rectangle([cx, BAR_Y + 10, cx + w_, BAR_Y + 40], radius=8,
                                        fill=AMBER if on else PANE_BG, outline=BORDER)
                    d.text((cx + 8, BAR_Y + 14), lab, font=f, fill=DARK if on else DIM)
                if t >= self.left_done_t:
                    status = L.res["status"]
                    if status == "cap":
                        big, small = "Never finishes", f"hit the {L.n_logged:,}-token cap, still looping"
                    elif status == "loop":
                        big, small = "Stuck in a loop", f"run stopped by the loop detector at {L.n_logged:,}"
                    else:
                        big, small = "Wrong answer" if not L.res["correct"] else "Finished", ""
                    d.text((x + 4, sy), big, font=font(FONT_SANS_B, 30), fill=AMBER)
                    if small:
                        d.text((x + 4, sy + 38), small, font=fit(small, FONT_SANS, 20, PANE_W), fill=DIM)
                elif t >= T_LOOP:
                    d.text((x + 4, sy), "Repeating itself…", font=font(FONT_SANS_B, 28), fill=AMBER)
            else:
                if t >= self.right_done_t:
                    if R.res["correct"]:
                        msg, c = f"Solved in {R.n_logged:,} tokens", GREEN
                    else:
                        msg, c = f"Finished, wrong ({R.n_logged:,} tokens)", DIM
                    d.text((x + 4, sy), msg, font=fit(msg, FONT_SANS_B, 30, PANE_W), fill=c)
                else:
                    d.text((x + 4, sy), "Reasoning…", font=font(FONT_SANS, 28), fill=DIM)

    def _cross_time(self, m: int) -> float | None:
        keys = self.sched["left"]
        target = m * self.L.n_retok / max(self.L.n_logged, 1)
        for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
            if v0 <= target <= v1 and v1 > v0:
                return t0 + (target - v0) / (v1 - v0) * (t1 - t0)
        return None

    # -- main timeline frame
    def main_frame(self, t: float) -> Image.Image:
        im = self.static.copy()
        d = ImageDraw.Draw(im)
        kL, kR = interp(self.sched["left"], t), interp(self.sched["right"], t)
        self._pane(d, MARGIN, self.L, kL, t, "L")
        self._pane(d, MARGIN + PANE_W + GAP, self.R, kR, t, "R")
        # badges
        for x0, keys, k, s in ((MARGIN, self.sched["left"], kL, self.L),
                               (MARGIN + PANE_W + GAP, self.sched["right"], kR, self.R)):
            x_rate = slope(keys, t) / NORMAL_RATE
            if k < s.n_retok and x_rate >= 1.5:
                pill(d, x0 + 10, PANE_Y + 10, f"▶▶ {round(x_rate)}x", BLUE, WHITE)
        if t >= T_LOOP:
            pulse = 0.5 + 0.5 * math.cos(2 * math.pi * 1.5 * (t - T_LOOP)) if t < T_LEFT_FAST else 1.0
            pill(d, MARGIN + PANE_W - 10, PANE_Y + 10, "↻ LOOP", blend(AMBER_DIM_BG, AMBER, pulse), DARK,
                 px=26, anchor_right=True)
        self._bar(d, t, kL, kR)
        if t >= T_CARD:
            im = self._end_card(im, min(1.0, (t - T_CARD) / 0.4))
        return im

    def _end_card(self, im: Image.Image, a: float) -> Image.Image:
        over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        d.rectangle([0, PANE_Y - 50, W, BAR_Y + 110], fill=(8, 10, 14, int(225 * a)))
        st = self.stats
        y = PANE_Y + 200
        f = font(FONT_SANS_B, 56)
        al = int(255 * a)
        for label, before, after in (("Loop rate", st["base_loop"], st["edit_loop"]),
                                     ("Accuracy", st["base_acc"], st["edit_acc"])):
            parts = [(label + "  ", WHITE), (f"{100 * before:.1f}%", AMBER), ("  →  ", DIM),
                     (f"{100 * after:.1f}%", GREEN)]
            total = sum(text_w(f, p) for p, _ in parts)
            x = (W - total) / 2
            for p, c in parts:
                d.text((x, y), p, font=f, fill=c + (al,))
                x += text_w(f, p)
            y += 110
        s3 = "No training. Laptop RTX 4050."
        d.text((W // 2, y + 20), s3, font=fit(s3, FONT_SANS_B, 48, W - 80), fill=WHITE + (al,), anchor="ma")
        out = im.convert("RGBA")
        out.alpha_composite(over)
        return out.convert("RGB")

    # -- explainer cards
    def card(self, title: str, body: list[str], table: list[list[str]] | None = None,
             foot: str | None = None) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = 230
        ft = font(FONT_SANS_B, 54)
        for s in wrap_px(title, ft, W - 120):
            d.text((W // 2, y), s, font=ft, fill=WHITE, anchor="ma")
            y += 70
        y += 40
        fb = font(FONT_SANS, 38)
        for para in body:
            for s in wrap_px(para, fb, W - 140):
                d.text((70, y), s, font=fb, fill=TEXT)
                y += 52
            y += 26
        if table:
            fh, fr = font(FONT_SANS_B, 32), font(FONT_SANS, 32)
            cols_x = [70, 600, 830]
            for r_i, row in enumerate(table):
                f = fh if r_i == 0 else fr
                for c_i, cell in enumerate(row):
                    color = DIM if r_i == 0 else (WHITE if c_i == 0 else TEXT)
                    d.text((cols_x[c_i], y), cell, font=f, fill=color)
                y += 56
                if r_i == 0:
                    d.line([70, y - 8, W - 70, y - 8], fill=BORDER, width=2)
        if foot:
            ff = font(FONT_SANS, 22)
            for i, s in enumerate(wrap_px(foot, ff, W - 120)):
                d.text((W // 2, FOOTER_Y - 30 + i * 28), s, font=ff, fill=DIM, anchor="ma")
        return im

    def finish(self, im: Image.Image) -> Image.Image:
        if self.watermark is None:
            return im
        out = im.convert("RGBA")
        out.alpha_composite(self.watermark)
        return out.convert("RGB")


# ----------------------------------------------------------------------------- timelines
def fade(a: Image.Image, b: Image.Image, x: float) -> Image.Image:
    return Image.blend(a, b, max(0.0, min(1.0, x)))


def frames_short(r: Renderer):
    first = r.main_frame(0.0)
    last = r.main_frame(T_MAIN_END)
    n_main = int(T_MAIN_END * FPS)
    for i in range(n_main):
        yield i / FPS, (first if i == 0 else r.main_frame(i / FPS))
    n_x = int(T_XFADE * FPS)
    for i in range(n_x):   # ends one frame before frame 0, so the replay is seamless
        yield T_MAIN_END + i / FPS, fade(last, first, (i + 1) / n_x)


def frames_explainer(r: Renderer, cards: dict):
    intro, ctrl, meth, close = cards["intro"], cards["control"], cards["method"], cards["close"]
    black = Image.new("RGB", (W, H), BG)
    t0 = 0.0
    seq = []
    # intro card with fades
    for i in range(int(EX_INTRO * FPS)):
        t = i / FPS
        im = fade(black, intro, t / 0.5) if t < 0.5 else intro
        if t > EX_INTRO - 0.5:
            im = fade(intro, r.main_frame(0.0), (t - (EX_INTRO - 0.5)) / 0.5)
        seq.append((t0 + t, lambda im=im: im))
    t0 += EX_INTRO
    for i in range(int(T_MAIN_END * FPS)):
        tt = i / FPS
        seq.append((t0 + tt, lambda tt=tt: r.main_frame(tt)))
    t0 += T_MAIN_END
    end_main = r.main_frame(T_MAIN_END)
    for card, dur, nxt in ((ctrl, EX_CONTROL, meth), (meth, EX_METHOD, close), (close, EX_CLOSE, None)):
        prev = end_main if card is ctrl else None
        for i in range(int(dur * FPS)):
            tt = i / FPS
            im = card
            if prev is not None and tt < 0.5:
                im = fade(prev, card, tt / 0.5)
            if nxt is not None and tt > dur - 0.5:
                im = fade(card, nxt, (tt - (dur - 0.5)) / 0.5)
            seq.append((t0 + tt, lambda im=im: im))
        t0 += dur
    for t, f in seq:
        yield t, f()


def explainer_cards(r: Renderer, a, pct: str) -> dict:
    st = r.stats
    intro = r.card(
        "What is weight-space programming?",
        ["Instead of training a model, an AI reads its weights and rewrites a tiny part of them "
         "directly: no gradients, no fine-tuning, just analysis and a targeted edit.",
         "Here, Claude was asked to make DeepSeek-R1 1.5B better at math this way, with a budget "
         "of 50 model runs on a laptop GPU.",
         "Watch the same problem on the original and the edited model."],
        foot="Inspired by MIT CSAIL, \"Sparks of Weight-Space Programming\" (Nov 2026)")
    table = [["Arm", "Accuracy", "Loop rate"],
             ["Original", f"{100 * st['base_acc']:.1f}%", f"{100 * st['base_loop']:.1f}%"],
             ["Claude's edit", f"{100 * st['edit_acc']:.1f}%", f"{100 * st['edit_loop']:.1f}%"]]
    for label, c in st["controls"]:
        table.append([label, f"{100 * c['acc']:.1f}%", f"{100 * c['loop']:.1f}%"])
    if not st["controls"]:
        table.append(["Random edit", "TBD", "TBD"])
    control = r.card(
        "Is it the edit, or would any edit do?",
        ["Control: change the same number of weights by the same amount, in random directions."],
        table=table, foot=f"Same {st['n']} problems for every arm · greedy decoding · files in the repo")
    method = r.card(
        "How it was tested",
        ["• Evaluation plan committed before the edit was made",
         "• Test problems sealed away from Claude while it worked",
         f"• Paired comparison on the same {st['n']} problems",
         "• Random-edit controls and 95% bootstrap CIs",
         "• Code, logs and raw outputs public"])
    close = r.card(
        f"No training. {pct} of weights changed. Laptop RTX 4050.",
        [a.url] if a.url else [], foot="Write-up, model and code: links in the post")
    return {"intro": intro, "control": control, "method": method, "close": close}


# ----------------------------------------------------------------------------- main
def fmt_pct(p: float) -> str:
    return f"{p:.2g}%" if p < 1 else f"{p:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="results file, original model")
    ap.add_argument("--right", required=True, help="results file, edited model")
    ap.add_argument("--id", required=True, help="problem id")
    ap.add_argument("--set", required=True, help="math500 | practice:math_train ... (for the problem text)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["short", "explainer"], default="short")
    ap.add_argument("--patch-info", default=None, help="work/edit/patch_info.json (for 0.0Z%%)")
    ap.add_argument("--control", action="append", default=[], help='"Label=results/rand_w_....jsonl"')
    ap.add_argument("--left-label", default=LEFT_LABEL)
    ap.add_argument("--right-label", default=RIGHT_LABEL)
    ap.add_argument("--caption2", default=None, help="override caption line 2 (test renders)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--test", action="store_true", help="watermark: practice-data test render")
    ap.add_argument("--stills", default=None, help="also save PNG stills at key times into this dir")
    ap.add_argument("--diag-only", action="store_true", help="print the schedule/weak check, no render")
    a = ap.parse_args()

    if not a.test and a.set.startswith("practice:") and a.right_label == RIGHT_LABEL and not a.patch_info:
        raise SystemExit("practice data without --patch-info: use --test (watermarked) for test renders")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    problem = load_problem(a.set, a.id)
    L, R = Stream(tok, load_result(a.left, a.id)), Stream(tok, load_result(a.right, a.id))
    loop = find_loop(L.ids)
    if loop is None:
        raise SystemExit(f"{a.id}: no verbatim loop found in the left response; pick another problem")
    sched = plan_schedules(L, R, loop)

    if a.patch_info:
        pct = fmt_pct(json.loads(Path(a.patch_info).read_text())["changed_pct"])
    else:
        pct = "0.0Z%"
    ids = rates(a.left)["ids"] & rates(a.right)["ids"]
    bl, ed = rates(a.left, ids), rates(a.right, ids)
    controls = []
    for c in a.control:
        label, path = c.split("=", 1)
        controls.append((label, rates(path, ids)))
    stats = {"n": len(ids), "base_acc": bl["acc"], "base_loop": bl["loop"], "edit_acc": ed["acc"],
             "edit_loop": ed["loop"], "controls": controls}

    diag = {"id": a.id, "left": {"status": L.res["status"], "correct": L.res["correct"],
                                 "n_tokens": L.n_logged, "retokenized": L.n_retok},
            "right": {"status": R.res["status"], "correct": R.res["correct"], "n_tokens": R.n_logged,
                      "retokenized": R.n_retok},
            "loop_start": sched["loop_start"], "period": sched["period"], "first_repeat": sched["onset"],
            "panes_share_speed_to_10s": sched["coupled"],
            "max_speed_left": round(sched["max_x_left"], 1), "max_speed_right": round(sched["max_x_right"], 1),
            "end_card": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()
                         if k != "controls"},
            "weak": sched["weak"]}
    print(json.dumps(diag, indent=2))
    for w_ in sched["weak"]:
        print(f"weak: {w_}")
    if not R.res["correct"]:
        print("warning: the right-hand answer is graded WRONG; no green 'Solved' will be shown")
    if L.res["correct"]:
        print("warning: the left-hand answer is graded correct; this is not a fixed problem")
    if a.diag_only:
        return

    r = Renderer(a, L, R, problem, sched, stats, pct)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps({**diag, "args": vars(a)}, indent=2))

    if a.variant == "short":
        frames = frames_short(r)
        stills_t = [0.0, 1.0, 5.0, 9.0, 11.0, 14.0, 17.0, 20.0, 22.8, 25.0, 29.0]
    else:
        frames = frames_explainer(r, explainer_cards(r, a, pct))
        stills_t = [3.0, 12.0, 20.0, 29.0, 34.0, 40.0, 50.0, 57.0]

    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sdir = Path(a.stills) if a.stills else None
    if sdir:
        sdir.mkdir(parents=True, exist_ok=True)
    want = {round(s * FPS) for s in stills_t}
    n = 0
    for t, im in frames:
        im = r.finish(im)
        proc.stdin.write(im.tobytes())
        if sdir and n in want:
            im.save(sdir / f"{out.stem}_t{t:05.1f}.png")
        n += 1
        if n % 150 == 0:
            print(f"  {n / FPS:.0f}s rendered", file=sys.stderr)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out}  ({n} frames, {n / FPS:.1f} s)")


if __name__ == "__main__":
    main()
