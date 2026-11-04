"""Side-by-side demo video: original model vs edited model on one problem (LinkedIn, landscape).

  # 30 s LinkedIn cut (loops seamlessly)
  python make_demo_video.py --left results/base_math500_greedy.jsonl \
      --right results/edit_math500_greedy.jsonl --set math500 --id m500-17 \
      --patch-info work/edit/patch_info.json --out demo/m500-17.mp4

  # 60 s explainer for the write-up / README (adds "what is WSP" and random-control cards)
  python make_demo_video.py ... --variant explainer \
      --control "Random (same weights)=results/rand_w_math500_greedy.jsonl" \
      --control "Random (same size)=results/rand_s_math500_greedy.jsonl"

  # standalone GIF of the bottom "what is happening" strip (README / blog)
  python make_demo_video.py ... --gif demo/strip.gif

Design: light neobrutalism (cream paper, thick black rules, heavy condensed caps, one electric-blue
block, hard offset shadows, red hand-written notes). 1920x1080, 30 fps, H.264, no audio.

Honesty rules this script enforces:
  * every character in the two panes is the logged model response, decoded exactly; nothing is
    shortened, rewritten or extrapolated. Fast-forwarding skips time, never text, and each pane
    shows a speed tag with its real factor whenever it runs faster than normal;
  * token counters run to the logged n_tokens and stop there. "Never finishes" is only shown when
    the logged run hit the token cap; a run killed by the loop detector says so;
  * end-card numbers are computed from the two results files given (all shared problems);
  * the bottom strip is labelled ILLUSTRATION. The layers it lights up are read from the real
    patch (patch_info.json per_tensor); its loop snippet and boxed answer are the real logged text;
  * the footer names the problem id and both results files.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import BASE_MODEL, PRACTICE_DIR, SEALED_DIR, last_boxed, read_jsonl

# ----------------------------------------------------------------------------- canvas
W, H, FPS = 1920, 1080, 30
M = 40                                        # side margin
MAST = (24, 136)                              # masthead y-range
PROB = (150, 226)                             # pinned problem strip
PANE_Y0, PANE_Y1 = 244, 740                   # panes incl. header
HEADER_H = 60
PANE_W, GAP = 900, 40
PANE_X = (M, M + PANE_W + GAP)                # 40, 980
STATUS_Y = 752                                # status row under each pane
STRIP = (812, 1032)                           # bottom illustration strip
FOOTER_Y = 1046
SHADOW = 8
MONO_PX, LINE_H = 46, 56                      # >= 44 px: landscape is shown ~0.6x of portrait
COLS = 30                                     # max characters per pane line
PANE_PAD = 20

PAPER = (244, 241, 234)
WHITE = (255, 255, 255)
INK = (17, 17, 17)
DIM = (110, 106, 98)
RULE = (201, 196, 184)
BLUE = (31, 59, 255)
RED = (235, 51, 36)
RED_DARK = (150, 24, 14)
YELLOW = (255, 224, 138)
ORANGE = (255, 159, 28)
GREEN = (34, 197, 94)
GREEN_DARK = (11, 120, 52)

# ----------------------------------------------------------------------------- timing
NORMAL_RATE = 12.0     # tokens / second at "1x"
T_FF = 6.0             # panes share speed until here at most
T_LOOP = 10.0          # left pane reaches the loop onset
T_LEFT_FAST = 18.0
T_RIGHT_DONE = 19.0
T_LEFT_DONE = 22.5
T_CARD = 23.0
T_MAIN_END = 28.0      # end card held until here
T_XFADE = 2.0          # short cut: cross-fade back to frame 0
MILESTONES = (8000, 12000, 16000)
CYCLE = 7.5            # bottom strip animation period (30 s / 4: the replay stays seamless)

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
UB = "/usr/share/fonts/truetype/ubuntu/"
WINF = "/mnt/c/Windows/Fonts/"
F_DISPLAY = _find([WINF + "impact.ttf", UB + "Ubuntu-C.ttf", DJ + "DejaVuSans-Bold.ttf"])
F_SANS = _find([WINF + "arial.ttf", UB + "Ubuntu-R.ttf", DJ + "DejaVuSans.ttf"])
F_SANS_B = _find([WINF + "arialbd.ttf", UB + "Ubuntu-B.ttf", DJ + "DejaVuSans-Bold.ttf"])
F_MONO = _find([WINF + "consola.ttf", DJ + "DejaVuSansMono.ttf"])
F_MONO_B = _find([WINF + "consolab.ttf", DJ + "DejaVuSansMono-Bold.ttf"])
F_HAND = _find([WINF + "Inkfree.ttf", WINF + "segoepr.ttf", UB + "Ubuntu-MI.ttf"])
F_SYM = _find([DJ + "DejaVuSans-Bold.ttf", F_SANS_B or ""])          # ▶ ↻ → glyphs
F_CJK = _find(["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", WINF + "msyh.ttc"])

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


F_MONO_FB = _find([DJ + "DejaVuSansMono.ttf", DJ + "DejaVuSans.ttf"])
_notdef: dict = {}


def has_glyph(path: str, px: int, ch: str) -> bool:
    """False if the font would draw its .notdef box for ch (compares against a private-use char)."""
    if ch.isspace():
        return True
    f = font(path, px)
    key = (path, px)
    if key not in _notdef:
        m = f.getmask("\U000F0000")
        _notdef[key] = (m.size, bytes(m))
    m = f.getmask(ch)
    return (m.size, bytes(m)) != _notdef[key]


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


def find_loop(ids: list[int], n: int = 20, frac: float = 0.6, win: int = 512,
              min_len: int = 400) -> tuple[int, int, list[bool]] | None:
    """Loop region by the same notion as common.is_looping: repeated verbatim 20-grams.

    R1 loops are usually near-periodic (the same paragraphs cycling with small variants),
    not exactly periodic, so: rep[i] = the 20-gram starting at token i already occurred
    earlier in this answer. Onset = earliest token i such that both the next `win` tokens
    and the whole rest of the answer have >= `frac` repeated positions. Returns
    (onset, typical repeat distance, token mask of repeated spans from the onset on), or None.
    """
    m = len(ids) - n + 1
    if m < min_len:
        return None
    seen: dict[tuple, int] = {}
    rep = [0] * m
    dist = [0] * m
    for i in range(m):
        g = tuple(ids[i:i + n])
        if g in seen:
            rep[i], dist[i] = 1, i - seen[g]
        seen[g] = i
    pre = [0]
    for v in rep:
        pre.append(pre[-1] + v)
    onset = None
    for i in range(m - min_len + 1):
        w_end = min(m, i + win)
        if (pre[w_end] - pre[i]) >= frac * (w_end - i) and (pre[m] - pre[i]) >= frac * (m - i):
            onset = i
            break
    if onset is None:
        return None
    while onset < m and not rep[onset]:
        onset += 1
    d = Counter(dist[i] for i in range(onset, m) if rep[i])
    P = d.most_common(1)[0][0] if d else 200
    mask = [False] * len(ids)
    for i in range(onset, m):
        if rep[i]:
            for j in range(i, i + n):
                mask[j] = True
    return onset, P, mask


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


def plan_schedules(L: Stream, R: Stream, loop: tuple[int, int, list[bool]]) -> dict:
    S, P, _ = loop
    onset = S                                  # first token of the dense-repetition region
    nL, nR = L.n_retok, R.n_retok
    s0 = int(max(8, min(60, 0.3 * onset, 0.3 * nR)))
    if s0 + NORMAL_RATE * T_LOOP >= onset:
        shared = [(0.0, s0), (T_LOOP, onset)]
    else:
        shared = [(0.0, s0), (T_FF, s0 + NORMAL_RATE * T_FF), (T_LOOP, onset)]
    coupled = onset <= 0.8 * nR
    if coupled:
        right = shared + [(T_RIGHT_DONE, nR)]
    else:   # loop starts after the edited answer is nearly done: panes diverge after T_FF
        right = [(0.0, s0), (T_FF, min(s0 + NORMAL_RATE * T_FF, 0.4 * nR)),
                 (T_LOOP, 0.6 * nR), (T_RIGHT_DONE, nR)]
    mid = min(nL, onset + max(6 * P, NORMAL_RATE * 8 * 6))
    left = shared + [(T_LEFT_FAST, mid), (T_LEFT_DONE, nL)]
    left, right = monotone(left), monotone(right)

    def max_x(keys):
        return max((v1 - v0) / (t1 - t0) for (t0, v0), (t1, v1) in zip(keys, keys[1:])) / NORMAL_RATE

    # Pilot (4 Nov): R1 loops typically start after 3-8k tokens and fixed answers are 2-6k,
    # so the panes usually share speed only for the first seconds and then fast-forward
    # independently (each pane shows its own real factor). Only extremes count as weak.
    weak = []
    if onset > 8000:
        weak.append(f"loop starts very late (token {onset})")
    if nR > 6000:
        weak.append(f"edited answer is very long ({R.n_logged} tokens)")
    if max(max_x(left), max_x(right)) > 300:
        weak.append(f"needs {max(max_x(left), max_x(right)):.0f}x fast-forward")
    return {"left": left, "right": right, "onset": onset, "loop_start": S, "period": P,
            "coupled": coupled, "max_x_left": max_x(left), "max_x_right": max_x(right),
            "weak": weak}


def base_weight_count() -> int:
    """Exact parameter count of the base model, from the safetensors headers (no tensor load)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    n = 0
    for f in sorted(Path(snapshot_download(BASE_MODEL)).glob("model*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                n += math.prod(sf.get_slice(k).get_shape())
    return n


def edited_layers(patch_info: dict | None) -> tuple[list[int], list[str]]:
    """Layer indices and a short description of the tensors the real patch changed."""
    if not patch_info or not patch_info.get("per_tensor"):
        return [], []
    per = patch_info["per_tensor"]
    layers = sorted({int(m.group(1)) for k in per if (m := re.search(r"layers\.(\d+)\.", k))})
    desc = []
    for k, n in sorted(per.items(), key=lambda kv: -kv[1])[:2]:
        short = re.sub(r"^model\.", "", k).replace("layers.", "L").replace(".weight", "")
        desc.append(f"{short} ({n:,} weights)")
    return layers, desc


# ----------------------------------------------------------------------------- drawing helpers
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
    while px > 12 and text_w(font(path, px), s) > width:
        px -= 1
    return font(path, px)


def blend(c1, c2, a: float):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def brut_box(d: ImageDraw.ImageDraw, box, fill=WHITE, border=4, shadow=SHADOW):
    x0, y0, x1, y1 = box
    if shadow:
        d.rectangle([x0 + shadow, y0 + shadow, x1 + shadow, y1 + shadow], fill=INK)
    d.rectangle([x0, y0, x1, y1], fill=fill, outline=INK, width=border)


def chip(d, x, y, label, fill, fg, path=None, px=28, right=False, border=3):
    f = font(path or F_SYM, px)
    w_ = int(text_w(f, label)) + 24
    h_ = px + 16
    if right:
        x -= w_
    d.rectangle([x, y, x + w_, y + h_], fill=fill, outline=INK, width=border)
    d.text((x + 12, y + 7), label, font=f, fill=fg)
    return w_


def mixed(d, x, y, parts, anchor_center_w=None):
    """Draw [(text, font, color)] in one row; returns width."""
    total = sum(text_w(f, s) for s, f, _ in parts)
    if anchor_center_w is not None:
        x = x + (anchor_center_w - total) / 2
    for s, f, c in parts:
        d.text((x, y), s, font=f, fill=c)
        x += text_w(f, s)
    return total


# pixel character (original design, not a logo): X = body, O = eye
CHAR = ["..XXXXXX..",
        ".XXXXXXXX.",
        "XXOXXXXOXX",
        "XXXXXXXXXX",
        ".XXXXXXXX.",
        ".X.X..X.X."]
CHAR_COLOR = (224, 122, 79)


class Renderer:
    def __init__(self, a, L: Stream, R: Stream, problem: dict, sched: dict, stats: dict, pct: str,
                 patch_info: dict | None):
        self.a, self.L, self.R, self.sched, self.stats, self.pct = a, L, R, sched, stats, pct
        self.cw = font(F_MONO, MONO_PX).getlength("M")
        self._kinds: dict[str, int] = {}
        self.n_lines = (PANE_Y1 - PANE_Y0 - HEADER_H - 2 * PANE_PAD) // LINE_H
        # char mask of the repeated spans (highlighted), from the token mask
        self.loop_mask = bytearray(len(L.text))
        prev = 0
        for k, on in enumerate(sched["loop_mask"]):
            end = L.char_end[k]
            if on:
                self.loop_mask[prev:end] = b"\x01" * (end - prev)
            prev = end
        rt = R.text
        b = rt.rfind("\\boxed{")
        self.box = None
        if b >= 0 and last_boxed(rt) is not None:
            j, depth = rt.index("{", b) + 1, 1
            while j < len(rt) and depth:
                depth += {"{": 1, "}": -1}.get(rt[j], 0)
                j += 1
            self.box = (b, j)
        self.answer = rt[self.box[0]:self.box[1]] if self.box else None
        # real loop snippet for the strip: the first repeated span after the onset
        lc = L.chars_at(sched["onset"])
        snippet = " ".join(L.text[lc:lc + 120].split())
        self.loop_snippet = snippet[:34].rsplit(" ", 1)[0] + " …" if len(snippet) > 34 else snippet
        self.layers, self.layer_desc = edited_layers(patch_info)
        self.total_weights = base_weight_count()
        self.right_done_t = next((t / 100 for t in range(0, int(T_RIGHT_DONE * 100) + 1)
                                  if interp(sched["right"], t / 100) >= R.n_retok), T_RIGHT_DONE)
        self.static = self._static(problem)
        self.net_static, self.net_nodes = self._network_static()
        self.watermark = self._watermark() if a.test else None

    # -- static layer: masthead, problem, pane frames, strip frame, footer
    def _static(self, problem: dict) -> Image.Image:
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        y0, y1 = MAST
        # masthead: [title | mono stack | blue block]
        d.rectangle([M, y0, W - M, y1], fill=PAPER, outline=INK, width=5)
        cA, cB = 1060, 1380
        d.line([cA, y0, cA, y1], fill=INK, width=5)
        d.line([cB, y0, cB, y1], fill=INK, width=5)
        title = "SAME MODEL. SAME QUESTION."
        ft = fit(title, F_DISPLAY, 84, cA - M - 40)
        d.text((M + 22, (y0 + y1) // 2), title, font=ft, fill=INK, anchor="lm")
        fm = font(F_MONO_B, 22)
        for i, s in enumerate(("NO TRAINING", "NO GRADIENTS", "LAPTOP RTX 4050")):
            d.text((cA + 22, y0 + 18 + i * 30), s, font=fm, fill=INK)
        d.rectangle([cB + 3, y0 + 3, W - M - 3, y1 - 3], fill=BLUE)
        cap = (self.a.caption2 or f"One has {self.pct} of its weights edited.").upper()
        fc = font(F_DISPLAY, 40)
        lines = wrap_px(cap, fc, W - M - cB - 40)[:2]
        for i, s in enumerate(lines):
            d.text((cB + 22, y0 + 14 + i * 46 + (23 if len(lines) == 1 else 0)), s, font=fc, fill=WHITE)
        # pinned problem
        py0, py1 = PROB
        d.rectangle([M, py0, W - M, py1], fill=WHITE, outline=INK, width=4)
        d.rectangle([M, py0, M + 170, py1], fill=INK)
        d.text((M + 85, (py0 + py1) // 2), "PROBLEM", font=font(F_DISPLAY, 34), fill=WHITE, anchor="mm")
        fq = font(F_SANS, 29)
        q = " ".join(problem["question"].split())
        lines = wrap_px(q, fq, W - 2 * M - 210)
        if len(lines) > 2:
            lines = lines[:2]
            while text_w(fq, lines[1] + " …") > W - 2 * M - 210:
                lines[1] = lines[1].rsplit(" ", 1)[0]
            lines[1] += " …"
        for i, s in enumerate(lines):
            d.text((M + 192, py0 + 6 + i * 34 + (16 if len(lines) == 1 else 0)), s, font=fq, fill=INK)
        # panes
        for k, label in enumerate((self.a.left_label, self.a.right_label)):
            x = PANE_X[k]
            brut_box(d, (x, PANE_Y0, x + PANE_W, PANE_Y1))
            d.rectangle([x, PANE_Y0, x + PANE_W, PANE_Y0 + HEADER_H], fill=INK if k == 0 else BLUE,
                        outline=INK, width=4)
            f = fit(label.upper(), F_DISPLAY, 36, 500)
            d.text((x + 20, PANE_Y0 + HEADER_H // 2), label.upper(), font=f, fill=WHITE, anchor="lm")
        # strip frame
        sy0, sy1 = STRIP
        brut_box(d, (M, sy0, W - M - SHADOW, sy1))
        # footer
        foot = (f"{'TEST RENDER · PRACTICE DATA · ' if self.a.test else ''}PROBLEM {problem['id']} · "
                f"LEFT {Path(self.a.left).name} · RIGHT {Path(self.a.right).name} · "
                "YELLOW = TEXT REPEATED WORD FOR WORD FROM EARLIER IN THE ANSWER · >> = REAL FAST-FORWARD")
        d.text((W // 2, FOOTER_Y + 12), foot, font=fit(foot, F_MONO, 18, W - 2 * M), fill=DIM, anchor="mm")
        return im

    def _watermark(self) -> Image.Image:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        txt = Image.new("RGBA", (1800, 160), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((900, 80), "TEST RENDER · PRACTICE DATA", font=font(F_DISPLAY, 120),
                                 fill=(17, 17, 17, 34), anchor="mm")
        txt = txt.rotate(18, expand=True)
        layer.alpha_composite(txt, ((W - txt.width) // 2, (H - txt.height) // 2))
        return layer

    # -- one pane's text
    def _pane(self, d, x0: int, s: Stream, k: float, t: float, side: str):
        n_chars = s.chars_at(k)
        if n_chars == 0:
            return
        streaming = k < s.n_retok
        li = bisect.bisect_right(s.starts, n_chars) - 1
        first = max(0, li - self.n_lines + 1)
        mono, mono_b = font(F_MONO, MONO_PX), font(F_MONO_B, MONO_PX)
        fb_mono = font(F_MONO_FB, MONO_PX - 2)
        loop_on = side == "L" and t >= T_LOOP
        flash = 0.5 + 0.5 * math.cos(2 * math.pi * 1.5 * (t - T_LOOP)) if T_LOOP <= t < T_LEFT_FAST else 0.0
        y = PANE_Y0 + HEADER_H + PANE_PAD
        cur_xy = None
        for j in range(first, li + 1):
            a_, b_ = s.lines[j]
            b_ = min(b_, n_chars)
            x = x0 + PANE_PAD + 4
            col = 0
            i = a_
            while i < b_:
                st = self._style(side, i, loop_on, flash)
                kind = self._kind(s.text[i])
                j2 = i + 1
                while j2 < b_ and self._style(side, j2, loop_on, flash) == st and self._kind(s.text[j2]) == kind:
                    j2 += 1
                run = s.text[i:j2]
                cols = sum(2 if is_wide(c) else 1 for c in run)
                fg, bg, bold = st
                xx = x + col * self.cw
                if bg is not None:
                    d.rectangle([xx, y - 2, xx + cols * self.cw, y + LINE_H - 6], fill=bg)
                if kind == 2 and F_CJK:            # wide CJK: two cells per character
                    fc = font(F_CJK, MONO_PX - 4)
                    for q, ch in enumerate(run):
                        d.text((xx + 2 * q * self.cw, y), ch, font=fc, fill=fg)
                elif kind == 1:                    # glyph missing in the main mono font
                    for q, ch in enumerate(run):
                        d.text((xx + q * self.cw, y + 2), ch, font=fb_mono, fill=fg)
                else:
                    d.text((xx, y), run, font=mono_b if bold else mono, fill=fg)
                col += cols
                i = j2
            cur_xy = (x + col * self.cw, y)
            y += LINE_H
        if streaming and cur_xy and int(t * 4) % 2 == 0:
            cx, cy = cur_xy
            d.rectangle([cx + 2, cy + 2, cx + self.cw - 2, cy + LINE_H - 10], fill=INK)

    def _kind(self, ch: str) -> int:
        """0 = main mono font has the glyph, 1 = use the fallback mono, 2 = wide (CJK)."""
        k = self._kinds.get(ch)
        if k is None:
            if is_wide(ch):
                k = 2
            else:
                k = 0 if has_glyph(F_MONO, MONO_PX, ch) else 1
                if k == 1 and not has_glyph(F_MONO_FB, MONO_PX - 2, ch):
                    print(f"warning: no font has a glyph for {ch!r} (U+{ord(ch):04X})", file=sys.stderr)
            self._kinds[ch] = k
        return k

    def _style(self, side: str, i: int, loop_on: bool, flash: float):
        if side == "L" and loop_on and self.loop_mask[i]:
            if flash > 0:
                return (INK, blend(YELLOW, ORANGE, flash), True)
            return (INK, YELLOW, False)
        if side == "R" and self.box and self.box[0] <= i < self.box[1]:
            return (INK, GREEN, True)
        return (INK, None, False)

    # -- header counters, badges, status row
    def _chrome(self, d, t: float, kL: float, kR: float):
        L, R = self.L, self.R
        for side, x0, s, k, keys in (("L", PANE_X[0], L, kL, self.sched["left"]),
                                     ("R", PANE_X[1], R, kR, self.sched["right"])):
            n = s.counter_at(k)
            xr = x0 + PANE_W - 18
            # header counter (right aligned)
            ftc = font(F_MONO_B, 34)
            d.text((xr, PANE_Y0 + HEADER_H // 2), f"{n:,} TOK", font=ftc, fill=WHITE, anchor="rm")
            # LOOP badge in the left header
            if side == "L" and t >= T_LOOP:
                pulse = 0.5 + 0.5 * math.cos(2 * math.pi * 1.5 * (t - T_LOOP)) if t < T_LEFT_FAST else 1.0
                wtok = text_w(ftc, f"{n:,} TOK")
                chip(d, xr - wtok - 18, PANE_Y0 + 8, "↻ LOOP", blend(RED_DARK, RED, pulse), WHITE,
                     px=26, right=True)
            # status row
            sy = STATUS_Y
            x_rate = slope(keys, t) / NORMAL_RATE
            right_edge = x0 + PANE_W
            if k < s.n_retok and x_rate >= 1.5:
                right_edge -= chip(d, right_edge, sy, f"▶▶ {round(x_rate)}x", WHITE, BLUE, px=26,
                                   right=True) + 10
            if side == "L":
                for m in reversed([m for m in MILESTONES if m <= L.n_logged]):
                    on = n >= m
                    right_edge -= chip(d, right_edge, sy, f"{m // 1000}K", YELLOW if on else PAPER,
                                       INK if on else DIM, path=F_DISPLAY, px=26, right=True) + 8
                if t >= T_LEFT_DONE:
                    st = L.res["status"]
                    if st == "cap":
                        big, small = "never finishes!", f"hit the {L.n_logged:,}-token cap, still looping"
                    elif st == "loop":
                        big, small = "stuck in a loop", f"stopped by the loop detector at {L.n_logged:,}"
                    else:
                        big, small = ("wrong answer" if not L.res["correct"] else "finished"), ""
                    fb = font(F_HAND, 44)
                    d.text((x0 + 4, sy - 2), big, font=fb, fill=RED)
                    if small:
                        d.text((x0 + 18 + text_w(fb, big), sy + 12), small,
                               font=fit(small, F_MONO, 20, max(120, int(right_edge - x0 - text_w(fb, big) - 30))),
                               fill=DIM)
                elif t >= T_LOOP:
                    d.text((x0 + 4, sy - 2), "repeating itself…", font=font(F_HAND, 44), fill=RED)
                else:
                    d.text((x0 + 4, sy + 6), "REASONING…", font=font(F_MONO_B, 26), fill=DIM)
            else:
                if t >= self.right_done_t:
                    st = R.res["status"]
                    if R.res["correct"]:
                        chip(d, x0, sy - 2, f"SOLVED IN {R.n_logged:,} TOKENS", GREEN, INK, path=F_DISPLAY, px=32)
                    elif st == "cap":
                        chip(d, x0, sy - 2, f"HIT THE {R.n_logged:,}-TOKEN CAP", PAPER, INK, path=F_DISPLAY, px=32)
                    elif st == "loop":
                        chip(d, x0, sy - 2, f"STUCK IN A LOOP ({R.n_logged:,})", PAPER, INK, path=F_DISPLAY, px=32)
                    else:
                        chip(d, x0, sy - 2, f"ANSWERED, WRONG ({R.n_logged:,})", PAPER, INK, path=F_DISPLAY, px=32)
                else:
                    d.text((x0 + 4, sy + 6), "REASONING…", font=font(F_MONO_B, 26), fill=DIM)

    # -- bottom strip: what is happening (illustration)
    def _network_static(self):
        sx0, sy0, sx1, sy1 = M, STRIP[0], W - M - SHADOW, STRIP[1]
        w, h = sx1 - sx0 - 8, sy1 - sy0 - 8
        im = Image.new("RGBA", (w, h), WHITE + (255,))
        d = ImageDraw.Draw(im)
        nx0, nx1, ny0, ny1 = 420, 1250, 38, h - 52
        nl, nn = 28, 5
        nodes = [[(nx0 + i * (nx1 - nx0) / (nl - 1), ny0 + j * (ny1 - ny0) / (nn - 1)) for j in range(nn)]
                 for i in range(nl)]
        for i in range(nl - 1):
            for (xa, ya) in nodes[i]:
                for (xb, yb) in nodes[i + 1]:
                    d.line([xa, ya, xb, yb], fill=RULE, width=1)
        for col in nodes:
            for (x, y) in col:
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=WHITE, outline=INK, width=2)
        d.text(((nx0 + nx1) / 2, h - 30),
               f"ILLUSTRATION · DEEPSEEK-R1 1.5B · 28 LAYERS · {self.total_weights / 1e9:.2f}B WEIGHTS",
               font=font(F_MONO_B, 20), fill=INK, anchor="mm")
        # pixel character + label
        cell = 13
        cx0, cy0 = 70, 46
        for r_, row in enumerate(CHAR):
            for c_, v in enumerate(row):
                if v != ".":
                    d.rectangle([cx0 + c_ * cell, cy0 + r_ * cell, cx0 + (c_ + 1) * cell - 1,
                                 cy0 + (r_ + 1) * cell - 1], fill=INK if v == "O" else CHAR_COLOR)
        d.text((cx0 + 5 * cell, cy0 + 6 * cell + 30), "CLAUDE", font=font(F_DISPLAY, 34), fill=INK, anchor="mm")
        d.text((cx0 + 5 * cell, cy0 + 6 * cell + 62), "reads + edits weights", font=font(F_HAND, 22),
               fill=RED, anchor="mm")
        # arrow to output
        d.line([1290, h / 2 - 10, 1370, h / 2 - 10], fill=INK, width=5)
        d.polygon([(1370, h / 2 - 24), (1394, h / 2 - 10), (1370, h / 2 + 4)], fill=INK)
        self.char_eye = (cx0 + 7.5 * cell, cy0 + 2.5 * cell)
        self.net_box = (nx0, nx1, ny0, ny1)
        return im, nodes

    def strip(self, t: float) -> Image.Image:
        ts = t % CYCLE
        im = self.net_static.copy()
        d = ImageDraw.Draw(im, "RGBA")
        nodes = self.net_nodes
        nx0, nx1, ny0, ny1 = self.net_box
        nl = len(nodes)
        layers = self.layers or ([] if not self.a.test else [9, 14, 15])   # test: placeholders
        # phase 1 (0-2 s): beam sweeps the layers
        if ts < 2.2:
            bx = nx0 + (nx1 - nx0) * min(1.0, ts / 2.0)
            ex, ey = self.char_eye
            ov = Image.new("L", im.size, 0)
            ImageDraw.Draw(ov).polygon([(ex, ey), (bx - 40, ny0 - 6), (bx + 40, ny0 - 6),
                                        (bx + 40, ny1 + 6), (bx - 40, ny1 + 6)], fill=70)
            im.paste(Image.new("RGB", im.size, RED), (0, 0), ov)
            d = ImageDraw.Draw(im, "RGBA")
        # phase 2 (2-3.6 s): edited layers light up, their incoming weights recolour
        edit_a = 0.0
        if 2.0 <= ts < 6.6:
            edit_a = min(1.0, (ts - 2.0) / 0.5)
        if ts >= 6.6:
            edit_a = max(0.0, 1 - (ts - 6.6) / 0.9)
        if edit_a > 0:
            valid = [L_ for L_ in layers if 0 <= L_ < nl]
            for i in valid:
                if i > 0:
                    for (xa, ya) in nodes[i - 1]:
                        for (xb, yb) in nodes[i]:
                            d.line([xa, ya, xb, yb], fill=BLUE + (int(220 * edit_a),), width=3)
                for (x, y) in nodes[i]:
                    d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=RED + (int(255 * edit_a),), outline=INK, width=2)
            # one label per run of consecutive layers ("L14-15")
            runs, cur = [], []
            for i in valid:
                if cur and i != cur[-1] + 1:
                    runs.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                runs.append(cur)
            for run in runs:
                lab = f"L{run[0]}" if len(run) == 1 else f"L{run[0]}-{run[-1]}"
                xm = (nodes[run[0]][0][0] + nodes[run[-1]][0][0]) / 2
                d.text((xm, ny0 - 26), lab, font=font(F_MONO_B, 18), fill=INK + (int(255 * edit_a),), anchor="mm")
        # phase 3 (3.6-6.6 s): signal flows through, output flips to the answer
        after = 3.6 <= ts < 6.9
        if after:
            frac = ((ts - 3.6) / 1.2) % 1.0
            for j in (0, 2, 4):
                x = nx0 + (nx1 - nx0) * frac
                y = nodes[0][j][1]
                d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=GREEN + (255,), outline=INK, width=2)
        # output card
        ox0, oy0, ox1, oy1 = 1410, 22, im.width - 20, im.height - 22
        d.rectangle([ox0, oy0, ox1, oy1], fill=WHITE, outline=INK, width=4)
        d.rectangle([ox0, oy0, ox1, oy0 + 40], fill=INK)
        d.text((ox0 + 14, oy0 + 20), "OUTPUT · " + ("AFTER EDIT" if after else "BEFORE"),
               font=font(F_DISPLAY, 26), fill=WHITE, anchor="lm")
        body_w = ox1 - ox0 - 28
        if after and self.answer:
            f = fit(self.answer, F_MONO_B, 34, body_w - 24)
            tw = text_w(f, self.answer)
            bx0 = ox0 + 14
            d.rectangle([bx0, oy0 + 66, bx0 + tw + 24, oy0 + 124], fill=GREEN, outline=INK, width=3)
            d.text((bx0 + 12, oy0 + 95), self.answer, font=f, fill=INK, anchor="lm")
            d.text((bx0, oy0 + 136), "solved", font=font(F_HAND, 30), fill=GREEN_DARK)
        else:
            reps = 1 + int((ts % 3.6) / 1.2)
            for r_ in range(min(reps, 2)):
                d.text((ox0 + 14, oy0 + 54 + r_ * 36), self.loop_snippet,
                       font=fit(self.loop_snippet, F_MONO, 24, body_w), fill=RED)
            if reps >= 3:
                d.text((ox1 - 16, oy1 - 8), "again… and again…", font=font(F_HAND, 28), fill=RED, anchor="rb")
        # caption: what really changed
        if self.layers:
            cap = f"EDITED: {' · '.join(self.layer_desc)} · {self.pct} OF WEIGHTS"
        elif self.a.test:
            cap = "TEST RENDER · NO EDIT YET · HIGHLIGHTED LAYERS ARE PLACEHOLDERS"
        else:
            cap = ""
        if cap:
            d.text((14, 10), cap, font=fit(cap, F_MONO_B, 18, 1100), fill=BLUE)
        return im.convert("RGB")

    # -- main timeline frame
    def main_frame(self, t: float) -> Image.Image:
        im = self.static.copy()
        d = ImageDraw.Draw(im)
        kL, kR = interp(self.sched["left"], t), interp(self.sched["right"], t)
        self._pane(d, PANE_X[0], self.L, kL, t, "L")
        self._pane(d, PANE_X[1], self.R, kR, t, "R")
        self._chrome(d, t, kL, kR)
        im.paste(self.strip(t), (M + 4, STRIP[0] + 4))
        if t >= T_CARD:
            im = self._end_card(im, min(1.0, (t - T_CARD) / 0.4))
        return im

    def _end_card(self, im: Image.Image, a: float) -> Image.Image:
        card = im.copy()
        d = ImageDraw.Draw(card)
        x0, y0, x1, y1 = 300, 270, W - 300, 770
        brut_box(d, (x0, y0, x1, y1), fill=PAPER, border=6, shadow=12)
        st = self.stats
        fD, fS = font(F_DISPLAY, 100), font(F_SYM, 80)
        y = y0 + 50
        for label, before, after in (("LOOP RATE", st["base_loop"], st["edit_loop"]),
                                     ("ACCURACY", st["base_acc"], st["edit_acc"])):
            mixed(d, x0, y, [(label + "  ", fD, INK), (f"{100 * before:.1f}%", fD, RED), ("  →  ", fS, INK),
                             (f"{100 * after:.1f}%", fD, GREEN_DARK)], anchor_center_w=x1 - x0)
            y += 140
        s3 = "NO TRAINING. LAPTOP RTX 4050."
        d.rectangle([x0 + 6, y + 10, x1 - 6, y1 - 6], fill=BLUE)
        d.text(((x0 + x1) // 2, (y + 10 + y1) // 2), s3, font=fit(s3, F_DISPLAY, 80, x1 - x0 - 80),
               fill=WHITE, anchor="mm")
        return Image.blend(im, card, a)

    # -- explainer cards
    def card(self, title: str, body: list[str], table: list[list[str]] | None = None,
             foot: str | None = None, note: str | None = None) -> Image.Image:
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = 140, 90, W - 150, H - 110
        brut_box(d, (x0, y0, x1, y1), border=6, shadow=14)
        ft = font(F_DISPLAY, 92)
        y = y0 + 50
        for s in wrap_px(title.upper(), ft, x1 - x0 - 120):
            d.text((x0 + 60, y), s, font=ft, fill=INK)
            y += 104
        d.rectangle([x0 + 60, y + 6, x0 + 260, y + 18], fill=BLUE)
        y += 54
        fb = font(F_SANS, 42)
        for para in body:
            for s in wrap_px(para, fb, x1 - x0 - 140):
                d.text((x0 + 60, y), s, font=fb, fill=INK)
                y += 56
            y += 22
        if table:
            fh, fr = font(F_DISPLAY, 40), font(F_SANS_B, 40)
            cols_x = [x0 + 60, x0 + 900, x0 + 1260]
            for r_i, row in enumerate(table):
                f = fh if r_i == 0 else fr
                for c_i, cell in enumerate(row):
                    d.text((cols_x[c_i], y), cell, font=f, fill=DIM if r_i == 0 else INK)
                y += 62
                d.line([x0 + 60, y - 8, x1 - 60, y - 8], fill=INK, width=4 if r_i == 0 else 1)
        if note:
            d.text((x1 - 60, y1 - 70), note, font=font(F_HAND, 40), fill=RED, anchor="rb")
        if foot:
            d.text((W // 2, H - 60), foot, font=fit(foot, F_MONO, 22, W - 200), fill=DIM, anchor="mm")
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
        t = T_MAIN_END + i / FPS
        a = last.copy()
        a.paste(r.strip(t), (M + 4, STRIP[0] + 4))     # strip keeps animating into frame 0
        yield t, fade(a, first, (i + 1) / n_x)


def frames_explainer(r: Renderer, cards: dict):
    intro, ctrl, meth, close = cards["intro"], cards["control"], cards["method"], cards["close"]
    blank = Image.new("RGB", (W, H), PAPER)
    seq = []
    t0 = 0.0
    for i in range(int(EX_INTRO * FPS)):
        t = i / FPS
        im = fade(blank, intro, t / 0.5) if t < 0.5 else intro
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
         "of 50 model runs on a laptop GPU."],
        note="watch the same problem, before vs after →",
        foot="Inspired by MIT CSAIL, \"Sparks of Weight-Space Programming\" (Nov 2026)")
    right_name = "Claude's edit" if a.right_label == RIGHT_LABEL else a.right_label
    table = [["ARM", "ACCURACY", "LOOP RATE"],
             ["Original", f"{100 * st['base_acc']:.1f}%", f"{100 * st['base_loop']:.1f}%"],
             [right_name, f"{100 * st['edit_acc']:.1f}%", f"{100 * st['edit_loop']:.1f}%"]]
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
         f"• Paired comparison on the same {st['n']} problems, with a measured run-to-run noise floor",
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
    ap.add_argument("--patch-info", default=None, help="work/edit/patch_info.json (0.0Z%% and edited layers)")
    ap.add_argument("--control", action="append", default=[], help='"Label=results/rand_w_....jsonl"')
    ap.add_argument("--left-label", default=LEFT_LABEL)
    ap.add_argument("--right-label", default=RIGHT_LABEL)
    ap.add_argument("--caption2", default=None, help="override the blue masthead caption (test renders)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--test", action="store_true", help="watermark: practice-data test render")
    ap.add_argument("--stills", default=None, help="also save PNG stills at key times into this dir")
    ap.add_argument("--gif", default=None, help="also write the bottom strip as a looping GIF")
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
    sched["loop_mask"] = loop[2]

    patch_info = json.loads(Path(a.patch_info).read_text()) if a.patch_info else None
    pct = fmt_pct(patch_info["changed_pct"]) if patch_info else "0.0Z%"
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
            "loop_onset_token": sched["onset"], "typical_repeat_distance": sched["period"],
            "repeated_tokens_after_onset": sum(loop[2]),
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

    r = Renderer(a, L, R, problem, sched, stats, pct, patch_info)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps({**diag, "args": vars(a)}, indent=2))

    if a.gif:
        gif_frames = []
        for i in range(int(CYCLE * 15)):
            fr = r.strip(i / 15)
            fr = fr.resize((fr.width * 2 // 3, fr.height * 2 // 3), Image.LANCZOS)
            gif_frames.append(fr.convert("P", palette=Image.ADAPTIVE, colors=64))
        gif_frames[0].save(a.gif, save_all=True, append_images=gif_frames[1:], duration=67, loop=0,
                           optimize=True)
        print(f"wrote {a.gif}")

    if a.variant == "short":
        frames = frames_short(r)
        stills_t = [0.0, 1.0, 3.0, 5.0, 9.0, 11.0, 14.0, 17.0, 20.0, 22.8, 25.0, 29.0]
    else:
        frames = frames_explainer(r, explainer_cards(r, a, pct))
        stills_t = [3.0, 12.0, 20.0, 29.0, 34.0, 40.0, 50.0, 57.0]

    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
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
        if n % 300 == 0:
            print(f"  {n / FPS:.0f}s rendered", file=sys.stderr)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out}  ({n} frames, {n / FPS:.1f} s)")


if __name__ == "__main__":
    main()
