"""Figures for the write-up, the README and the blog article.

  python figures.py --base results/base_math500_greedy.jsonl \
      --arm "Claude's edit=results/edit_math500_greedy.jsonl" \
      --arm "Random edit (same weights)=results/rand_w_math500_greedy.jsonl" \
      --arm "Random edit (random positions)=results/rand_s_math500_greedy.jsonl" \
      --arm "Base + presence penalty=results/base_pp_math500_greedy.jsonl" \
      --arm "Edit + presence penalty=results/edit_pp_math500_greedy.jsonl" \
      --rerun results/base_rerun_math500_greedy.jsonl --rerun results/base_rerun2_math500_greedy.jsonl ... \
      --out figures/

Draws (PNG, 1600x900, readable on a phone):
  1. fig1_accuracy_loops.png  accuracy and loop rate per arm, 95% bootstrap CI whiskers, rerun noise band;
  2. fig2_projection.png      residual-stream projection on the editor's direction across one answer,
                              before vs after (only with --projection, a JSON written after freeze:
                              {"title": ..., "before": [floats], "after": [floats], "loop_start": int|null});
  3. fig3_fixed_broken.png    problems fixed / broken per arm vs base, with the reruns' range for scale.

Statistics come from analyze.py (same bootstrap, same seed), so the numbers match its tables.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from analyze import boot_ci, per_problem  # noqa: E402

PAPER, INK, BLUE, RED = "#F4F1EA", "#111111", "#1F3BFF", "#EB3324"
YELLOW, ORANGE, GREEN, GREY = "#FFE08A", "#FF9F1C", "#22C55E", "#BDB8AD"

WINF = "/mnt/c/Windows/Fonts/"
for f in ("impact.ttf", "arial.ttf", "arialbd.ttf", "consola.ttf", "consolab.ttf"):
    if Path(WINF + f).exists():
        font_manager.fontManager.addfont(WINF + f)
HAVE_IMPACT = Path(WINF + "impact.ttf").exists()
DISPLAY = "Impact" if HAVE_IMPACT else "DejaVu Sans"
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 15, "axes.facecolor": "white",
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER, "axes.edgecolor": INK, "axes.linewidth": 2.5,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "xtick.major.width": 2,
    "ytick.major.width": 2, "axes.spines.top": False, "axes.spines.right": False,
})
SHADOW = [pe.SimplePatchShadow(offset=(5, -5), shadow_rgbFace=INK, alpha=1.0), pe.Normal()]


def colour(label: str) -> tuple[str, str | None]:
    """(face colour, hatch) by arm name, so every figure uses the same coding."""
    s = label.lower()
    if "random" in s:
        return GREY, "//"
    if "penalty" in s and "edit" in s:
        return GREEN, None
    if "penalty" in s:
        return YELLOW, None
    if "edit" in s:
        return BLUE, None
    return "#FFFFFF", None


def wrap(label: str) -> str:
    return "\n".join(textwrap.wrap(label, 24))


def title(fig, text: str, sub: str | None = None) -> None:
    fig.text(0.04, 0.93, text.upper(), fontsize=30, fontfamily=DISPLAY, color=INK, va="center")
    if sub:
        fig.text(0.04, 0.865, sub, fontsize=14, color="#555555", va="center")


def watermark(fig, test: bool) -> None:
    if test:
        fig.text(0.5, 0.5, "TEST · PRACTICE DATA", fontsize=70, color="#00000018", rotation=18,
                 ha="center", va="center", fontfamily=DISPLAY, zorder=100)


def level(path: str, key: str, ids: list[str], rng) -> tuple[float, float, float]:
    pp = per_problem(path)
    v = np.array([pp[i][key] for i in ids])
    lo, hi = boot_ci(v, rng)
    return float(v.mean()), lo, hi


def fig_levels(base: str, arms: list[tuple[str, str]], reruns: list[str], out: Path, test: bool, note: str):
    rng = np.random.default_rng(0)
    ids = sorted(per_problem(base))
    rows = [("Original model", base)] + arms
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=100)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.12, wspace=0.12)
    title(fig, "Accuracy and loop rate per arm", note)
    for ax, key, name in ((axes[0], "acc", "ACCURACY"), (axes[1], "loop", "LOOP RATE")):
        ax.set_title(name, fontfamily=DISPLAY, fontsize=22, loc="left", color=INK, pad=12)
        if reruns:   # run-to-run noise: spread of the base reruns (normal kernels)
            vals = [level(r, key, ids, rng)[0] for r in reruns]
            ax.axvspan(100 * min(vals), 100 * max(vals), color=ORANGE, alpha=0.30, lw=0, zorder=0)
        for y, (label, path) in enumerate(rows):
            m, lo, hi = level(path, key, ids, rng)
            face, hatch = colour(label)
            bar = ax.barh(len(rows) - 1 - y, 100 * m, height=0.62, color=face, hatch=hatch,
                          edgecolor=INK, linewidth=2.5, zorder=2)
            bar[0].set_path_effects(SHADOW)
            ax.errorbar(100 * m, len(rows) - 1 - y, xerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none",
                        ecolor=INK, elinewidth=2.5, capsize=7, capthick=2.5, zorder=3)
            ax.text(100 * hi + 1.5, len(rows) - 1 - y, f"{100 * m:.1f}%", va="center", fontsize=16,
                    fontweight="bold", color=INK, zorder=4)
        ax.set_xlim(0, 112)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xticklabels([f"{t}%" for t in range(0, 101, 20)])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([wrap(r[0]) for r in rows][::-1] if ax is axes[0] else [], fontsize=16)
        ax.grid(axis="x", color="#00000015", lw=1, zorder=0)
    fig.text(0.04, 0.035, f"n = {len(ids)} problems, paired · whiskers: 95% bootstrap CI (10,000 resamples) "
             + (f"· orange band: spread of {len(reruns)} base reruns" if reruns else ""), fontsize=12, color="#555555", fontfamily="Consolas")
    watermark(fig, test)
    fig.savefig(out / "fig1_accuracy_loops.png")
    plt.close(fig)


def fig_fixed_broken(base: str, arms: list[tuple[str, str]], reruns: list[str], out: Path, test: bool,
                     note: str):
    b = per_problem(base)

    def fb(path):
        a = per_problem(path)
        ids = sorted(set(a) & set(b))
        fixed = sum(1 for i in ids if a[i]["acc"] > b[i]["acc"])
        broken = sum(1 for i in ids if a[i]["acc"] < b[i]["acc"])
        return fixed, broken

    rows = [(label, *fb(path)) for label, path in arms]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.subplots_adjust(left=0.27, right=0.95, top=0.78, bottom=0.14)
    title(fig, "Problems fixed vs broken", note)
    if reruns:
        rr = [fb(r) for r in reruns]
        nets = [f - k for f, k in rr]
        ax.axvspan(-max(k for _, k in rr), max(f for f, _ in rr), color=ORANGE, alpha=0.22, lw=0, zorder=0)
        fig.text(0.04, 0.035, f"orange band: {len(reruns)} base reruns with no edit fixed up to "
                 f"{max(f for f, _ in rr)} and broke up to {max(k for _, k in rr)} problems "
                 f"(net {min(nets):+d} to {max(nets):+d})", fontsize=12, color="#555555", fontfamily="Consolas")
    span = max([max(f, k) for _, f, k in rows] + [1])
    for y, (label, fixed, broken) in enumerate(rows):
        yy = len(rows) - 1 - y
        face, hatch = colour(label)
        g = ax.barh(yy, fixed, height=0.6, color=face, hatch=hatch, edgecolor=INK, lw=2.5, zorder=2)
        r = ax.barh(yy, -broken, height=0.6, color=RED, edgecolor=INK, lw=2.5, zorder=2)
        g[0].set_path_effects(SHADOW)
        r[0].set_path_effects(SHADOW)
        ax.text(fixed + span * 0.02, yy, f"+{fixed}", va="center", fontsize=16, fontweight="bold")
        ax.text(-broken - span * 0.02, yy, f"−{broken}", va="center", ha="right", fontsize=16,
                fontweight="bold", color=RED)
        ax.text(span * 1.32, yy, f"net {fixed - broken:+d}", va="center", ha="right", fontsize=16,
                fontfamily="Consolas", fontweight="bold")
    ax.axvline(0, color=INK, lw=2.5)
    ax.set_xlim(-span * 1.3, span * 1.35)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([wrap(r[0]) for r in rows][::-1], fontsize=16)
    ax.set_xlabel("← broken (right → wrong)          fixed (wrong → right) →", fontsize=14)
    ax.grid(axis="x", color="#00000015", lw=1, zorder=0)
    watermark(fig, test)
    fig.savefig(out / "fig3_fixed_broken.png")
    plt.close(fig)


def fig_projection(path: str, out: Path, test: bool):
    d = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.12)
    title(fig, d.get("title", "Projection on the edited direction"), d.get("subtitle"))
    ax.plot(d["before"], color=RED, lw=2.5, label="original model")
    ax.plot(d["after"], color=BLUE, lw=2.5, label="after the edit")
    if d.get("loop_start") is not None:
        ax.axvline(d["loop_start"], color=INK, lw=2, ls="--")
        ax.text(d["loop_start"], ax.get_ylim()[1], "  loop starts (original)", va="top", fontsize=13)
    ax.set_xlabel("token position in the answer")
    ax.set_ylabel("projection")
    leg = ax.legend(frameon=True, fontsize=15, loc="upper left")
    leg.get_frame().set_edgecolor(INK)
    leg.get_frame().set_linewidth(2)
    watermark(fig, test)
    fig.savefig(out / "fig2_projection.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", default=[], help='"Label=path/to/results.jsonl"')
    ap.add_argument("--rerun", action="append", default=[], help="base reruns, for the noise band")
    ap.add_argument("--projection", help="JSON with before/after projections (after freeze)")
    ap.add_argument("--note", default="", help="subtitle, e.g. 'MATH-500, 200-problem fast track, greedy'")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--test", action="store_true", help="watermark: test render on practice data")
    a = ap.parse_args()
    arms = [tuple(s.split("=", 1)) for s in a.arm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_levels(a.base, arms, a.rerun, out, a.test, a.note)
    if arms:
        fig_fixed_broken(a.base, arms, a.rerun, out, a.test, a.note)
    if a.projection:
        fig_projection(a.projection, out, a.test)
    print(f"wrote {sorted(p.name for p in out.glob('fig*.png'))} in {out}")


if __name__ == "__main__":
    main()
