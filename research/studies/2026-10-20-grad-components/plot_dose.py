#!/usr/bin/env python3
"""Exchange-rate figure from dose/records.jsonl: likeness gain vs snapshot change and vs scene change, one curve per component
(dose 0, ×1, ×2, ×3 in style-dial energy units; mean ± SE over contexts). Overlapping curves = one exchange rate = a conditioning-space ceiling.

  python plot_dose.py   -> runs/2026-10-20-grad-components/dose/exchange_rate.png (+ .svg)
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(os.environ.get("DETANGLE_RESEARCH_ROOT", "detangle-research")).resolve() / "runs/2026-10-20-grad-components"
# validated categorical slots 1-6 (fixed order) + a distinct marker per series (contrast WARN -> secondary encoding + direct labels)
SERIES = [("style (dial)", "style dial", "#2a78d6", "o"), ("style_spec", "style_spec", "#eb6834", "s"), ("pca4", "pca4", "#1baf7a", "D"),
          ("pca2", "pca2", "#eda100", "^"), ("identity_noclothes", "identity − clothes", "#e87ba4", "v"), ("rand_span1", "random (control)", "#008300", "X")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"


def main():
    rows = [json.loads(l) for l in (RUN / "dose/records.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["J0"]]
    doses = sorted({float(k.split("|")[1]) for r in rows for k in r if "|" in k})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, j, xl in zip(axes, ("snap", "scene"), ("snapshot look change (→ more snapshot-like)", "scene change vs Base (← scene lost)")):
        for key, label, col, mk in SERIES:
            pts = [(0.0, 0.0, 0.0, 0.0)]
            for d in doses:
                k = f"{key}|{d:g}"
                ok = [r for r in rows if r.get(k)]
                if len(ok) < 3:
                    continue
                dx = np.array([r[k][j] - r["J0"][j] for r in ok])
                dy = np.array([r[k]["id"] - r["J0"]["id"] for r in ok])
                pts.append((dx.mean(), dy.mean(), dx.std(ddof=1) / np.sqrt(len(ok)), dy.std(ddof=1) / np.sqrt(len(ok))))
            p = np.array(pts)
            ax.errorbar(p[1:, 0], p[1:, 1], xerr=p[1:, 2], yerr=p[1:, 3], fmt="none", ecolor=col, elinewidth=1, alpha=0.5)
            ax.plot(p[:, 0], p[:, 1], color=col, lw=2, marker=mk, ms=7, mec="white", mew=1.5, label=label, zorder=3)
            ax.annotate(label, (p[-1, 0], p[-1, 1]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(xl, color=INK, fontsize=9)
    axes[0].set_ylabel("likeness gain (ArcFace to centroid)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Exchange rate of text-side amp components from style 0 (doses ×{', ×'.join(f'{d:g}' for d in doses)}; n={len(rows)} contexts)", color=INK, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(RUN / f"dose/exchange_rate.{ext}", dpi=160, facecolor="white")
    print(RUN / "dose/exchange_rate.png")


if __name__ == "__main__":
    main()
