#!/usr/bin/env python3
"""Identity map figure (local only): 2-D PCA of face-region activations (face minus image-wide mean, standardised) at a shallow-early and a
deep-late point. Colour = group, marker = group (secondary encoding), celebrity clusters labelled at their means."""
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(os.environ.get("DETANGLE_RESEARCH_ROOT", "detangle-research")).resolve() / "runs/2026-10-20-grad-components/idmap"
GROUPS = [("young celebrity", "Celebrities ~25–35", "#2a78d6", "o"), ("older celebrity", "Celebrities ~55–75", "#eb6834", "s"),
          ("generic 30", "“a 30-year-old woman”", "#1baf7a", "^"), ("generic 60", "“a 60-year-old woman”", "#eda100", "v"),
          ("subject repair", "Subject A (repair)", "#e87ba4", "D"), ("subject naive", "Subject A (original LoRA)", "#6250d6", "P")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"


def main():
    recs = torch.load(RUN / "records.pt", weights_only=False)
    gkey = lambda r: f"subject {r['arm']}" if r["group"] == "subject" else r["group"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key, title in zip(axes, [(4, 2), (12, 7)], ["Block 4, step 2 (shallow, early)", "Block 12, step 7 (deep, late)"]):
        X = np.array([r["face"][key].numpy().ravel() - r["whole"][key].numpy().ravel() for r in recs], dtype=np.float64)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        P = (Z - Z.mean(0)) @ Vt[:2].T
        for g, label, col, mk in GROUPS:
            idx = [i for i, r in enumerate(recs) if gkey(r) == g]
            ax.scatter(P[idx, 0], P[idx, 1], c=col, marker=mk, s=46, edgecolors="white", linewidths=1, label=label, zorder=3)
        for who in dict.fromkeys(r["who"] for r in recs if "celebrity" in r["group"]):
            idx = [i for i, r in enumerate(recs) if r["who"] == who]
            m = P[idx].mean(0)
            ax.annotate(who.split()[-1], m, xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("PC 1", color=MUTED, fontsize=8)
        ax.set_ylabel("PC 2", color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Face-region activations (face minus image average), 4 seeds per subject", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(RUN / "identity_map.png", dpi=150, facecolor="white")
    print(RUN / "identity_map.png")


if __name__ == "__main__":
    main()
