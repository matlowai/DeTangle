#!/usr/bin/env python3
"""Controls for the identity map: (1) face minus whole-image mean (removes prompt-wide 'which name' information carried by every token),
(2) identity subspace fitted within each age group (age removed), (3) where Subject A's offset lands in that age-free subspace."""
import json
import os
from pathlib import Path
import numpy as np
import torch

RUN = Path(os.environ.get("DETANGLE_RESEARCH_ROOT", "detangle-research")).resolve() / "runs/2026-10-20-grad-components/idmap"
YOUNG = ["Florence Pugh", "Anya Taylor-Joy", "Saoirse Ronan", "Emma Stone", "Jennifer Lawrence", "Kristen Stewart"]


def nn_acc(X, y, seeds):
    hits = 0
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if seeds[j] != seeds[i]]
        cents = {c: X[[j for j in tr if y[j] == c]].mean(0) for c in set(y)}
        sims = {c: float(X[i] @ v / (np.linalg.norm(X[i]) * np.linalg.norm(v) + 1e-9)) for c, v in cents.items()}
        hits += max(sims, key=sims.get) == y[i]
    return hits / len(y)


def main():
    recs = torch.load(RUN / "records.pt", weights_only=False)
    cel = [r for r in recs if "celebrity" in r["group"]]
    y, seeds = [r["who"] for r in cel], [r["seed"] for r in cel]
    young = np.array([t in YOUNG for t in y])
    gen30 = [r for r in recs if r["group"] == "generic 30"]
    out = {}
    for key in sorted(cel[0]["face"]):
        F = np.array([r["face"][key].numpy().ravel() for r in cel], dtype=np.float64)
        Wh = np.array([r["whole"][key].numpy().ravel() for r in cel], dtype=np.float64)
        D = F - Wh  # face-specific: what the face region has beyond the image-wide average
        mu, sd = D.mean(0), D.std(0) + 1e-6
        Z = (D - mu) / sd
        e = {"nn_face_minus_whole": nn_acc(Z, y, seeds),
             "nn_within_young": nn_acc(Z[young], [t for t, k in zip(y, young) if k], [s for s, k in zip(seeds, young) if k]),
             "nn_within_older": nn_acc(Z[~young], [t for t, k in zip(y, young) if not k], [s for s, k in zip(seeds, young) if not k])}
        # identity subspace with age removed: celebrity means centred within their age group
        means = []
        for grp in (young, ~young):
            names = sorted({t for t, k in zip(y, grp) if k})
            m = np.array([Z[[t == n for t in y]].mean(0) for n in names])
            means.append(m - m.mean(0))
        U = np.linalg.svd(np.vstack(means), full_matrices=False)[2][:8].T
        age = Z[~young].mean(0) - Z[young].mean(0)
        e["age_axis_in_agefree_identity"] = float(np.sum((age / np.linalg.norm(age)) @ U) ** 2) if False else float(np.sum(((age / np.linalg.norm(age)) @ U) ** 2))
        G = (np.array([r["face"][key].numpy().ravel() - r["whole"][key].numpy().ravel() for r in gen30]) - mu) / sd
        for arm in ("repair", "naive"):
            S = (np.array([r["face"][key].numpy().ravel() - r["whole"][key].numpy().ravel() for r in recs if r["group"] == "subject" and r["arm"] == arm]) - mu) / sd
            d = S.mean(0) - G.mean(0)
            e[f"subject_{arm}_in_agefree_identity"] = float(np.sum((d @ U) ** 2) / np.sum(d ** 2))
        e["random_8d"] = 8 / Z.shape[1]
        out[f"block {key[0]} step {key[1]}"] = {k: round(v, 4) for k, v in e.items()}
    (RUN / "controls.json").write_text(json.dumps(out, indent=1))
    print(f"{'block/step':18s} nnFace-Whole nnYoung nnOlder ageInAgeFreeID subjRep subjOrig random")
    for k, v in out.items():
        print(f"{k:18s} {v['nn_face_minus_whole']:.2f}  {v['nn_within_young']:.2f}  {v['nn_within_older']:.2f}  {v['age_axis_in_agefree_identity']:.3f}  {v['subject_repair_in_agefree_identity']:.3f}  {v['subject_naive_in_agefree_identity']:.3f}  {v['random_8d']:.4f}")


if __name__ == "__main__":
    main()
