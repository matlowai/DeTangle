#!/usr/bin/env python3
"""Paired per-seed differences of arms vs reference arms (her likeness, celebrity kept) + guard: her face vs each arm's top names.

  python paired.py --arms knn1_200,knn3_200,mean_cap487_400 --refs pb_b0_27_200                        (run 1, seeds 181909-12)
  python paired.py --arms span_200,hull_200 --refs pb_b0_27_200,knn1_200 --guard guard_v2.json        (run 2)
  python paired.py --seeds2 --arms knn1_nopugh_200 --refs pb_b0_27_200 --guard guard_seeds2.json       (run 2, fresh seeds 181913-16)
"""
import argparse, json
import os
from pathlib import Path
import numpy as np

RUN = Path(os.environ.get("DETANGLE_RESEARCH_ROOT", "detangle-research")).resolve() / "runs/2026-11-01-knn-prebind"
OLD = RUN.parent / "2026-10-26-krea-gate"
BASELINE = "pb_b0_27_200"
RF = ["bench", "cafe", "crew", "noir", "viking", "pugh", "hathaway", "chastain"]
GRP = ["bench", "cafe", "crew", "pugh", "hathaway", "chastain"]
CF = ["adams_first", "theron_first", "johansson_first", "weaver_first"]

ap = argparse.ArgumentParser()
ap.add_argument("--arms", required=True)
ap.add_argument("--refs", default=BASELINE)
ap.add_argument("--seeds2", action="store_true")
ap.add_argument("--guard", default=None, help="guard json in the run folder (default guard.json / guard_seeds2.json)")
a = ap.parse_args()
suffix = "_seeds2" if a.seeds2 else ""
seeds = range(181913, 181917) if a.seeds2 else range(181909, 181913)
rows = json.loads((RUN / f"eval{suffix}.json").read_text())
rows += [r for r in json.loads((OLD / f"prebind_eval{suffix}.json").read_text()) if r["arm"] == BASELINE and r["seed"] in seeds]


def score(r):
    """(her, celebrity kept, her on the celebrity's face) with prebind.summarize's face assignment."""
    h = list(r["her"])
    if "named" in r and len(h) == 2:
        j = int(np.argmax(r["named"]))
        return h[1 - j], r["named"][j], h[j]
    return (max(h) if h else None), None, None


idx = {(r["arm"], r["case"], r["seed"]): score(r) for r in rows}
for ref in a.refs.split(","):
    for arm in a.arms.split(","):
        for lab, cs, k in (("her, rlsa-first all 8", RF, 0), ("her, group 6", GRP, 0), ("her, solo 2", ["noir", "viking"], 0),
                           ("celebrity kept, celeb-first 4", CF, 1), ("her, celeb-first 4", CF, 0)):
            d = np.array([idx[(arm, c, s)][k] - idx[(ref, c, s)][k] for c in cs for s in seeds
                          if (arm, c, s) in idx and (ref, c, s) in idx and idx[(arm, c, s)][k] is not None and idx[(ref, c, s)][k] is not None])
            if len(d):
                print(f"{arm} vs {ref} | {lab} | n {len(d)} | mean {d.mean():+.3f} | se {d.std(ddof=1) / np.sqrt(len(d)):.3f} | better {(d > 0).sum()}/{len(d)}")
g = json.loads((RUN / (a.guard or f"guard{suffix}.json")).read_text())
for arm in [BASELINE] + a.refs.split(",")[1:] + a.arms.split(","):
    if arm not in g:
        continue
    top = list(g[arm].get("top_names_blocks7_11_late", {})) or ["Florence Pugh"]
    for kind, cs in (("group", GRP), ("solo", ["noir", "viking"])):
        rs = g[arm]["rows"][kind]
        print(arm, kind, "her-face vs top names", {n: {c: round(float(np.mean([r["ref"][n] for r in rs if r["case"] == c])), 3) for c in cs} for n in top})
