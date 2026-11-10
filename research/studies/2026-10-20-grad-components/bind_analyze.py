#!/usr/bin/env python3
"""Binding heads from bind/bind.json.
  binding score (block, head) = mean over two-celebrity renders and their two faces of
      attention(face -> own name span) - attention(face -> other name span)       (late steps 5 and 7)
  Faces are identified by ArcFace against the celebrity centroids. Then, in the top heads, the rlsa scenes:
  attention of Subject A's face vs the other faces to the rlsa span (and to the other person's span)."""
import json
import os
from pathlib import Path
import numpy as np

RUN = Path(os.environ.get("DETANGLE_RESEARCH_ROOT", "detangle-research")).resolve() / "runs/2026-10-20-grad-components/bind"
LATE = (5, 7)


def main():
    rows = json.loads((RUN / "bind.json").read_text())
    pairs = [r for r in rows if r["kind"] == "pair"]
    score, count = {}, {}
    for r in pairs:
        a, b = r["names"]["A"], r["names"]["B"]
        faces = r["faces"]
        ia = int(np.argmax([f["scores"].get(a, -1) for f in faces]))
        ib = int(np.argmax([f["scores"].get(b, -1) for f in faces]))
        if ia == ib:
            continue
        for at in r["attn"]:
            if at["step"] not in LATE:
                continue
            fa, fb = at["faces"][ia], at["faces"][ib]
            s = (np.array(fa["A"]) - np.array(fa["B"]) + np.array(fb["B"]) - np.array(fb["A"])) / 2
            key = at["block"]
            score[key] = score.get(key, 0) + s
            count[key] = count.get(key, 0) + 1
    S = {b: score[b] / count[b] for b in score}
    flat = sorted(((float(v[h]), b, h) for b, v in S.items() for h in range(len(v))), reverse=True)
    out = {"n_pair_renders": len(pairs), "top_binding_heads": [{"block": b, "head": h, "own_minus_other": round(v, 4)} for v, b, h in flat[:15]],
           "per_block_max": {b: round(float(v.max()), 4) for b, v in sorted(S.items())}, "rlsa": {}}
    top = [(b, h) for _, b, h in flat[:10]]
    for r in [x for x in rows if x["kind"].startswith("rlsa_")]:
        faces = r["faces"]
        her = int(np.argmax([f["scores"]["her"] for f in faces]))
        rec = {"seed": r["seed"], "her_likeness_per_face": [round(f["scores"]["her"], 3) for f in faces], "her_face": her}
        for span in r["names"]:
            vals = np.zeros(len(faces))
            n = 0
            for at in r["attn"]:
                if at["step"] in LATE:
                    for b, h in top:
                        if at["block"] == b:
                            vals += np.array([f[span][h] for f in at["faces"]])
                            n += 1
            rec[f"attn_to_{span}_top_heads"] = [round(v, 4) for v in (vals / max(n, 1))]
        out["rlsa"].setdefault(r["kind"], []).append(rec)
    (RUN / "bind_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
