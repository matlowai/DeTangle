#!/usr/bin/env python3
"""Which famous identities make up the model's space-station woman, and what does A-side surgery fall back to?

  python actress_panel.py      # -> runs/2026-10-19-entanglement-probes/RESULTS_PANEL.md
Concepts: normalized mean ArcFace embedding of 4 studio portraits per actress (ACTRESSES + PANEL, Base only).
Per face group: (1) cosine to every concept; (2) a non-negative least-squares fit of the face embeddings onto the concept embeddings
(weights normalized to sum 1, with the fit's cosine), i.e. which concepts the face is "made of" in identity space. Embedding-space
attribution only: it describes resemblance, not what the model internally computes.
"""
import json, statistics, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probes  # noqa: E402

sys.path.insert(0, str(probes.OLD))
import score_and_select  # noqa: E402

app, centroid, _, _ = score_and_select.setup()
rows = json.loads((probes.RUN / "render" / "manifest.json").read_text())
cache = {}


def emb(path):
    if path not in cache:
        faces = app.get(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        cache[path] = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding if faces else None
    return cache[path]


def embs(pred):
    return [e for e in (emb(r["file"]) for r in rows if pred(r)) if e is not None]


people = {n: f"S_{n.split()[1].lower()}_studio" for n in probes.ACTRESSES}
people.update({n: "P_" + n.replace(" ", "").lower() for n in probes.PANEL})
C = {}
for name, pid in people.items():
    es = embs(lambda r, pid=pid: r["group"] == "S" and r["prompt_id"] == pid)
    c = np.mean(es, axis=0)
    C[name] = c / np.linalg.norm(c)
names = list(C)
M = np.stack([C[n] for n in names], axis=1)  # 512 x K

groups = [("Base default woman · space station", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_space"),
          ("Base default woman · space station, no film-still", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_space_nofilm"),
          ("Base default woman · arcade + film-still", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_arcade_film"),
          ("Base default woman · beach", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_beach"),
          ("Base default woman · studio", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_studio"),
          ("Character LoRA original · space station", lambda r: r["group"] == "E1" and r["prompt_id"] == "V0" and r["arm"] == "source" and r["role"] == "target"),
          ("Character LoRA after A-side surgery · space station", lambda r: r["group"] == "E1" and r["prompt_id"] == "V0" and r["arm"] in ("aside100", "aside125") and r["role"] == "target"),
          ("Character LoRA after A-side surgery · wizard workshop", lambda r: r["group"] == "E1" and r["prompt_id"] == "wizard" and r["arm"] in ("aside100", "aside125") and r["role"] == "target")]
lines = ["# Identity panel: what the space-station woman is made of (Base Krea 2 Turbo, ArcFace)", "",
         f"{len(names)} concepts (4 studio portraits each, rendered by name). Reference points: different actresses' concepts score 0.03–0.38 against each other; "
         "a single render against another actress averages 0.15; an actress's own space-station render scores 0.74–0.85 against her concept.", "",
         "Top resemblances (mean cosine to each concept) and a non-negative mix fit (share of the fitted weight; fit = cosine between the mix and the faces).", ""]
for label, pred in groups:
    es = embs(pred)
    if not es:
        continue
    sims = sorted(((statistics.fmean(float(e @ C[n]) for e in es), n) for n in names), reverse=True)
    target = np.mean(es, axis=0)
    w, _ = nnls(M, target)
    fit = M @ w
    fitcos = float(fit @ target / (np.linalg.norm(fit) * np.linalg.norm(target)))
    mix = sorted(((wi / w.sum(), n) for wi, n in zip(w, names) if wi > 0), reverse=True)
    char = statistics.fmean(float(e @ centroid) for e in es)
    lines += [f"**{label}** (n={len(es)}; character likeness {char:.2f})", "",
              "- closest: " + ", ".join(f"{n} {v:.2f}" for v, n in sims[:6]),
              f"- mix (fit {fitcos:.2f}): " + ", ".join(f"{n} {v:.0%}" for v, n in mix[:6]), ""]
(probes.RUN / "RESULTS_PANEL.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
