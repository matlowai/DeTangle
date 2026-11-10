#!/usr/bin/env python3
"""Is the space-station default face specifically the model's Sigourney Weaver concept? (group S of probes.py, Base only)

  python actress_overlap.py      # -> runs/2026-10-19-entanglement-probes/RESULTS_S.md
Each actress's concept = normalized mean ArcFace embedding of her 4 studio-portrait renders. For each default-woman scene (4 seeds),
mean cosine to every concept; "Weaver excess" = cosine to Weaver minus the mean cosine to the four control actresses. The same is
reported for the E1 space-station renders (Base, original LoRA, A-side surgery @1.0/1.25, triggered) to see what the surgery falls back to.
"""
import json, statistics, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probes  # noqa: E402

sys.path.insert(0, str(probes.OLD))
import score_and_select  # noqa: E402

app, _, _, _ = score_and_select.setup()
rows = json.loads((probes.RUN / "render" / "manifest.json").read_text())


def emb(path):
    faces = app.get(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding


def embs(pred):
    return [e for e in (emb(r["file"]) for r in rows if pred(r)) if e is not None]


names = [n.split()[1].lower() for n in probes.ACTRESSES]
concept, self_sim, space_keep = {}, {}, {}
for n in names:
    studio = embs(lambda r, n=n: r["group"] == "S" and r["prompt_id"] == f"S_{n}_studio")
    c = np.mean(studio, axis=0)
    concept[n] = c / np.linalg.norm(c)
    self_sim[n] = statistics.fmean(float(a @ b) for i, a in enumerate(studio) for b in studio[i + 1:])
    space_keep[n] = statistics.fmean(float(e @ concept[n]) for e in embs(lambda r, n=n: r["group"] == "S" and r["prompt_id"] == f"S_{n}_space"))

lines = ["# Is the space-station default face the model's Sigourney Weaver? (Base, Krea 2 Turbo)", "",
         "Concept = mean ArcFace embedding of 4 studio portraits rendered by name. Cosines: ~0.9 same rendered identity, <0.3 unrelated.", "",
         "| Actress | studio self-consistency | her space-station render vs her concept |", "|---|---|---|"]
lines += [f"| {n} | {self_sim[n]:.2f} | {space_keep[n]:.2f} |" for n in names]
groups = [("default woman · space station (film still)", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_space"),
          ("default woman · space station, no film-still sentence", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_space_nofilm"),
          ("default woman · arcade + film-still sentence", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_arcade_film"),
          ("default woman · arcade", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_arcade"),
          ("default woman · beach", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_beach"),
          ("default woman · studio", lambda r: r["group"] == "S" and r["prompt_id"] == "S_default_studio"),
          ("E1 space station · Base (trigger word, no LoRA)", lambda r: r["group"] == "E1" and r["prompt_id"] == "V0" and r["arm"] == "base" and r["role"] == "target"),
          ("E1 space station · original LoRA, triggered", lambda r: r["group"] == "E1" and r["prompt_id"] == "V0" and r["arm"] == "source" and r["role"] == "target"),
          ("E1 space station · A-side @1.0, triggered", lambda r: r["group"] == "E1" and r["prompt_id"] == "V0" and r["arm"] == "aside100" and r["role"] == "target"),
          ("E1 space station · A-side @1.25, triggered", lambda r: r["group"] == "E1" and r["prompt_id"] == "V0" and r["arm"] == "aside125" and r["role"] == "target")]
lines += ["", "| Faces | n | " + " | ".join(names) + " | Weaver excess |", "|---|---|" + "---|" * (len(names) + 1)]
for label, pred in groups:
    es = embs(pred)
    sims = {n: statistics.fmean(float(e @ concept[n]) for e in es) for n in names}
    excess = sims["weaver"] - statistics.fmean(sims[n] for n in names if n != "weaver")
    lines.append(f"| {label} | {len(es)} | " + " | ".join(f"{sims[n]:.2f}" for n in names) + f" | {excess:+.2f} |")
(probes.RUN / "RESULTS_S.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
