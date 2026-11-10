#!/usr/bin/env python3
"""Non-identity side effects of the repair: does it keep the prompt's outfit, background and style, and the background blur?

Per render (local models only: CLIP ViT-L/14 from ComfyUI/models/clip, DINOv2-base from the HF cache):
  outfit / background / style: CLIP cosine between the image and the prompt's own outfit, background and closing style sentences
  snapshot: CLIP(image, "flat amateur snapshot") - CLIP(image, "cinematic film still with dramatic lighting")
  bg_sharp: log variance of the Laplacian outside the person region (above and beside an expanded face box), minus Base's
  scene_to_base / scene_to_original: DINOv2 cosine with the face region greyed out, against Base and the original LoRA (same prompt, seed)
  python side_effects.py   # -> runs/2026-10-20-overnight/side_effects.json + SIDE_EFFECTS.md
"""
import json, os, re, statistics as st, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overnight  # noqa: E402

RUN = overnight.RUN
CLIP_DIR = Path(os.environ.get("DETANGLE_CLIP_DIR", "clip-vit-large-patch14"))  # local CLIP ViT-L/14 (Hugging Face format)
# (step, arm, role) -> column label; genres use seeds 181901-04 and 181905-08, scenes use the confirm / fresh runs
ARMS = {("genres", "base", "generic"): "Base", ("genres", "source", "target"): "original", ("genres", "aside", "target"): "A-side",
        ("genres", "b12_15_35_amp6", "target"): "repair x6",
        ("fresh_genres", "base", "generic"): "Base", ("fresh_genres", "source", "target"): "original", ("fresh_genres", "aside", "target"): "A-side",
        ("fresh_genres", "win6", "target"): "repair x6", ("fresh_genres", "win8", "target"): "repair x8", ("fresh_fill", "base", "target"): "Base + rlsa",
        ("confirm", "base", "generic"): "Base", ("confirm", "source", "target"): "original", ("confirm2", "b12_15_35_amp6", "target"): "repair x6",
        ("amp10", "b35_amp10", "target"): "repair x10", ("fresh_scenes", "base", "generic"): "Base", ("fresh_scenes", "source", "target"): "original",
        ("fresh_scenes", "aside", "target"): "A-side", ("fresh_scenes", "win6", "target"): "repair x6", ("fresh_scenes", "win8", "target"): "repair x8",
        ("blend", "base", "generic"): "Base", ("blend", "source", "target"): "original", ("blend", "aside", "target"): "A-side", ("blend", "amp4", "target"): "amp x4 only",
        ("blend", "b25", "target"): "blend 8-15 25% only", ("blend", "b25_amp4", "target"): "blend 25% + amp x4",
        ("genres_decomp", "amp6_only", "target"): "amp x6 only", ("genres_decomp", "blend_only", "target"): "blend 12-15 35% only",
        ("style_variants", "nostyle6", "target"): "repair nostyle x6", ("style_variants", "nostyle8", "target"): "repair nostyle x8",
        ("style_variants", "trigtok6", "target"): "repair trigtok x6", ("style_variants", "nostyle6_style6", "target"): "nostyle x6 + style x6",
        ("style_sweep", "nostyle6_style2", "target"): "nostyle x6 + style x2", ("style_sweep", "nostyle6_style3", "target"): "nostyle x6 + style x3",
        ("style_sweep", "nostyle8_style2", "target"): "nostyle x8 + style x2", ("style_sweep", "nostylespec6", "target"): "nostyle-spec x6", ("style_sweep", "nostylespec8", "target"): "nostyle-spec x8",
        ("identity_amps", "id6", "target"): "identity amp x6", ("identity_amps", "idnc6", "target"): "identity-minus-clothing x6",
        ("identity_amps", "idnc20", "target"): "identity-minus-clothing x20", ("identity_amps", "idnc40", "target"): "identity-minus-clothing x40",
        ("grad_cands", "nostyle6", "target"): "gc style 0", ("grad_cands", "nostyle6_style2", "target"): "gc style x2",
        ("grad_cands", "nostyle6_spec2", "target"): "gc style_spec x2", ("grad_cands", "nostyle6_pca4x2", "target"): "gc pca4 x2",
        ("capture_genres", "win6", "target"): "cap: repair x6", ("capture_genres", "cap6", "target"): "cap: genres-amp x6", ("capture_genres", "cap8", "target"): "cap: genres-amp x8",
        ("block_xr", "b50_nostyle6", "target"): "bx: blend 50 + style 0", ("block_xr", "b65_nostyle6", "target"): "bx: blend 65 + style 0", ("block_xr", "b100_nostyle6", "target"): "bx: blend 100 + style 0",
        ("capture_genres2", "cap2_6", "target"): "cap2: genres2-amp x6", ("capture_style", "g2ns6", "target"): "cs: genres2 style 0",
        ("capture_style", "g2ns6_s2", "target"): "cs: genres2 style x2", ("capture_style", "g2ns6_s3", "target"): "cs: genres2 style x3",
        ("sched_server", "static3", "target"): "ss: static style x3", ("sched_server", "hook_all3", "target"): "ss: hook all-steps x3",
        ("sched_server", "hook_ml6", "target"): "ss: hook steps 3-7 x6", ("sched_server", "hook_late6", "target"): "ss: hook steps 6-7 x6"}


def sentences(prompt):
    s = [x.strip() for x in re.split(r"(?<=\.)\s+", prompt.removeprefix("rlsa, ")) if x.strip()]
    outfit = next((x for x in s if " wears " in x or " wearing " in x), None)
    bg = next((x for x in s if "behind her" in x), None)
    return outfit, bg, s[-1]


def main():
    from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
    dino = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(dev).eval()
    text_cache = {}

    def text_emb(t):
        if t not in text_cache:
            with torch.no_grad():
                e = clip.text_projection(clip.text_model(**proc(text=[t], return_tensors="pt", padding=True, truncation=True).to(dev)).pooler_output)
            text_cache[t] = torch.nn.functional.normalize(e, dim=-1)[0]
        return text_cache[t]

    rows = []
    for step in dict.fromkeys(k[0] for k in ARMS):
        if not (RUN / step / "scores.json").exists():
            continue
        scores = {r["id"]: r for r in json.loads((RUN / step / "scores.json").read_text())}
        for m in json.loads((RUN / step / "render/manifest.json").read_text()):
            label = ARMS.get((step, m["arm"], m["role"]))
            if label and scores[m["id"]]["faces"]:
                rows.append({"step": step, "scene": m["prompt_id"], "seed": m["seed"], "arm": label, "file": m["file"], "prompt": m["text"],
                             "face": max(scores[m["id"]]["faces"], key=lambda f: f["area"])["bbox"]})
    snap, cine = text_emb("a flat amateur smartphone snapshot"), text_emb("a cinematic film still with dramatic lighting")
    for r in rows:
        im = Image.open(r["file"]).convert("RGB")
        W, H = im.size
        x0, y0, x1, y1 = r["face"]
        w, h = x1 - x0, y1 - y0
        box = (max(0, x0 - 0.6 * w), max(0, y0 - 0.5 * h), min(W, x1 + 0.6 * w), H)  # person region: face, hair, shoulders and below
        g = np.asarray(im.convert("L"), dtype=np.float32)
        lap = np.abs(np.asarray(Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=np.float32))
        mask = np.ones_like(g, bool)
        mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = False
        if mask.mean() < 0.05:  # the person spans the frame: use the band above the hair
            mask[:] = False
            mask[:max(int(box[1]), int(0.12 * H))] = True
        r["bg_sharp_raw"] = float(np.log(lap[mask].var() + 1e-6))
        with torch.no_grad():
            e = torch.nn.functional.normalize(clip.visual_projection(clip.vision_model(**proc(images=im, return_tensors="pt").to(dev)).pooler_output), dim=-1)[0]
            outfit, bg, style = sentences(r["prompt"])
            r["outfit"] = float(e @ text_emb(outfit)) if outfit else None
            r["background"] = float(e @ text_emb(bg)) if bg else None
            r["style"] = float(e @ text_emb(style))
            r["snapshot"] = float(e @ snap - e @ cine)
            masked = im.copy()
            masked.paste((128, 128, 128), (int(x0 - 0.15 * w), int(y0 - 0.15 * h), int(x1 + 0.15 * w), int(y1 + 0.1 * h)))
            d = dino(**dino_proc(images=masked, return_tensors="pt").to(dev)).pooler_output
            r["dino"] = torch.nn.functional.normalize(d, dim=-1)[0].cpu().numpy().tolist()
    ref = {(r["scene"], r["seed"], r["arm"]): r for r in rows}
    for r in rows:
        b, o = ref.get((r["scene"], r["seed"], "Base")), ref.get((r["scene"], r["seed"], "original"))
        r["bg_sharp"] = r["bg_sharp_raw"] - b["bg_sharp_raw"] if b else None
        r["scene_to_base"] = float(np.dot(r["dino"], b["dino"])) if b else None
        r["scene_to_original"] = float(np.dot(r["dino"], o["dino"])) if o else None
    for r in rows:
        r.pop("dino")
    (RUN / "side_effects.json").write_text(json.dumps(rows, indent=1))
    genres = {"noir", "victorian", "western", "superhero", "viking", "samurai"}
    order = ["Base", "Base + rlsa", "original", "A-side", "amp x4 only", "blend 8-15 25% only", "blend 25% + amp x4", "amp x6 only", "blend 12-15 35% only", "repair x6", "repair nostyle x6", "repair nostyle x8", "repair trigtok x6", "nostyle x6 + style x6", "nostyle x6 + style x2", "nostyle x6 + style x3", "nostyle x8 + style x2", "nostyle-spec x6", "nostyle-spec x8", "identity amp x6", "identity-minus-clothing x6", "identity-minus-clothing x20", "identity-minus-clothing x40", "repair x8", "repair x10"]
    keys = ["outfit", "background", "style", "snapshot", "bg_sharp", "scene_to_base", "scene_to_original"]
    L = ["# Side effects of the repair beyond identity", "",
         "CLIP cosine of the image with the prompt's own outfit / background / style sentences (higher = closer to the prompt); snapshot = CLIP(\"flat amateur smartphone snapshot\") − CLIP(\"cinematic film still with dramatic lighting\") (higher = more snapshot-like); "
         "bg_sharp = log Laplacian variance outside the person region minus Base's (positive = sharper background than Base); scene_to_* = DINOv2 cosine with the face greyed out (1 = same scene/costume/composition). Means over all renders per arm.", ""]
    for title, pick in (("Six unseen genres (seeds 181901–08)", lambda r: r["scene"] in genres), ("Nine protocol scenes", lambda r: r["scene"] not in genres and r["step"] != "blend"),
                        ("Decomposition: space station + beach, seeds 181901–02 (blend step)", lambda r: r["step"] == "blend" and r["scene"] in ("space_station", "beach"))):
        L += [f"## {title}", "", "| arm | n | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        for arm in order:
            rs = [r for r in rows if pick(r) and r["arm"] == arm]
            if rs:
                L.append(f"| {arm} | {len(rs)} | " + " | ".join(f"{st.fmean(v):.3f}" if (v := [r[k] for r in rs if r[k] is not None]) else "–" for k in keys) + " |")
        L.append("")
    L += ["## Per genre (outfit / background / snapshot / bg_sharp / scene_to_original)", "", "| genre | original | A-side | repair x6 | repair x8 |", "|---|---|---|---|---|"]
    for g in sorted(genres):
        cells = []
        for arm in ("original", "A-side", "repair x6", "repair x8"):
            rs = [r for r in rows if r["scene"] == g and r["arm"] == arm]
            f = lambda k: st.fmean(v) if (v := [r[k] for r in rs if r[k] is not None]) else float("nan")
            cells.append(f"{f('outfit'):.3f} / {f('background'):.3f} / {f('snapshot'):+.3f} / {f('bg_sharp'):+.2f} / {f('scene_to_original'):.2f}" if rs else "–")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    (RUN / "SIDE_EFFECTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
