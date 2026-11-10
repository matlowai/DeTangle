#!/usr/bin/env python3
"""Iterative identity peeling on Base Krea 2, round 1: Sigourney Weaver (next step 2).

Edit: closed-form least squares on the last text projection `txtmlp.3` (bf16, 6144x6144). Text tokens reach the image blocks only
through txtmlp(txtfusion(context)), so the edit is prompt-conditional by construction and leaves image-only computation alone.
  source (generic sci-fi woman prompts): c'_t = c_t - (c_twin - c)_t        (twin = same prompt naming the identity, token-aligned)
  preserve: every panel actress named in sci-fi/studio/beach scenes, generic women in non-sci-fi scenes, men, non-person prompts.
  dW = -(sum D H^T)(G_src + lam G_pre + eps I)^-1 ; LoRA strength scales the offset (fit at alpha 1).
Arms: anti-Weaver (raw twin offset); control anti-Streep (same construction, an actress outside the space prior).

  python peel.py fit       # GPU0 standalone: encode prompts, capture txtmlp.3 in/out, solve, conditioning-space metrics, write adapters
  python peel.py prepare   # cells.json
  python peel.py render    # isolated ComfyUI on GPU0 via probes.render (second LoRA node for the character check)
  python peel.py score     # ArcFace: 19-concept panel, NNLS mix, character likeness, face match to Base same seed
  python peel.py report    # RESULTS.md
"""
import argparse, json, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-19-entanglement-probes"))
import probes  # noqa: E402
import common  # noqa: E402  (on sys.path via probes)

RUN = common.RUNS / "2026-10-19-peel"
MODELS = Path(os.environ.get("DETANGLE_COMFY_MODELS", str(probes.COMFY / "models")))
SEEDS = probes.SEEDS_S
ANCHOR = "photograph of a woman"
FILM = probes.FILM
PANEL = probes.ACTRESSES + probes.PANEL  # 19 concept identities
LAMS, EPS = (1.0, 4.0), 1e-4

FACE = "Her face is large in the frame and unobstructed, with natural skin texture and both eyes sharply in focus."
NEW_SCIFI = {
    "N1": f"A close-up head-and-shoulders photograph of a woman in the pilot seat of a spaceship cockpit, facing the camera. {FACE} She is wearing a gray padded flight jacket zipped to the collar. Glowing control panels and a starfield through the canopy are softly blurred behind her. {FILM}",
    "N2": f"A close-up head-and-shoulders photograph of a woman inside a Mars habitat module, facing the camera with a determined expression. {FACE} She wears a rust-orange utility suit with a high collar. A curved window showing red dunes is softly blurred behind her. {FILM}",
    "N3": f"A close-up head-and-shoulders photograph of a woman standing beside a row of frosted cryo-sleep pods, facing the camera. {FACE} She wears a white crew jumpsuit with a closed collar. Cold blue light fills the corridor behind her. {FILM}",
    "N4": f"A close-up head-and-shoulders photograph of a woman on an alien desert planet, facing the camera, helmet off. {FACE} She wears a scuffed white spacesuit with a high neck ring. Two pale moons hang in the hazy sky behind her. {FILM}",
    "N5": f"A close-up head-and-shoulders photograph of a woman on the bridge of a starship, facing the camera with a calm expression. {FACE} She wears a dark navy uniform with a high collar and a small silver insignia. Holographic displays glow softly out of focus behind her. {FILM}",
    "N6": f"A close-up head-and-shoulders photograph of a woman in a zero-gravity orbital laboratory, facing the camera. {FACE} She wears a blue crew polo shirt buttoned to the collar. Floating equipment and a porthole showing Earth are softly blurred behind her. {FILM}",
}
OTHER = ["A wide landscape photograph of misty pine forests and a calm mountain lake at dawn.",
         "A still life photograph of ceramic bowls, lemons and a linen cloth on a wooden table.",
         "A street photograph of a rainy city crossing at night with neon reflections on wet asphalt.",
         "A close-up photograph of a red vintage motorcycle parked in front of a brick wall."]
# (place clause, outfit, background); the last HELD_SF / HELD_EV places are held out of the fit entirely
SF_PLACES = [("aboard a space station", "a gray crew jumpsuit zipped to the collar", "A row of round viewports showing a starfield is softly blurred behind"),
             ("in a lunar base corridor", "a white thermal undersuit with a high collar", "Pressurized doors and warning lights are softly blurred behind"),
             ("in an asteroid mining outpost", "a scuffed orange work suit with a closed collar", "Drilling machinery and floodlights are softly blurred behind"),
             ("in a spacecraft airlock", "a silver pressure suit with the helmet removed", "A circular hatch with blinking status lights is softly blurred behind"),
             ("in the mess hall of a deep-space freighter", "a worn olive flight jacket over a gray shirt", "Steel tables and dim overhead lamps are softly blurred behind"),
             ("in a starship medical bay", "a pale blue medical uniform with a high collar", "Glowing diagnostic screens are softly blurred behind"),
             ("on an icy exoplanet", "a white insulated expedition suit with a high neck", "Blue glaciers under a ringed planet are softly blurred behind"),
             ("in a space shuttle cockpit", "a navy flight suit with mission patches", "Switch panels and a curved windshield showing Earth are softly blurred behind"),
             ("in a futuristic orbital hangar", "a black pilot suit with a high collar", "Docked fighter craft and catwalks are softly blurred behind"),
             ("inside a generation ship hydroponics deck", "a green crew coverall buttoned to the collar", "Rows of plants under violet grow lights are softly blurred behind"),
             ("on the surface of Mars", "a dusty white spacesuit with the helmet off", "A rover and red rocky hills are softly blurred behind"),
             ("in a space station control room", "a dark gray uniform with a closed collar", "Banks of monitors and a large window onto a nebula are softly blurred behind"),
             ("in an alien jungle base camp", "a khaki survey suit with a high collar", "Bioluminescent plants and a landed dropship are softly blurred behind"),
             ("in a derelict spaceship corridor", "a battered tan flight suit zipped to the collar", "Flickering emergency lights and pipes are softly blurred behind"),
             ("in a space elevator observation lounge", "a cream crew sweater with a high neck", "The curve of Earth and thin clouds are softly blurred behind"),
             ("in a moon colony greenhouse dome", "a light gray work suit with a closed collar", "A glass dome showing the black lunar sky is softly blurred behind")]
EV_PLACES = [("in a cozy cafe", "a cream knit sweater with a high neck", "Wooden tables and a chalkboard menu are softly blurred behind"),
             ("in a quiet library", "a dark green cardigan buttoned to the collar", "Tall bookshelves are softly blurred behind"),
             ("in a modern office", "a charcoal blazer over a white shirt buttoned to the collar", "Glass walls and desks are softly blurred behind"),
             ("in a sunny kitchen", "a striped cotton shirt buttoned to the collar", "Open shelves with plates and plants are softly blurred behind"),
             ("in a city park", "a camel wool coat with a high collar", "Autumn trees and a path are softly blurred behind"),
             ("on a mountain trail", "a red hiking jacket zipped to the collar", "Pine trees and distant peaks are softly blurred behind"),
             ("on a busy city street", "a black trench coat with a high collar", "Storefronts and traffic are softly blurred behind"),
             ("in a subway station", "a gray hooded jacket zipped to the collar", "A tiled wall and a departing train are softly blurred behind"),
             ("in an art museum", "a navy turtleneck sweater", "Framed paintings on white walls are softly blurred behind"),
             ("in a flower garden", "a pale yellow blouse buttoned to the collar", "Rose bushes and a stone wall are softly blurred behind"),
             ("in a bakery", "a white apron over a blue shirt buttoned to the collar", "Shelves of bread are softly blurred behind"),
             ("on a farm", "a plaid flannel shirt buttoned to the collar", "A red barn and fields are softly blurred behind"),
             ("on a lake dock", "a light blue rain jacket zipped to the collar", "Still water and a forested shore are softly blurred behind"),
             ("in a ski lodge", "a patterned wool sweater with a high neck", "A stone fireplace and wooden beams are softly blurred behind"),
             ("in a farmers market", "a denim jacket over a white shirt buttoned to the collar", "Fruit stalls and awnings are softly blurred behind"),
             ("in a school classroom", "a beige cardigan over a collared shirt", "A chalkboard and desks are softly blurred behind"),
             ("in a hospital hallway", "light green scrubs with a closed collar", "Corridor doors and soft ceiling lights are softly blurred behind"),
             ("in a train station", "a brown leather jacket zipped to the collar", "Departure boards and arches are softly blurred behind"),
             ("on a city rooftop at dusk", "a black wool coat with a high collar", "City lights and a pink sky are softly blurred behind"),
             ("in a vineyard", "a white linen shirt buttoned to the collar", "Rows of grapevines and hills are softly blurred behind"),
             ("in a fishing harbor", "a yellow oilskin jacket zipped to the collar", "Fishing boats and nets are softly blurred behind"),
             ("in a theater dressing room", "a dark red velvet blazer over a black top with a high neck", "A mirror ringed with bulbs is softly blurred behind"),
             ("in an artist's studio", "a paint-flecked denim shirt buttoned to the collar", "Canvases and easels are softly blurred behind"),
             ("in a laundromat", "an oversized gray sweatshirt with a crew neck", "Rows of washing machines are softly blurred behind"),
             ("in a medieval castle hall", "a dark blue wool gown with a high collar", "Stone arches and tapestries are softly blurred behind"),
             ("in a 1920s jazz club", "a black beaded dress with a high neckline", "A brass band on a small stage is softly blurred behind"),
             ("in a desert canyon", "a sand-colored field shirt buttoned to the collar", "Red rock walls are softly blurred behind"),
             ("in a busy newsroom", "a white blouse buttoned to the collar under a gray vest", "Desks with monitors and papers are softly blurred behind"),
             ("in a snowy village street", "a burgundy wool coat with a high collar", "Wooden houses and falling snow are softly blurred behind"),
             ("in a hotel lobby", "a cream silk blouse buttoned to the collar", "Marble columns and a chandelier are softly blurred behind")]
HELD_SF, HELD_EV = 4, 8
EXPR = ["", " with a calm expression", " with a relaxed neutral expression", " with a slight smile", " with a determined expression"]
SF_STYLE = [FILM, "Cinematic science-fiction movie still.", "Photograph.", "Soft cinematic light illuminates her face."]
EV_STYLE = ["Photorealistic live-action drama film still.", "Photorealistic live-action period film still.", "Documentary photograph.", "Photograph.", "Soft cinematic light illuminates her face."]


def gen(places, styles, seed, per=3, man=False):
    rng = random.Random(seed)
    out = []
    for place, outfit, bg in places:
        for _ in range(per):
            t = f"A close-up head-and-shoulders photograph of a woman {place}, facing the camera{rng.choice(EXPR)}. {FACE} She wears {outfit}. {bg} her. {rng.choice(styles)}"
            if man:
                t = t.replace("a woman", "a man").replace("Her face", "His face").replace("She wears", "He wears").replace("behind her", "behind him").replace("her face", "his face")
            out.append(t)
    return out


def name(text, who):
    assert ANCHOR in text, text[:80]
    return text.replace(ANCHOR, "photograph of " + who, 1)


def prompt_sets():
    v = probes.prompts()
    p = common.read_plan()["prompts"]
    scifi_fit = {k: v[k] for k in ("V0", "V2", "V3", "V4", "V5", "V6")} | {k: NEW_SCIFI[k] for k in ("N1", "N2", "N3", "N4")}
    scifi_fit |= {f"G{i}": t for i, t in enumerate(gen(SF_PLACES[:-HELD_SF], SF_STYLE, 1))}
    scifi_held = {k: v[k] for k in ("V1", "V7", "V8")} | {k: NEW_SCIFI[k] for k in ("N5", "N6")}
    scifi_held |= {f"H{i}": t for i, t in enumerate(gen(SF_PLACES[-HELD_SF:], SF_STYLE, 2, per=2))}
    scenes = {s: p[s + "_generic"]["text"] for s in ("calibration_studio", "beach", "arcade", "wizard_workshop", "ferry_terminal", "planetarium", "concert_hall", "greenhouse")}
    general_pre = [t for k, t in scenes.items() if k != "greenhouse"] + gen(EV_PLACES[:-HELD_EV], EV_STYLE, 3)
    general_held = [scenes["greenhouse"]] + gen(EV_PLACES[-HELD_EV:], EV_STYLE, 4, per=2)
    men = gen(SF_PLACES[:-HELD_SF], SF_STYLE, 5, per=1, man=True) + gen(EV_PLACES[:-HELD_EV], EV_STYLE, 6, per=1, man=True)
    return scifi_fit, scifi_held, scenes, general_pre, general_held, men


def fit(specific=False, names=("Sigourney Weaver", "Meryl Streep"), protect=None):
    """protect: Gram file of triggered-prompt txtmlp.3 inputs captured under the character adapter; added to the preserve term."""
    import torch
    sys.path.insert(0, str(probes.COMFY))
    import comfy.sd, comfy.model_management as mm
    from safetensors.torch import save_file
    dev = mm.get_torch_device()
    clip = comfy.sd.load_clip(ckpt_paths=[str(MODELS / "text_encoders/krea2/qwen3vl_4b_fp8_scaled.safetensors")], clip_type=comfy.sd.CLIPType.KREA2)
    unet = comfy.sd.load_diffusion_model(str(MODELS / "diffusion_models/krea2/krea2_turbo_fp8_scaled.safetensors"))
    dm = unet.model.diffusion_model
    cache = {}

    def enc(text, keep=True):
        """(token ids after the stripped template, H = txtmlp.3 input, C = output), fp32 on CPU."""
        if text in cache:
            return cache[text]
        tok = clip.tokenize(text)
        ids = [t[0] for t in tok["qwen3vl_4b"][0]]
        cond = clip.encode_from_tokens(tok)
        mm.load_models_gpu([unet])
        with torch.no_grad():
            x = dm.txtfusion(dm._unpack_context(cond.to(dev, torch.bfloat16)))
            h = dm.txtmlp[:3](x)
            c = dm.txtmlp[3](h)
        out = (ids[len(ids) - cond.shape[1]:], h[0].float().cpu(), c[0].float().cpu())
        if keep:
            cache[text] = out
        return out

    def offset(text, who):
        """Token-aligned twin offset c_twin - c for the generic prompt's tokens."""
        ids, _, c = enc(text)
        tids, _, ct = enc(name(text, who))
        p = 0
        while ids[p] == tids[p]:
            p += 1
        s = 0
        while ids[-1 - s] == tids[-1 - s]:
            s += 1
        d = torch.empty_like(c)
        d[:p] = ct[:p] - c[:p]
        d[len(ids) - s:] = ct[len(tids) - s:] - c[len(ids) - s:]
        d[p:len(ids) - s] = ct[p:len(tids) - s].mean(0) - c[p:len(ids) - s]
        return d

    scifi_fit, scifi_held, scenes, general_pre, general_held, men = prompt_sets()
    rng = random.Random(7)
    others = [w for w in PANEL if w not in ("Sigourney Weaver", "Meryl Streep")]
    preserve = [name(t, w) for k, t in scifi_fit.items() for w in (PANEL if k.startswith("V") else ["Sigourney Weaver", "Meryl Streep"] + rng.sample(others, 4))]
    preserve += [name(t, w) for t in general_pre for w in rng.sample(PANEL, 3)]
    preserve += [name(t, w) for t in (scenes["calibration_studio"], scenes["beach"]) for w in PANEL]
    preserve += general_pre + men + OTHER
    preserve = list(dict.fromkeys(preserve))
    d = 6144
    G = torch.zeros(d, d, dtype=torch.float64, device=dev)
    rows_p = 0
    for i, t in enumerate(preserve):
        h = enc(t, keep=t.startswith("A close-up head-and-shoulders photograph of Sigourney Weaver"))[1].to(dev, torch.float64)
        G += h.T @ h
        rows_p += h.shape[0]
        if i % 100 == 0:
            print("preserve", i, "/", len(preserve), flush=True)
    Gs = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for t in scifi_fit.values():
        h = enc(t)[1].to(dev, torch.float64)
        Gs += h.T @ h
    memo = {}

    def target(t, who):
        """Raw twin offset, or (specific) minus the mean offset of the other 18 panel actresses: cancels the shared famous-actress part."""
        if (t, who) not in memo:
            D = offset(t, who)
            if specific:
                D = D - torch.stack([offset(t, k) for k in PANEL if k != who]).mean(0)
            memo[t, who] = D
        return memo[t, who]
    R = {}
    for who in names:
        R[who] = torch.zeros(d, d, dtype=torch.float64, device=dev)
        for t in scifi_fit.values():
            R[who] += target(t, who).to(dev, torch.float64).T @ enc(t)[1].to(dev, torch.float64)
    W = dm.txtmlp[3].weight.detach().float().cpu()
    report = {"eps": EPS, "n_preserve": len(preserve), "n_source": len(scifi_fit), "rows_preserve": rows_p,
              "rows_source": int(sum(enc(t)[1].shape[0] for t in scifi_fit.values())), "n_held_scifi": len(scifi_held), "n_held_general": len(general_held)}
    checks = {"held_general": general_held,
              "held_general_filmstill": [t for t in general_held if "film still" in t],
              "held_named_weaver_scifi": [name(t, "Sigourney Weaver") for t in scifi_held.values()],
              "held_named_ferguson_scifi": [name(t, "Rebecca Ferguson") for t in scifi_held.values()],
              "held_named_weaver_general": [name(t, "Sigourney Weaver") for t in general_held],
              "held_men_scifi": gen(SF_PLACES[-HELD_SF:], SF_STYLE, 8, per=1, man=True),
              "preserve_named_weaver": [name(t, "Sigourney Weaver") for t in scifi_fit.values()],
              "preserve_general": general_pre}
    (RUN / "adapters").mkdir(parents=True, exist_ok=True)
    Gp = torch.load(protect).to(dev) if protect else 0
    for lam in ((4.0,) if specific else LAMS):
        A = Gs + lam * (G + Gp)
        A += EPS * A.diagonal().mean() * torch.eye(d, dtype=torch.float64, device=dev)
        L = torch.linalg.cholesky(A)
        for who in R:
            tag = ("spec_" if specific else "anti_") + f"{who.split()[1].lower()}_l{lam:g}" + ("_tp" if protect else "")
            dW = -torch.cholesky_solve(R[who].T, L).T.float().cpu()  # -(R)(A)^-1, A symmetric
            Wr = (W + dW).to(torch.bfloat16).float() - W  # what ComfyUI's bf16 merge realizes at strength 1
            rec = {}
            for label, texts in (("fit_scifi", list(scifi_fit.values())), ("held_scifi", list(scifi_held.values()))):
                num = den = 0.0
                for t in texts:
                    h, D = enc(t)[1], target(t, who)
                    num += float(((h @ Wr.T) + D).pow(2).sum()); den += float(D.pow(2).sum())
                rec[label + "_achieved"] = 1 - num / den
            ref = [float(target(t, who).norm() / enc(t)[2].norm()) for t in scifi_fit.values()]
            rec["target_rel_offset"] = sum(ref) / len(ref)
            if specific:
                cos = [float(torch.nn.functional.cosine_similarity(target(t, who).flatten(), offset(t, who).flatten(), dim=0)) for t in scifi_fit.values()]
                rec["cos_to_raw_offset"] = sum(cos) / len(cos)
            for label, texts in checks.items():
                rel = [float((enc(t)[1] @ Wr.T).norm() / enc(t)[2].norm()) for t in texts]
                rec[label] = f"mean {sum(rel) / len(rel):.3f} max {max(rel):.3f} (n={len(rel)})"
            if protect:
                rec["triggered_rel_energy"] = float(torch.sqrt(torch.trace(Wr.to(dev, torch.float64) @ Gp @ Wr.to(dev, torch.float64).T) / torch.trace(W.to(dev, torch.float64) @ Gp @ W.to(dev, torch.float64).T)))
            report[tag] = rec
            save_file({"diffusion_model.txtmlp.3.diff": dW.contiguous()}, RUN / "adapters" / f"peel_{tag}.safetensors")
            print(tag, json.dumps(rec, indent=1), flush=True)
    suffix = ("" if names == ("Sigourney Weaver", "Meryl Streep") else "_" + "_".join(n.split()[1].lower() for n in names)) + ("_tp" if protect else "")
    common.write(RUN / (("fit_specific" if specific else "fit") + suffix + ".json"), report)


def cells():
    scifi_fit, scifi_held, scenes, *_ = prompt_sets()
    measure = {"sf_V0": scifi_fit["V0"], "sf_V1": scifi_held["V1"], "sf_N5": scifi_held["N5"], "sf_V8": scifi_held["V8"],
               "w_V0": name(scifi_fit["V0"], "Sigourney Weaver"), "w_N5": name(scifi_held["N5"], "Sigourney Weaver"),
               "w_studio": name(scenes["calibration_studio"], "Sigourney Weaver"), "f_N5": name(scifi_held["N5"], "Rebecca Ferguson"),
               "g_beach": scenes["beach"], "g_studio": scenes["calibration_studio"], "g_greenhouse": scenes["greenhouse"]}
    rows = []

    def add(pid, arm, adapters, role="generic"):
        for seed in SEEDS:
            rows.append({"id": f"R1_{pid}_{arm}_{role}_{seed}", "group": "R1", "prompt_id": pid, "arm": arm, "adapters": adapters, "role": role,
                         "seed": seed, "text": ("rlsa, " if role == "target" else "") + measure[pid]})
    for pid in measure:
        add(pid, "base", [])
        add(pid, "weaver100", [["peel_anti_weaver_l4.safetensors", 1.0]])
        add(pid, "weaver200", [["peel_anti_weaver_l4.safetensors", 2.0]])
    for pid in ("sf_V0", "sf_V1", "sf_N5", "sf_V8"):
        add(pid, "streep200", [["peel_anti_streep_l4.safetensors", 2.0]])
    add("sf_V0", "aside", [["a_side.safetensors", 1.0]], "target")
    for s in (1.0, 2.0):
        add("sf_V0", f"aside_weaver{int(s * 100)}", [["a_side.safetensors", 1.0], ["peel_anti_weaver_l4.safetensors", s]], "target")
    for pid in ("sf_V0", "sf_N5", "w_V0"):  # round 2: Weaver-specific direction (minus the other actresses' mean offset)
        add(pid, "spec200", [["peel_spec_weaver_l4.safetensors", 2.0]])
        add(pid, "spec400", [["peel_spec_weaver_l4.safetensors", 4.0]])
        add(pid, "streepspec400", [["peel_spec_streep_l4.safetensors", 4.0]])
    add("sf_V0", "aside_spec400", [["a_side.safetensors", 1.0], ["peel_spec_weaver_l4.safetensors", 4.0]], "target")
    add("sf_V0", "aside_spec200", [["a_side.safetensors", 1.0], ["peel_spec_weaver_l4.safetensors", 2.0]], "target")
    add("sf_V0", "aside_streepspec400", [["a_side.safetensors", 1.0], ["peel_spec_streep_l4.safetensors", 4.0]], "target")
    for s in (6.0, 8.0):  # round 2b: dose on the character LoRA, everyday drift at x4
        add("sf_V0", f"aside_spec{int(s * 100)}", [["a_side.safetensors", 1.0], ["peel_spec_weaver_l4.safetensors", s]], "target")
        add("sf_V0", f"spec{int(s * 100)}", [["peel_spec_weaver_l4.safetensors", s]])
    for pid in ("g_studio", "g_beach", "g_greenhouse"):
        add(pid, "spec400", [["peel_spec_weaver_l4.safetensors", 4.0]])
    return rows


def graph(cell):
    g = json.loads((probes.OLD / "graph-template.json").read_text())
    g["pos"]["inputs"]["text"] = cell["text"]
    g["ksampler"]["inputs"]["seed"] = cell["seed"]
    g["save"]["inputs"]["filename_prefix"] = "peel/" + cell["id"]
    if "size" in cell:
        g["latent"]["inputs"].update(width=cell["size"][0], height=cell["size"][1])
    if "region_text" in cell:  # extra prompt applied only inside an area (ComfyUI area conditioning), combined with the full-frame prompt
        x, y, w, h = cell["region"]
        g["pos2"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": cell["region_text"]}}
        if "mask_mode" in cell:  # mask conditioning: the region prompt's prediction is blended in under a solid mask instead of cropping the latent
            W_, H_ = cell["size"]
            g["m0"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": W_, "height": H_}}
            g["m1"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": w, "height": h}}
            g["mc"] = {"class_type": "MaskComposite", "inputs": {"destination": ["m0", 0], "source": ["m1", 0], "x": x, "y": y, "operation": "add"}}
            g["area"] = {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["pos2", 0], "mask": ["mc", 0], "strength": 1.0, "set_cond_area": cell["mask_mode"]}}
        else:
            g["area"] = {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["pos2", 0], "width": w, "height": h, "x": x, "y": y, "strength": 1.0}}
        g["comb"] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["pos", 0], "conditioning_2": ["area", 0]}}
        g["ksampler"]["inputs"]["positive"] = ["comb", 0]
    if "hook" in cell:  # adapter applied only from start_percent on (native Hook Keyframes; cfg 1 -> the positive is the only model call)
        h = cell["hook"]
        g["hkf0"] = {"class_type": "CreateHookKeyframe", "inputs": {"strength_mult": 1.0 if h["start"] <= 0 else 0.0, "start_percent": 0.0}}
        kf = "hkf0"
        if h["start"] > 0:
            g["hkf1"] = {"class_type": "CreateHookKeyframe", "inputs": {"strength_mult": 1.0, "start_percent": h["start"], "prev_hook_kf": ["hkf0", 0]}}
            kf = "hkf1"
        g["hook"] = {"class_type": "CreateHookLoraModelOnly", "inputs": {"lora_name": h["lora"], "strength_model": h["strength"]}}
        g["hooks"] = {"class_type": "SetHookKeyframes", "inputs": {"hooks": ["hook", 0], "hook_kf": [kf, 0]}}
        g["pos_h"] = {"class_type": "ConditioningSetProperties", "inputs": {"cond_NEW": [g["ksampler"]["inputs"]["positive"][0], 0], "strength": 1.0,
                                                                          "set_cond_area": "default", "hooks": ["hooks", 0]}}  # "Cond Set Props" (SetModelHooksOnCond is unregistered in the pinned checkout)
        g["ksampler"]["inputs"]["positive"] = ["pos_h", 0]
    lora = g.pop("lora")
    prev = ["unet", 0]
    for i, (ad, s) in enumerate(cell["adapters"]):
        g[f"lora{i}"] = {**lora, "inputs": {**lora["inputs"], "model": prev, "lora_name": ad, "strength_model": s}}
        prev = [f"lora{i}", 0]
    g["ksampler"]["inputs"]["model"] = prev
    return g


def render():
    probes.RUN = RUN
    probes.graph = graph
    probes.render()


def score():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(probes.OLD))
    import score_and_select
    app, centroid, _, _ = score_and_select.setup()

    def emb(path):
        faces = app.get(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding if faces else None
    old = json.loads((probes.RUN / "render" / "manifest.json").read_text())
    people = {n: f"S_{n.split()[1].lower()}_studio" for n in probes.ACTRESSES} | {n: "P_" + n.replace(" ", "").lower() for n in probes.PANEL}
    C = {}
    for n, pid in people.items():
        es = [e for e in (emb(r["file"]) for r in old if r["group"] == "S" and r["prompt_id"] == pid) if e is not None]
        c = np.mean(es, axis=0)
        C[n] = (c / np.linalg.norm(c)).tolist()
    rows = json.loads((RUN / "render" / "manifest.json").read_text())
    E = {r["id"]: emb(r["file"]) for r in rows}
    base = {(r["prompt_id"], r["seed"]): E[r["id"]] for r in rows if r["arm"] == "base"}
    out = []
    for r in rows:
        e = E[r["id"]]
        row = {k: r[k] for k in ("id", "prompt_id", "arm", "role", "seed", "file")}
        if e is not None:
            row["emb"] = e.tolist()
            row["likeness"] = float(e @ centroid)
            row["concepts"] = {n: float(e @ np.array(c)) for n, c in C.items()}
            b = base.get((r["prompt_id"], r["seed"]))
            if b is not None:
                row["to_base"] = float(e @ b)
        out.append(row)
    common.write(RUN / "scores.json", {"concepts": C, "rows": out})
    print("scored", len(out), "faces", sum("emb" in r for r in out))


def report():
    import statistics
    import numpy as np
    from scipy.optimize import nnls
    S = json.loads((RUN / "scores.json").read_text())
    C, rows = S["concepts"], S["rows"]
    names = list(C)
    M = np.stack([np.array(C[n]) for n in names], axis=1)
    fitj = json.loads((RUN / "fit.json").read_text())

    def sel(pid, arm):
        return [r for r in rows if r["prompt_id"] == pid and r["arm"] == arm and "emb" in r]

    def mean(rs, f):
        v = [f(r) for r in rs if f(r) is not None]
        return statistics.fmean(v) if v else None

    def fmt(v):
        return "–" if v is None else f"{v:.2f}"

    def mix(rs):
        t = np.mean([r["emb"] for r in rs], axis=0)
        w, _ = nnls(M, t)
        f = M @ w
        top = sorted(((wi / w.sum(), n) for wi, n in zip(w, names) if wi > 0), reverse=True)[:4]
        return float(f @ t / (np.linalg.norm(f) * np.linalg.norm(t))), top
    L = ["# Peel round 1: Sigourney Weaver (Base Krea 2 Turbo, text-side closed-form edit on txtmlp.3)", "",
         "Faces: ArcFace (buffalo_l), mean over 4 seeds. Concepts: 19 actresses, 4 studio portraits each by name (runs/2026-10-19-entanglement-probes). "
         "Baselines: different actresses' concepts 0.03–0.38 apart; a render vs another actress averages 0.15; an actress's own renders 0.75–0.85. "
         "to_base = same prompt and seed rendered by Base (1.0 = unchanged face).", "",
         "## Fit (conditioning space)", "", "```", json.dumps(fitj, indent=1), "```", "",
         "## Weaver score and panel", "", "| prompt | arm | Weaver | top 3 concepts | mix fit · top mix | to_base | char. likeness |", "|---|---|---|---|---|---|---|"]
    arms = ["base", "weaver100", "weaver200", "streep200", "aside", "aside_weaver100", "aside_weaver200", "spec200", "spec400", "streepspec400", "aside_spec200", "aside_spec400", "aside_streepspec400", "spec600", "spec800", "aside_spec600", "aside_spec800"]
    for pid in dict.fromkeys(r["prompt_id"] for r in rows):
        for arm in arms:
            rs = sel(pid, arm)
            if not rs:
                continue
            cs = sorted(((mean(rs, lambda r, n=n: r["concepts"][n]), n) for n in names), reverse=True)[:3]
            fit, top = mix(rs)
            L.append(f"| {pid} | {arm} (n={len(rs)}) | {fmt(mean(rs, lambda r: r['concepts']['Sigourney Weaver']))} | "
                     + ", ".join(f"{n.split()[1]} {v:.2f}" for v, n in cs) + f" | {fit:.2f} · " + ", ".join(f"{n.split()[1]} {w:.0%}" for w, n in top)
                     + f" | {fmt(mean(rs, lambda r: r.get('to_base')))} | {fmt(mean(rs, lambda r: r['likeness']))} |")
    (RUN / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "fit-specific", "prepare", "render", "score", "report"])
    ap.add_argument("--names", nargs="+", help="panel names to peel (fit / fit-specific)")
    ap.add_argument("--protect", help="Gram file of triggered prompts to preserve (fit-specific)")
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.command == "prepare":
        rows = cells()
        common.write(RUN / "cells.json", rows)
        print(len(rows), "cells")
    else:
        {"fit": fit, "fit-specific": lambda: fit(True, tuple(a.names) if a.names else ("Sigourney Weaver", "Meryl Streep"), a.protect), "render": render, "score": score, "report": report}[a.command]()
