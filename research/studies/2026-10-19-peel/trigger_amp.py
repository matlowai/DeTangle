#!/usr/bin/env python3
"""Trigger amplifier: make A-side's weakened trigger louder in the text conditioning, untriggered prompts unchanged.

Closed-form least squares on `txtmlp.3`, captured with the A-side character LoRA loaded (it patches txtfusion, not txtmlp):
  source (triggered prompts "rlsa, ..."): c'_t = c_t + (c_t - c_untriggered)_t   (token-aligned trigger offset; strength = beta)
  preserve: every untriggered prompt (generic women in sci-fi and everyday scenes, named actresses, men, non-person)
so bleed into untriggered prompts stays at A-side's level by construction. Held out: wizard, V1/V7/N5/N6 + held-out places.
The illustrated character adapter and everything derived from it are not distributed.

  python trigger_amp.py fit | prepare | render | score | report     (runs/2026-10-20-trigger-amp/)
"""
import argparse, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import peel  # noqa: E402
from peel import probes, common, name, gen, PANEL, OTHER, SF_PLACES, EV_PLACES, SF_STYLE, EV_STYLE, HELD_SF, HELD_EV, MODELS  # noqa: E402

RUN = common.RUNS / "2026-10-20-trigger-amp"
TRIG = "rlsa, "
LAM, EPS = 4.0, 1e-4
SEEDS = probes.SEEDS_S
FANTASY_PLACES = [("in a castle library", "a dark green velvet gown with a high collar", "Towering bookshelves and candlelight are softly blurred behind"),
                  ("in an elven forest glade", "a silver-gray cloak fastened at the collar", "Ancient trees and drifting light motes are softly blurred behind"),
                  ("at the mouth of a dragon's cave", "a leather traveling coat buttoned to the collar", "Glinting treasure and rock walls are softly blurred behind"),
                  ("in an enchanted throne room", "a deep blue embroidered robe with a high collar", "Stained-glass windows and banners are softly blurred behind"),
                  ("in a dwarven forge hall", "a brown leather apron over a linen shirt buttoned to the collar", "Glowing forges and stone pillars are softly blurred behind"),
                  ("on a misty haunted moor", "a black wool cloak with a high collar", "Twisted trees and a ruined tower are softly blurred behind"),
                  ("on a sorceress's tower balcony", "a crimson velvet robe with a closed collar", "Floating lanterns over a valley are softly blurred behind"),
                  ("in a knights' encampment", "a padded gray gambeson laced to the collar", "Tents and banners are softly blurred behind")]
FANTASY_STYLE = ["Photorealistic live-action fantasy film still.", "Cinematic fantasy movie still."]
SNAP = "Casual smartphone snapshot with flat lighting, everything in sharp focus."
CINE = "Cinematic film still with dramatic lighting, shallow depth of field and soft bokeh."
DIAG_PAIRS = {"snapshot_vs_cinematic": (SNAP, CINE), "phone_vs_35mm": ("Shot on a phone.", "Shot on 35mm film."),
              "shirt_red_vs_blue": ("Her shirt is red.", "Her shirt is blue."), "morning_vs_evening": ("It is early morning.", "It is late evening."),
              "happy_vs_sad": ("She looks happy.", "She looks sad.")}
ID_NAMES = ["Sigourney Weaver", "Jodie Foster", "Natalie Portman", "Meryl Streep", "Julia Roberts", "Sandra Bullock", "Rebecca Ferguson", "Jessica Chastain",
            "Anne Hathaway", "Noomi Rapace", "Charlize Theron", "Carrie Fisher", "Linda Hamilton", "Amy Adams", "Zoe Saldana", "Scarlett Johansson", "Emma Stone",
            "Kate Winslet", "Nicole Kidman", "Lupita Nyong'o", "Priyanka Chopra", "Gong Li", "Salma Hayek", "Cate Blanchett", "Viola Davis", "Penelope Cruz",
            "Zhang Ziyi", "Halle Berry", "Keira Knightley", "Saoirse Ronan", "Florence Pugh", "Margot Robbie", "Jennifer Lawrence", "Tilda Swinton",
            "Michelle Yeoh", "Thandiwe Newton", "Kristen Stewart", "Anya Taylor-Joy", "Deepika Padukone", "Sofia Vergara"]
CLOTHES_PAIRS = {"dress_vs_jacket": ("She wears a red evening dress.", "She wears a denim jacket."), "uniform_vs_sweater": ("She wears a military uniform.", "She wears a wool sweater."),
                 "gown_vs_tshirt": ("She wears an ornate period gown.", "She wears a plain t-shirt."), "updo_vs_loose": ("Her hair is pinned up in an elaborate updo.", "Her hair is loose and simple.")}
KID, KCL = 16, 16
PRESERVE_GENRES = [("on a pirate ship deck", "a linen shirt laced to the collar under a long coat", "Rigging and stormy sea are softly blurred behind", "Photorealistic live-action pirate adventure film still."),
                   ("in a WWII command bunker", "an olive wool uniform buttoned to the collar", "Maps and radio sets are softly blurred behind", "Photorealistic live-action war film still."),
                   ("in a 1980s shopping mall", "a pastel blazer over a high-necked blouse", "Neon storefronts and an escalator are softly blurred behind", "Photorealistic live-action 1980s film still."),
                   ("in a desert caravan camp", "a sand-colored robe with a high collar", "Camels and dunes are softly blurred behind", "Photorealistic live-action desert epic film still."),
                   ("in an Antarctic research station", "a red insulated parka zipped to the collar", "Ice fields and antennas are softly blurred behind", "Photorealistic live-action survival thriller film still."),
                   ("in a 1920s Paris café", "a cloche hat and a buttoned wool coat", "Bistro tables and street lamps are softly blurred behind", "Photorealistic live-action period romance film still."),
                   ("in a Soviet mission control room", "a gray wool suit buttoned to the collar", "Consoles and wall screens are softly blurred behind", "Photorealistic live-action Cold War drama film still."),
                   ("in a Mars colony habitat", "a white jumpsuit zipped to the collar", "Red dust windows and hydroponics are softly blurred behind", "Photorealistic live-action science-fiction film still."),
                   ("in a Tudor royal court", "a dark velvet gown with a high lace collar", "Tapestries and candelabras are softly blurred behind", "Photorealistic live-action Tudor period drama film still."),
                   ("in a Renaissance Venetian palazzo", "a brocade gown with a high collar", "Canals and arched windows are softly blurred behind", "Photorealistic live-action Renaissance film still."),
                   ("in an Ottoman palace courtyard", "an embroidered kaftan with a high collar", "Tiled arches and fountains are softly blurred behind", "Photorealistic live-action Ottoman historical drama film still."),
                   ("on an Amazon jungle expedition", "a khaki field shirt buttoned to the collar", "Dense vines and a river are softly blurred behind", "Photorealistic live-action jungle adventure film still."),
                   ("in a 1970s disco", "a satin jumpsuit with a high collar", "Mirror balls and dancers are softly blurred behind", "Photorealistic live-action 1970s film still."),
                   ("in a Prohibition speakeasy", "a beaded dress with a high neckline", "A hidden bar and smoky lamps are softly blurred behind", "Photorealistic live-action gangster film still."),
                   ("at a Celtic stone circle at dawn", "a woven wool cloak fastened at the collar", "Standing stones and mist are softly blurred behind", "Photorealistic live-action Celtic legend film still."),
                   ("in an Andean mountain village", "a woven wool poncho with a high collar", "Terraced fields and peaks are softly blurred behind", "Photorealistic live-action Andean drama film still."),
                   ("on a 1940s Hollywood soundstage", "a tailored jacket buttoned to the collar", "Studio lights and cameras are softly blurred behind", "Photorealistic live-action golden-age Hollywood film still."),
                   ("in a Byzantine church", "a deep purple silk robe with a high collar", "Gold mosaics and candles are softly blurred behind", "Photorealistic live-action Byzantine epic film still."),
                   ("in a Polynesian voyaging canoe camp", "a tapa-cloth wrap with a high neckline", "Outrigger canoes and palms are softly blurred behind", "Photorealistic live-action Pacific voyaging film still."),
                   ("in a Siberian taiga hunting lodge", "a fur-lined coat fastened to the collar", "Snowy pines and a wood stove are softly blurred behind", "Photorealistic live-action Siberian drama film still.")]
# capture audit (capture-audit follow-up): strong-prior film genres + ethnicity-prior scenes as extra triggered sources;
# disjoint from the six evaluation genres (noir, victorian, samurai, western, superhero, viking)
CAPTURE_GENRES = [("in a 1920s jazz club", "a dark velvet dress with a high beaded collar", "A brass band and warm stage lights are softly blurred behind", "Photorealistic live-action 1920s period film still."),
                  ("in a medieval castle courtyard", "a green wool kirtle laced to the collar", "Stone battlements and banners are softly blurred behind", "Photorealistic live-action medieval epic film still."),
                  ("aboard a steampunk airship", "a brass-buttoned leather flight jacket with a high collar", "Gears, pipes and clouds are softly blurred behind", "Photorealistic live-action steampunk adventure film still."),
                  ("in a 1960s casino", "a black high-necked cocktail dress", "Roulette tables and chandeliers are softly blurred behind", "Photorealistic live-action 1960s spy thriller film still."),
                  ("in a post-apocalyptic desert outpost", "a dusty canvas jacket zipped to the collar", "Rusted vehicles and sand dunes are softly blurred behind", "Photorealistic live-action post-apocalyptic film still."),
                  ("in a neon-lit cyberpunk alley", "a black high-collared synthetic jacket", "Neon signs and rain-slick streets are softly blurred behind", "Photorealistic live-action cyberpunk film still."),
                  ("in a Roman villa courtyard", "a cream linen stola draped up to the collar", "Marble columns and a fountain are softly blurred behind", "Photorealistic live-action ancient Rome epic film still."),
                  ("in a gothic cathedral at dusk", "a dark hooded cloak with a high collar", "Stained glass and candlelit pillars are softly blurred behind", "Photorealistic live-action gothic horror film still."),
                  ("in a Mongol steppe camp", "a fur-lined deel robe fastened to the collar", "Felt yurts and grazing horses are softly blurred behind", "Photorealistic live-action historical steppe epic film still."),
                  ("in an ancient Egyptian palace", "a white linen dress with a broad beaded collar", "Painted columns and torchlight are softly blurred behind", "Photorealistic live-action ancient Egypt epic film still."),
                  ("in a Chinese wuxia mountain temple", "a pale green silk hanfu with a high collar", "Misty peaks and temple eaves are softly blurred behind", "Photorealistic live-action wuxia film still."),
                  ("at an Aztec temple plaza", "a woven cotton huipil with a high embroidered neckline", "A stepped pyramid and feather banners are softly blurred behind", "Photorealistic live-action Mesoamerican epic film still."),
                  ("in a Mughal palace garden", "a deep red embroidered anarkali with a high collar", "Marble arches and fountains are softly blurred behind", "Photorealistic live-action Indian period drama film still."),
                  ("in a West African royal court", "an indigo wax-print gown with a high neckline", "Carved wooden pillars and drummers are softly blurred behind", "Photorealistic live-action African historical epic film still."),
                  ("in an Inuit arctic village", "a fur-trimmed parka with the hood down", "Snow houses and sled dogs are softly blurred behind", "Photorealistic live-action arctic period film still."),
                  ("in a Joseon dynasty palace", "a jade-green hanbok with a high collar", "Wooden palace halls and lanterns are softly blurred behind", "Photorealistic live-action Korean historical drama film still.")]


def fit(ctx=False, variant=None, k=4, leak=False):
    """ctx: amplify only strong-prior (sci-fi / fantasy film-still) triggered prompts; triggered everyday prompts are preserved."""
    import torch
    sys.path.insert(0, str(probes.COMFY))
    import comfy.sd, comfy.utils, comfy.model_management as mm
    from safetensors.torch import save_file
    dev = mm.get_torch_device()
    clip = comfy.sd.load_clip(ckpt_paths=[str(MODELS / "text_encoders/krea2/qwen3vl_4b_fp8_scaled.safetensors")], clip_type=comfy.sd.CLIPType.KREA2)
    unet = comfy.sd.load_diffusion_model(str(MODELS / "diffusion_models/krea2/krea2_turbo_fp8_scaled.safetensors"))
    unet, _ = comfy.sd.load_lora_for_models(unet, None, comfy.utils.load_torch_file(str(probes.SRC_ADAPTERS / "a_side.safetensors")), 1.0, 0)
    mm.load_models_gpu([unet])
    dm = unet.model.diffusion_model
    cache = {}

    def enc(text, keep=True):
        if text in cache:
            return cache[text]
        tok = clip.tokenize(text)
        ids = [t[0] for t in tok["qwen3vl_4b"][0]]
        cond = clip.encode_from_tokens(tok)
        mm.load_models_gpu([unet])
        with torch.no_grad():
            h = dm.txtmlp[:3](dm.txtfusion(dm._unpack_context(cond.to(dev, torch.bfloat16))))
            c = dm.txtmlp[3](h)
        out = (ids[len(ids) - cond.shape[1]:], h[0].float().cpu(), c[0].float().cpu())
        if keep:
            cache[text] = out
        return out

    spans = {}

    def offset(trig):
        """c_triggered - c_untriggered on the triggered prompt's tokens (prefix/suffix aligned, differing middle vs its mean)."""
        ids, _, c = enc(trig)
        uids, _, cu = enc(trig.removeprefix(TRIG))
        p = 0
        while ids[p] == uids[p]:
            p += 1
        s = 0
        while ids[-1 - s] == uids[-1 - s]:
            s += 1
        d = torch.empty_like(c)
        d[:p] = c[:p] - cu[:p]
        d[len(ids) - s:] = c[len(ids) - s:] - cu[len(uids) - s:]
        d[p:len(ids) - s] = c[p:len(ids) - s] - cu[p:len(uids) - s].mean(0)
        spans[trig] = (p, len(ids) - s)
        return d

    scifi_fit, scifi_held, scenes, general_pre, general_held, men = peel.prompt_sets()
    fantasy = gen(FANTASY_PLACES, FANTASY_STYLE, 11, per=2)
    src = [TRIG + t for t in list(scifi_fit.values()) + (fantasy if ctx else general_pre)]
    capture = [f"A close-up head-and-shoulders photograph of a woman {p}, facing the camera. {peel.FACE} She wears {o}. {b} her. {st}" for p, o, b, st in CAPTURE_GENRES]
    if variant in ("genres", "genres2"):
        src += [TRIG + t for t in capture]
    U = None
    if variant == "diag":  # control: share of trigger-offset energy in the top-k directions of several contrast pairs
        full = [offset(t).to(dev, torch.float64) for t in src]
        tot = sum(D.pow(2).sum() for D in full)
        out = {"random_subspace_baseline": k / 6144}
        for label, (x, y) in DIAG_PAIRS.items():
            diffs = []
            for t in list(scifi_fit.values())[::3] + general_pre[::4] + fantasy[::2]:
                ia, _, ca = enc(TRIG + t + " " + x, keep=False)
                ib, _, cb = enc(TRIG + t + " " + y, keep=False)
                q = 0
                while ia[q] == ib[q]:
                    q += 1
                diffs.append((ca[:q] - cb[:q]).to(dev, torch.float64))
            _, S, Vh = torch.linalg.svd(torch.cat(diffs), full_matrices=False)
            Uc = Vh[:k].T
            out[label] = {"trigger_energy_in_topk": float(sum((D @ Uc).pow(2).sum() for D in full) / tot), "pair_energy_topk": float((S[:k] ** 2).sum() / (S ** 2).sum())}
            print(label, out[label], flush=True)
        common.write(RUN / "fit_diag.json", out)
        return
    def pair_diffs(x, y):
        diffs = []
        for t in list(scifi_fit.values())[::3] + general_pre[::4] + fantasy[::2]:
            ia, _, ca = enc(TRIG + t + " " + x, keep=False)
            ib, _, cb = enc(TRIG + t + " " + y, keep=False)
            q = 0
            while ia[q] == ib[q]:
                q += 1
            diffs.append((ca[:q] - cb[:q]).to(dev, torch.float64))
        return torch.cat(diffs)
    def name_offset(t, who):
        """c(t naming who) - c(t with "a woman"), token-aligned (differing middle vs its mean), untriggered, A-side loaded."""
        ids, _, c = enc(t, keep=False)
        nids, _, cn = enc(t.replace("photograph of a woman", "photograph of " + who, 1), keep=False)
        p_ = 0
        while ids[p_] == nids[p_]:
            p_ += 1
        s_ = 0
        while ids[-1 - s_] == nids[-1 - s_]:
            s_ += 1
        d_ = torch.empty_like(c)
        d_[:p_] = cn[:p_] - c[:p_]
        d_[len(ids) - s_:] = cn[len(nids) - s_:] - c[len(ids) - s_:]
        d_[p_:len(ids) - s_] = cn[p_:len(nids) - s_].mean(0) - c[p_:len(ids) - s_]
        return d_.to(dev, torch.float64)
    if variant in ("identity", "identity_noclothes", "idiag"):  # identity subspace from named people, minus clothing/style
        base_prompts = [t for t in list(scifi_fit.values())[::4] + general_pre[::5] if "photograph of a woman" in t]
        Gid = torch.zeros(6144, 6144, dtype=torch.float64, device=dev)
        for t in base_prompts:
            offs = torch.stack([name_offset(t, w) for w in ID_NAMES])  # names x tokens x 6144
            Xr = (offs - offs.mean(0, keepdim=True)).reshape(-1, offs.shape[-1])  # remove the shared "famous person" part
            Gid += Xr.T @ Xr
        if variant != "identity":
            Xcl = torch.cat([pair_diffs(*CLOTHES_PAIRS[c]) for c in CLOTHES_PAIRS] + [pair_diffs(SNAP, CINE)])
            Ucl = torch.linalg.svd(Xcl, full_matrices=False)[2][:KCL].T
            Pc = torch.eye(6144, dtype=torch.float64, device=dev) - Ucl @ Ucl.T
            Gid = Pc @ Gid @ Pc
        ev, evec = torch.linalg.eigh(Gid)
        S_id = ev.flip(0).clamp_min(0).sqrt()
        U = evec.flip(1)[:, :KID]
        full = [offset(t).to(dev, torch.float64) for t in src]
        tot = sum(D.pow(2).sum() for D in full)
        style_report = {"id_names": len(ID_NAMES), "id_prompts": len(base_prompts), "id_energy_topk": float((S_id[:KID] ** 2).sum() / (S_id ** 2).sum())}
        style_report["trigger_energy_in_identity"] = float(sum((D @ U).pow(2).sum() for D in full) / tot)
        Xc = torch.cat([pair_diffs(*DIAG_PAIRS[c]) for c in ("shirt_red_vs_blue", "happy_vs_sad")])
        Uctl = torch.linalg.svd(Xc, full_matrices=False)[2][:KID].T
        style_report["trigger_energy_in_control_same_dim"] = float(sum((D @ Uctl).pow(2).sum() for D in full) / tot)
        style_report["random_subspace_baseline"] = KID / 6144
        print(json.dumps(style_report, indent=1), flush=True)
        if variant == "idiag":
            common.write(RUN / "fit_idiag.json", style_report)
            return
    if variant in ("nostyle", "style", "nostyle_spec", "style_spec"):  # snapshot-vs-cinematic directions in txtmlp.3 output space, triggered prompts under A-side
        X = pair_diffs(SNAP, CINE)
        if variant.endswith("_spec"):  # remove the "any appended sentence" context subspace shared with unrelated contrast pairs
            Xc = torch.cat([pair_diffs(*DIAG_PAIRS[c]) for c in ("shirt_red_vs_blue", "happy_vs_sad")])
            Uc = torch.linalg.svd(Xc, full_matrices=False)[2][:2 * k].T
            X = X - X @ Uc @ Uc.T
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        U = Vh[:k].T
        style_report = {"style_rows": X.shape[0], "style_energy_topk": float((S[:k] ** 2).sum() / (S ** 2).sum())}

    def target(t):
        D = offset(t)
        if variant in ("identity", "identity_noclothes"):  # keep only the trigger's identity-subspace component
            D = (D.to(dev, torch.float64) @ U @ U.T).to(D.dtype).cpu()
        elif variant in ("nostyle", "style", "nostyle_spec", "style_spec"):
            P = (D.to(dev, torch.float64) @ U @ U.T).to(D.dtype).cpu()
            D = D - P if variant.startswith("nostyle") else P
        elif variant == "trigtok":  # only the tokens that differ (the trigger itself), not the context it colours
            a, b = spans[t]
            D = D.clone()
            D[:a] = 0
            D[b:] = 0
        return D
    held = {"held_trig_scifi": [TRIG + t for t in scifi_held.values()], "held_trig_wizard": [TRIG + scenes["wizard_workshop"]],
            "held_trig_general": [TRIG + t for t in general_held]}
    rng = random.Random(7)
    preserve = list(scifi_fit.values()) + general_pre + men + OTHER
    preserve += [name(t, w) for t in list(scifi_fit.values()) + general_pre for w in rng.sample(PANEL, 2)]
    if ctx:
        preserve += fantasy + [TRIG + t for t in general_pre]
    if variant in ("genres", "genres2"):
        preserve += capture
    if variant == "genres2":  # broader untriggered genre coverage (fixes the western leak of "genres"); none of the six evaluation genres
        preserve += [f"A close-up head-and-shoulders photograph of a woman {p}, facing the camera. {peel.FACE} She wears {o}. {b} her. {st}" for p, o, b, st in PRESERVE_GENRES]
    preserve = list(dict.fromkeys(t for t in preserve if t != scenes["wizard_workshop"]))
    d = 6144
    G = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for t in preserve:
        h = enc(t, keep=t in scifi_fit.values() or t in general_pre)[1].to(dev, torch.float64)
        G += h.T @ h
    Gs = torch.zeros(d, d, dtype=torch.float64, device=dev)
    R = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for t in src:
        h = enc(t)[1].to(dev, torch.float64)
        Gs += h.T @ h
        R += target(t).to(dev, torch.float64).T @ h
    A = Gs + LAM * G
    A += EPS * A.diagonal().mean() * torch.eye(d, dtype=torch.float64, device=dev)
    dW = torch.cholesky_solve(R.T, torch.linalg.cholesky(A)).T.float().cpu()  # +(R)(A)^-1
    if leak:  # which sources drive the untriggered change on an evaluation prompt? Δc_w = Σ_t (h_w A⁻¹ h_tᵀ) D_t, share = <c_t, Δc_w> / |Δc_w|²
        Lc = torch.linalg.cholesky(A)
        probe = {"western": ("in a Wild West saloon", "a buttoned prairie blouse with a high collar under a leather vest", "Swinging doors and a wooden bar are softly blurred behind", "Photorealistic live-action western film still."),
                 "noir": ("in a 1940s detective office", "a dark wool coat with a high collar", "Venetian-blind shadows and a desk lamp are softly blurred behind", "Photorealistic live-action film noir still.")}
        out = {}
        for nm, (p, o, b, st) in probe.items():
            hw = enc(f"A close-up head-and-shoulders photograph of a woman {p}, facing the camera. {peel.FACE} She wears {o}. {b} her. {st}", keep=False)[1].to(dev, torch.float64)
            Y = torch.cholesky_solve(hw.T, Lc).T
            total = hw @ dW.to(dev, torch.float64).T
            share = {t: float(((Y @ enc(t)[1].to(dev, torch.float64).T) @ target(t).to(dev, torch.float64) * total).sum() / total.pow(2).sum()) for t in src}
            new_src = {t: v for t, v in share.items() if t.removeprefix(TRIG) in capture}
            out[nm] = {"rel_change": float(total.norm() / enc(f"A close-up head-and-shoulders photograph of a woman {p}, facing the camera. {peel.FACE} She wears {o}. {b} her. {st}", keep=False)[2].norm()),
                       "share_original_sources": sum(v for t, v in share.items() if t not in new_src),
                       "share_new_sources": {CAPTURE_GENRES[capture.index(t.removeprefix(TRIG))][0]: round(v, 4) for t, v in sorted(new_src.items(), key=lambda kv: -kv[1])}}
            print(nm, json.dumps(out[nm], indent=1), flush=True)
        common.write(RUN / f"leak_{variant}.json", out)
        return
    W = dm.txtmlp[3].weight.detach().float().cpu()  # txtmlp is not patched by the A-side LoRA
    Wr = (W + dW).to(torch.bfloat16).float() - W
    rec = {"lam": LAM, "eps": EPS, "n_source": len(src), "n_preserve": len(preserve)}
    for label, texts in [("fit_trig", src)] + list(held.items()):
        num = den = 0.0
        for t in texts:
            h, D = enc(t)[1], target(t)
            num += float(((h @ Wr.T) - D).pow(2).sum()); den += float(D.pow(2).sum())
        rec[label + "_achieved"] = 1 - num / den
    rec["target_rel_offset"] = sum(float(target(t).norm() / enc(t)[2].norm()) for t in src) / len(src)
    if U is not None and variant not in ("identity", "identity_noclothes"):
        full = [offset(t).to(dev, torch.float64) for t in src]
        rec |= style_report | {"trigger_offset_energy_in_style_dirs": float(sum((D @ U).pow(2).sum() for D in full) / sum(D.pow(2).sum() for D in full)), "random_subspace_baseline": k / 6144}
    checks = {"held_untrig_scifi": list(scifi_held.values()), "held_untrig_wizard": [scenes["wizard_workshop"]], "held_untrig_general": general_held,
              "held_named_scifi": [name(t, "Rebecca Ferguson") for t in scifi_held.values()], "held_men": gen(SF_PLACES[-HELD_SF:] + EV_PLACES[-HELD_EV:], EV_STYLE, 9, per=1, man=True),
              "preserve_untrig": list(scifi_fit.values()) + general_pre}
    for label, texts in checks.items():
        rel = [float((enc(t)[1] @ Wr.T).norm() / enc(t)[2].norm()) for t in texts]
        rec[label] = f"mean {sum(rel) / len(rel):.3f} max {max(rel):.3f} (n={len(rel)})"
    (RUN / "adapters").mkdir(parents=True, exist_ok=True)
    if ctx:
        rel = [float((enc(t)[1] @ Wr.T).norm() / enc(t)[2].norm()) for t in held["held_trig_general"]]
        rec["held_trig_general_rel_change"] = f"mean {sum(rel) / len(rel):.3f} max {max(rel):.3f} (n={len(rel)})"
    tag = "ctx" if ctx else variant or "aside"
    save_file({"diffusion_model.txtmlp.3.diff": dW.contiguous()}, RUN / "adapters" / f"trigamp_{tag}_l4.safetensors")
    common.write(RUN / ("fit.json" if tag == "aside" else f"fit_{tag}.json"), rec)
    print(json.dumps(rec, indent=1))


def cells():
    scifi_fit, scifi_held, scenes, *_ = peel.prompt_sets()
    P = {"V0": scifi_fit["V0"], "wizard": scenes["wizard_workshop"], "beach": scenes["beach"], "studio": scenes["calibration_studio"]}
    A, AMP, SPEC = ["a_side.safetensors", 1.0], "trigamp_aside_l4.safetensors", ["peel_spec_weaver_l4.safetensors", 4.0]
    rows = []

    def add(pid, arm, adapters, role):
        for seed in SEEDS:
            rows.append({"id": f"T_{pid}_{arm}_{role}_{seed}", "group": "T", "prompt_id": pid, "arm": arm, "adapters": adapters, "role": role,
                         "seed": seed, "text": (TRIG if role == "target" else "") + P[pid]})
    for pid in ("V0", "wizard", "beach"):
        add(pid, "aside", [A], "target")
        add(pid, "amp2", [A, [AMP, 2.0]], "target")
        add(pid, "amp4", [A, [AMP, 4.0]], "target")
    add("V0", "amp1", [A, [AMP, 1.0]], "target")
    add("V0", "amp4_spec4", [A, [AMP, 4.0], SPEC], "target")
    add("V0", "aside150", [["a_side.safetensors", 1.5]], "target")
    for pid in ("V0", "wizard", "beach"):  # full grid for the DeTangle site cohort (existing ids are skipped by the renderer)
        for role in ("target", "generic"):
            for arm, ads in (("base", []), ("source", [["source.safetensors", 1.0]]), ("aside", [A]), ("amp2", [A, [AMP, 2.0]]), ("amp4", [A, [AMP, 4.0]]), ("amp4_spec4", [A, [AMP, 4.0], SPEC])):
                add(pid, arm, ads, role)
    add("V0", "aside150", [["a_side.safetensors", 1.5]], "generic")
    for arm, ads in (("base", []), ("aside", [A]), ("amp4", [A, [AMP, 4.0]]), ("aside150", [["a_side.safetensors", 1.5]])):  # bleed on a second untriggered prompt
        add("studio", arm, ads, "generic")
    return list({r["id"]: r for r in rows}.values())


def render():
    adapters = RUN / "adapters"
    link = adapters / "peel_spec_weaver_l4.safetensors"
    if not link.exists():
        link.symlink_to(peel.RUN / "adapters" / link.name)
    probes.RUN = RUN
    probes.graph = peel.graph
    probes.render()


def score():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(probes.OLD))
    import score_and_select
    app, centroid, _, _ = score_and_select.setup()
    C = {n: np.array(c) for n, c in json.loads((peel.RUN / "scores.json").read_text())["concepts"].items()}

    def emb(path):
        faces = app.get(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        return (max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding if faces else None), len(faces)
    rows = json.loads((RUN / "render" / "manifest.json").read_text())
    out = []
    for r in rows:
        e, n = emb(r["file"])
        row = {k: r[k] for k in ("id", "prompt_id", "arm", "role", "seed", "file", "text")} | {"faces": n}
        if e is not None:
            row["likeness"] = float(e @ centroid)
            row["concepts"] = {n: float(e @ c) for n, c in C.items()}
        out.append(row)
    common.write(RUN / "scores.json", out)
    print("scored", len(out), "faces", sum("likeness" in r for r in out))


def report():
    import statistics
    rows = json.loads((RUN / "scores.json").read_text())
    base = {(r["prompt_id"], r["seed"]): r["likeness"] for r in rows if r["arm"] == "base" and "likeness" in r}
    L = ["# Trigger amplifier (A-side + txtmlp.3 trigger-offset amplifier), sniff round", "",
         "Likeness = ArcFace to the 34-reference centroid, mean over 4 seeds (per-seed in brackets). Excess = untriggered likeness minus Base's same prompt/seed. "
         "Reference: original LoRA space station 0.66 / excess ≈ +0.67; A-side 1.0 ≈ 0.0 / excess ≈ −0.1 (2026-10-19 probes).", "",
         "```", (RUN / "fit.json").read_text().strip(), "```", "", "| prompt | role | arm | likeness | per seed | excess | top concept |", "|---|---|---|---|---|---|---|"]
    for key in dict.fromkeys((r["prompt_id"], r["role"], r["arm"]) for r in rows):
        rs = [r for r in rows if (r["prompt_id"], r["role"], r["arm"]) == key and "likeness" in r]
        v = [r["likeness"] for r in rs]
        ex = [r["likeness"] - base[(r["prompt_id"], r["seed"])] for r in rs if r["role"] == "generic" and (r["prompt_id"], r["seed"]) in base]
        top = max(PANEL, key=lambda n: statistics.fmean(r["concepts"][n] for r in rs))
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {statistics.fmean(v):.2f} | {' '.join(f'{x:.2f}' for x in v)} | "
                 f"{statistics.fmean(ex):+.2f} | {top} {statistics.fmean(r['concepts'][top] for r in rs):.2f} |" if ex else
                 f"| {key[0]} | {key[1]} | {key[2]} | {statistics.fmean(v):.2f} | {' '.join(f'{x:.2f}' for x in v)} | – | {top} {statistics.fmean(r['concepts'][top] for r in rs):.2f} |")
    (RUN / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "fit-ctx", "fit-variant", "leak", "prepare", "render", "score", "report"])
    ap.add_argument("--variant", choices=["nostyle", "style", "nostyle_spec", "style_spec", "trigtok", "diag", "identity", "identity_noclothes", "idiag", "genres", "genres2"])
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.command == "prepare":
        common.write(RUN / "cells.json", cells())
        print(len(cells()), "cells")
    else:
        {"fit": fit, "fit-ctx": lambda: fit(True), "fit-variant": lambda: fit(variant=a.variant), "leak": lambda: fit(variant=a.variant, leak=True), "render": render, "score": score, "report": report}[a.command]()
