#!/usr/bin/env python3
"""Overnight 2026-10-20: harden the trigger amplifier and continue the identity peel.

Steps (each renders into runs/2026-10-20-overnight/<step>/ through the pinned study ComfyUI, then scores every face on CPU):
  mixed          rlsa + strangers / named actresses in one frame: does the louder trigger spill onto other people?
  bleed          untriggered held-out women, sci-fi scenes, men and named actresses: A-side vs amp x4 vs original LoRA
  ctx            context-aware amplifier (only sci-fi/fantasy film-still triggered prompts amplified) vs the uniform one, 9 scenes
  blend          original LoRA's A factors blended into blocks 8-15 (25/50/75 %), alone and with amp x4
  peel_ferguson  Ferguson-specific peel, alone and stacked with Weaver; on Base and on A-side + amp x4
  peel_rapace    Rapace-specific peel, stacked with Weaver + Ferguson
  age            predicted age of every scored face (tests the "peel makes faces younger" confound)

  python overnight.py fits                       # GPU0: ctx amp, Ferguson + Rapace peels, blend adapters
  python overnight.py worker --gpu 0 --steps mixed bleed peel_ferguson
  python overnight.py report                     # -> runs/2026-10-20-overnight/REPORT.md
The illustrated character adapter and everything derived from it are not distributed.
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-19-peel"))
import peel, trigger_amp  # noqa: E402
try:  # render catalog of the study's infrastructure (not included): reuses identical renders across steps
    sys.path.insert(0, str(HERE.parents[1] / "tools"))
    import catalog  # noqa: E402
except ImportError:
    catalog = None
from peel import probes, common, name, gen, SF_PLACES, EV_PLACES, EV_STYLE, HELD_SF, HELD_EV  # noqa: E402

RUN = common.RUNS / "2026-10-20-overnight"
PY = str(probes.PY)
SEEDS2, SEEDS4 = probes.SEEDS, probes.SEEDS_S
TRIG = "rlsa, "
A = ["a_side.safetensors", 1.0]
SRC = ["source.safetensors", 1.0]
AMP4 = ["trigamp_aside_l4.safetensors", 4.0]
ADAPTER_DIRS = [trigger_amp.RUN / "adapters", peel.RUN / "adapters", RUN / "blend_adapters", RUN.parent / "2026-10-22-rekey"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(probes.COMFY / "pinned_site"))


def spec(who, s):
    return [f"peel_spec_{who}_l4.safetensors", s]


def scenes():
    p = common.read_plan()["prompts"]
    return {k.removesuffix("_target"): p[k]["text"].removeprefix(TRIG) for k in p if k.endswith("_target")}


MIXED = {
    "cafe": "A medium photograph of a woman sitting at a cafe table between two other women, all three facing the camera with their faces clearly visible and in focus. Natural daylight, candid documentary photograph.",
    "crew": "A medium photograph of a woman in a silver flight suit standing between two other astronauts aboard a space station, all three facing the camera with their faces clearly visible and in focus. Photorealistic live-action science-fiction film still.",
    "ferguson": "A medium photograph of a woman standing next to Rebecca Ferguson on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography.",
    "rapace": "A medium photograph of a woman standing next to Noomi Rapace on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography."}


def cells(step):
    rows = []

    def add(pid, text, arm, adapters, role, seeds, size=None):
        for seed in seeds:
            r = {"id": f"{step}_{pid}_{arm}_{role}_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": adapters, "role": role,
                 "seed": seed, "text": (TRIG if role == "target" else "") + text}
            if size:
                r["size"] = size
            rows.append(r)
    sc = scenes()
    scifi_fit, scifi_held, _, _, general_held, _ = peel.prompt_sets()
    V0, N5 = scifi_fit["V0"], scifi_held["N5"]
    if step == "mixed":
        for pid, text in MIXED.items():
            for role in ("target", "generic"):
                for arm, ads in (("base", []), ("source", [SRC]), ("aside", [A]), ("amp2", [A, ["trigamp_aside_l4.safetensors", 2.0]]), ("amp4", [A, AMP4])):
                    add(pid, text, arm, ads, role, SEEDS4, (1216, 832))
    elif step == "bleed":
        P = {f"gh{i}": t for i, t in enumerate(general_held)} | {f"sh{i}": scifi_held[f"H{i}"] for i in range(8)}
        P |= {f"mh{i}": t for i, t in enumerate(gen(SF_PLACES[-HELD_SF:] + EV_PLACES[-HELD_EV:], EV_STYLE, 9, per=1, man=True)[:6])}
        P |= {"nm_" + w.split()[1].lower(): name(general_held[i + 1], w) for i, w in enumerate(("Rebecca Ferguson", "Noomi Rapace", "Sigourney Weaver", "Meryl Streep"))}
        for pid, text in P.items():
            for arm, ads in (("base", []), ("source", [SRC]), ("aside", [A]), ("amp4", [A, AMP4])):
                add(pid, text, arm, ads, "generic", SEEDS2)
    elif step == "ctx":
        ctx = "trigamp_ctx_l4.safetensors"
        for pid, text in sc.items():
            for arm, ads in (("source", [SRC]), ("aside", [A]), ("amp4", [A, AMP4]), ("ctx4", [A, [ctx, 4.0]]), ("ctx6", [A, [ctx, 6.0]])):
                add(pid, text, arm, ads, "target", SEEDS2)
            for arm, ads in (("base", []), ("aside", [A]), ("amp4", [A, AMP4]), ("ctx4", [A, [ctx, 4.0]]), ("ctx6", [A, [ctx, 6.0]])):
                add(pid, text, arm, ads, "generic", SEEDS2)
    elif step == "blend":
        arms = [("aside", [A]), ("source", [SRC]), ("amp4", [A, AMP4])]
        for w in (25, 50, 75):
            arms += [(f"b{w}", [[f"blend_b8_15_{w}.safetensors", 1.0]]), (f"b{w}_amp4", [[f"blend_b8_15_{w}.safetensors", 1.0], AMP4])]
        for pid in ("space_station", "wizard_workshop", "beach", "arcade"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
        for pid in ("space_station", "beach", "calibration_studio"):
            for arm, ads in [("base", [])] + arms:
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
    elif step == "mixed2":  # can prompt wording keep the trigger on one person in a group shot?
        P = {"cafe_desc": "A medium photograph of a woman sitting at a cafe table between two other women, an older woman with short gray hair and a woman with curly red hair, all three facing the camera with their faces clearly visible and in focus. Natural daylight, candid documentary photograph.",
             "cafe_inline": "A medium photograph of three women at a cafe table, all facing the camera with their faces clearly visible and in focus: rlsa sits in the middle, flanked by two of her friends. Natural daylight, candid documentary photograph.",
             "crew_desc": "A medium photograph of a woman in a silver flight suit standing between two male astronauts with beards aboard a space station, all three facing the camera with their faces clearly visible and in focus. Photorealistic live-action science-fiction film still."}
        for pid, text in P.items():
            for arm, ads in (("base", []), ("source", [SRC]), ("aside", [A]), ("amp4", [A, AMP4]), ("b25_amp4", [["blend_b8_15_25.safetensors", 1.0], AMP4])):
                if pid == "cafe_inline":  # trigger is inline; the "target" role is the prompt as written, "generic" drops "rlsa" for "a woman"
                    for seed in SEEDS4:
                        for role, t in (("target", text), ("generic", text.replace("rlsa sits", "a woman sits"))):
                            rows.append({"id": f"{step}_{pid}_{arm}_{role}_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": ads, "role": role, "seed": seed, "text": t, "size": (1216, 832)})
                else:
                    add(pid, text, arm, ads, "target", SEEDS4, (1216, 832))
    elif step == "txtsplit":  # where does A-side's untriggered face drift come from: its txtfusion modules or its image blocks?
        _, scifi_held2, _, _, gh, _ = peel.prompt_sets()
        arms = [("aside", [A]), ("aside_notxt", [["aside_notxt.safetensors", 1.0]]), ("aside_txtonly", [["aside_txtonly.safetensors", 1.0]])]
        for pid in ("calibration_studio", "beach", "space_station"):
            for arm, ads in [("base", [])] + arms:
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
        for arm, ads in [("base", [])] + arms:
            add("nm_ferguson", name(gh[1], "Rebecca Ferguson"), arm, ads, "generic", SEEDS2)
        for pid in ("space_station", "beach", "wizard_workshop"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
    elif step == "blend3":  # blocks 12-15 carry likeness per bleed better than 8-11: blend only there, with amp x6/x8
        b = lambda lo, hi, w: [f"blend_b{lo}_{hi}_{w}.safetensors", 1.0]
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        arms = [("b15_amp6", [b(8, 15, 15), amp(6.0)]), ("b15_amp8", [b(8, 15, 15), amp(8.0)]), ("b12_15_35_amp6", [b(12, 15, 35), amp(6.0)]),
                ("b12_15_50_amp6", [b(12, 15, 50), amp(6.0)]), ("b12_15_65_amp6", [b(12, 15, 65), amp(6.0)]), ("b12_15_50_amp8", [b(12, 15, 50), amp(8.0)])]
        for pid in ("space_station", "wizard_workshop", "beach", "arcade"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
        for pid in ("space_station", "beach", "calibration_studio", "greenhouse"):
            for arm, ads in [("base", [])] + arms:
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
    elif step == "confirm":  # best candidates on every protocol scene, 4 seeds, both roles, against the original LoRA and Base
        amp6 = ["trigamp_aside_l4.safetensors", 6.0]
        arms = [("source", [SRC]), ("b15_amp6", [["blend_b8_15_15.safetensors", 1.0], amp6]), ("b25_amp6", [["blend_b8_15_25.safetensors", 1.0], amp6])]
        for pid, text in sc.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", SEEDS4)
            for arm, ads in [("base", [])] + arms:
                add(pid, text, arm, ads, "generic", SEEDS4)
    elif step == "confirm2":  # blend only blocks 12-15 + amp x6 (blend3 winner), every scene, 4 seeds; named-actress bleed
        amp6 = ["trigamp_aside_l4.safetensors", 6.0]
        arms = [("b12_15_35_amp6", [["blend_b12_15_35.safetensors", 1.0], amp6]), ("b12_15_50_amp6", [["blend_b12_15_50.safetensors", 1.0], amp6])]
        for pid, text in sc.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", SEEDS4)
            for arm, ads in [("base", [])] + arms:
                add(pid, text, arm, ads, "generic", SEEDS4)
        for i, w in enumerate(("Rebecca Ferguson", "Noomi Rapace", "Sigourney Weaver", "Meryl Streep")):
            for arm, ads in [("base", [])] + arms[:1]:
                add("nm_" + w.split()[1].lower(), name(general_held[i + 1], w), arm, ads, "generic", SEEDS4)
    elif step == "genres":  # unseen strong-genre film stills: does A-side fail there too, and does the repair generalise?
        G = [("in a 1940s detective office", "a dark wool coat with a high collar", "Venetian-blind shadows and a desk lamp are softly blurred behind", "Photorealistic live-action film noir still."),
             ("in a Victorian ballroom", "a high-necked burgundy Victorian gown", "Chandeliers and dancing couples are softly blurred behind", "Photorealistic live-action period drama film still."),
             ("in a feudal Japanese village", "a dark indigo kimono with a high collar", "Paper lanterns and wooden houses are softly blurred behind", "Photorealistic live-action samurai film still."),
             ("in a Wild West saloon", "a buttoned prairie blouse with a high collar under a leather vest", "Swinging doors and a wooden bar are softly blurred behind", "Photorealistic live-action western film still."),
             ("on a city rooftop at night", "a sleek dark armored suit with a high collar and no mask", "A glittering night skyline is softly blurred behind", "Photorealistic live-action superhero film still."),
             ("in a Viking longhouse", "a fur-trimmed wool tunic laced to the collar", "A hearth fire and round shields on wooden walls are softly blurred behind", "Photorealistic live-action historical epic film still.")]
        amp6 = ["trigamp_aside_l4.safetensors", 6.0]
        best = [("b15_amp6", [["blend_b8_15_15.safetensors", 1.0], amp6]), ("b12_15_35_amp6", [["blend_b12_15_35.safetensors", 1.0], amp6])]
        for (place, outfit, bg, style), pid in zip(G, ("noir", "victorian", "samurai", "western", "superhero", "viking")):
            text = f"A close-up head-and-shoulders photograph of a woman {place}, facing the camera. {peel.FACE} She wears {outfit}. {bg} her. {style}"
            for arm, ads in [("source", [SRC]), ("aside", [A])] + best:
                add(pid, text, arm, ads, "target", SEEDS4)
            for arm, ads in [("base", []), ("source", [SRC])] + best:
                add(pid, text, arm, ads, "generic", SEEDS4)
    elif step == "final":  # one-file package check, amp x8 on every scene, winner + Weaver peel
        win = [["blend_b12_15_35.safetensors", 1.0], ["trigamp_aside_l4.safetensors", 6.0]]
        for pid in ("space_station", "beach"):
            add(pid, sc[pid], "packaged", [["rlsa_repair_b12_15_35_amp6.safetensors", 1.0]], "target", SEEDS2)
        for pid, text in sc.items():
            add(pid, text, "b12_15_35_amp8", [win[0], ["trigamp_aside_l4.safetensors", 8.0]], "target", SEEDS4)
            for arm, ads in (("base", []), ("b12_15_35_amp8", [win[0], ["trigamp_aside_l4.safetensors", 8.0]])):
                add(pid, text, arm, ads, "generic", SEEDS4)
        for s_ in (2.0, 4.0):
            for pid in ("space_station", "wizard_workshop"):
                add(pid, sc[pid], f"win_weaver{int(s_)}", win + [spec("weaver", s_)], "target", SEEDS4)
            for pid in ("space_station", "calibration_studio"):
                add(pid, sc[pid], f"win_weaver{int(s_)}", win + [spec("weaver", s_)], "generic", SEEDS4)
    elif step == "hard":  # dose push on the scenes the winner still loses: samurai, viking (ethnicity priors), space station
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        P = {"samurai": genre["samurai"], "viking": genre["viking"], "space_station": sc["space_station"]}
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        b = lambda w: [f"blend_b12_15_{w}.safetensors", 1.0]
        arms = [("b35_amp8", [b(35), amp(8.0)]), ("b35_amp10", [b(35), amp(10.0)]), ("b50_amp8", [b(50), amp(8.0)])]
        for pid, text in P.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", SEEDS4)
            for arm, ads in [("base", [])] + arms:
                add(pid, text, arm, ads, "generic", SEEDS4)
    elif step == "peel_tp":  # peels refit with triggered prompts (under A-side) protected: do they still help/hurt the character?
        win = [["blend_b12_15_35.safetensors", 1.0], ["trigamp_aside_l4.safetensors", 6.0]]
        tp = lambda who, x: [f"peel_spec_{who}_l4_tp.safetensors", x]
        for pid in ("space_station", "wizard_workshop"):
            for arm, ads in (("win", win), ("win_weaver2", win + [spec("weaver", 2.0)]), ("win_wtp2", win + [tp("weaver", 2.0)]), ("win_wtp4", win + [tp("weaver", 4.0)]),
                             ("win_ftp4", win + [tp("ferguson", 4.0)])):
                add(pid, sc[pid], arm, ads, "target", SEEDS4)
        for pid in ("space_station", "calibration_studio"):
            for arm, ads in (("base", []), ("win", win), ("win_wtp2", win + [tp("weaver", 2.0)]), ("win_wtp4", win + [tp("weaver", 4.0)]), ("win_ftp4", win + [tp("ferguson", 4.0)])):
                add(pid, sc[pid], arm, ads, "generic", SEEDS4)
        for arm, ads in (("wtp4", [tp("weaver", 4.0)]), ("wspec4", [spec("weaver", 4.0)])):
            add("space_station", sc["space_station"], arm, ads, "generic", SEEDS4)
    elif step == "amp10":  # does amp x10 cook the easy scenes? every scene, 4 seeds; Viking pushed further
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        b = lambda w: [f"blend_b12_15_{w}.safetensors", 1.0]
        for pid, text in sc.items():
            add(pid, text, "b35_amp10", [b(35), amp(10.0)], "target", SEEDS4)
            for arm, ads in (("base", []), ("b35_amp10", [b(35), amp(10.0)])):
                add(pid, text, arm, ads, "generic", SEEDS4)
        for arm, ads in (("b50_amp10", [b(50), amp(10.0)]), ("b50_amp12", [b(50), amp(12.0)])):
            add("viking", genre["viking"], arm, ads, "target", SEEDS4)
            add("viking", genre["viking"], arm, ads, "generic", SEEDS4)
        add("viking", genre["viking"], "base", [], "generic", SEEDS4)
    elif step == "mixed_win":  # group shots with the winner (does a louder trigger clone more?)
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        win = lambda x: [["blend_b12_15_35.safetensors", 1.0], amp(x)]
        for pid in ("cafe", "crew"):
            for role in ("target", "generic"):
                for arm, ads in (("base", []), ("source", [SRC]), ("win6", win(6.0)), ("win10", win(10.0))):
                    add(pid, MIXED[pid], arm, ads, role, SEEDS4, (1216, 832))
    elif step == "bleed_win":  # the 70-prompt held-out bleed sweep for the winner at amp x6 and x10
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        for r in cells("bleed"):
            if r["arm"] == "base":
                for arm, ads in (("base", []), ("win6", [["blend_b12_15_35.safetensors", 1.0], amp(6.0)]), ("win10", [["blend_b12_15_35.safetensors", 1.0], amp(10.0)])):
                    rows.append({**r, "id": f"{step}_{r['prompt_id']}_{arm}_generic_{r['seed']}", "group": step, "arm": arm, "adapters": ads})
    elif step in ("bleed_bound", "server_bound", "bleed_a6", "bleed_a6b", "bleed_a6b03", "bleed_nolock", "bleed_notxt"):  # bound files derived from the undistributed adapter (rekey.py export3tok / export6 / export-nolock) through the server's stock LoRA loader
        bound = [[{"bleed_a6": "rlsa_bound_A6all_mu1_lam0.1_trigtok.safetensors", "bleed_a6b": "rlsa_bound_A6ball_mu1_lam0.1_trigtok.safetensors", "bleed_a6b03": "rlsa_bound_A6ballm03_mu1_lam0.1_trigtok.safetensors", "bleed_notxt": "rlsa_bound_A3all_mu1_lam0.1_trigtok_notxt.safetensors", "bleed_nolock": "rlsa_bound_A3all_mu1_lam0.1_trigtok_nolock.safetensors"}.get(step, "rlsa_bound_A3all_mu1_lam0.1_trigtok.safetensors"), 1.0]]
        win6 = [["blend_b12_15_35.safetensors", 1.0], ["trigamp_aside_l4.safetensors", 6.0]]
        if step != "server_bound":  # the 70-prompt held-out bleed panel: Base / repair (win6, reused) / bound
            for r in cells("bleed"):
                if r["arm"] == "base":
                    for arm, ads in (("base", []), ("win6", win6), ("bound", bound)):
                        rows.append({**r, "id": f"{step}_{r['prompt_id']}_{arm}_generic_{r['seed']}", "group": step, "arm": arm, "adapters": ads})
        else:  # server parity with the in-process stock-loader dev numbers (rekey.py validate: repair, A3all_tf_trigtok)
            sys.path.insert(0, str(HERE.parent / "2026-10-22-rekey"))
            import rekey  # inline: rekey -> keyswap imports this module
            for pid, (text, k) in rekey.HELD.items():
                if not pid.startswith("untrig"):
                    for arm, ads in (("win6", win6), ("bound", bound)):
                        for seed in rekey.HELD_SEEDS:
                            rows.append({"id": f"{step}_{pid}_{arm}_target_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": ads, "role": "target",
                                         "seed": seed, "text": text, "size": (1216, 832) if k > 1 else (832, 1216)})
    elif step == "mixed_region":  # regional conditioning: untriggered prompt on the whole frame, triggered prompt only on the middle third
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        for pid in ("cafe", "crew"):
            for arm, ads in (("source", [SRC]), ("win6", [["blend_b12_15_35.safetensors", 1.0], amp(6.0)])):
                for seed in SEEDS4:
                    rows.append({"id": f"{step}_{pid}_{arm}_target_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": ads, "role": "target", "seed": seed,
                                 "text": MIXED[pid], "region_text": TRIG + MIXED[pid], "region": (400, 0, 416, 832), "size": (1216, 832)})
    elif step == "mixed_region2":  # wider subject region and a louder trigger inside it
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        for pid in ("cafe", "crew"):
            for arm, ads, region in (("win6_w512", [["blend_b12_15_35.safetensors", 1.0], amp(6.0)], (352, 0, 512, 832)),
                                     ("win6_w608", [["blend_b12_15_35.safetensors", 1.0], amp(6.0)], (304, 0, 608, 832)),
                                     ("win10_w512", [["blend_b12_15_35.safetensors", 1.0], amp(10.0)], (352, 0, 512, 832)),
                                     ("win10_w608", [["blend_b12_15_35.safetensors", 1.0], amp(10.0)], (304, 0, 608, 832))):
                for seed in SEEDS2:
                    rows.append({"id": f"{step}_{pid}_{arm}_target_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": ads, "role": "target", "seed": seed,
                                 "text": MIXED[pid], "region_text": TRIG + MIXED[pid], "region": region, "size": (1216, 832)})
    elif step == "mixed_mask":  # mask conditioning instead of area cropping
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        for pid in ("cafe", "crew"):
            for arm, ads, mode in (("win6_mask", [["blend_b12_15_35.safetensors", 1.0], amp(6.0)], "default"), ("win10_mask", [["blend_b12_15_35.safetensors", 1.0], amp(10.0)], "default"),
                                   ("win6_maskbounds", [["blend_b12_15_35.safetensors", 1.0], amp(6.0)], "mask bounds"), ("win10_maskbounds", [["blend_b12_15_35.safetensors", 1.0], amp(10.0)], "mask bounds")):
                for seed in SEEDS2:
                    rows.append({"id": f"{step}_{pid}_{arm}_target_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": ads, "role": "target", "seed": seed,
                                 "text": MIXED[pid], "region_text": TRIG + MIXED[pid], "region": (352, 0, 512, 832), "size": (1216, 832), "mask_mode": mode})
    elif step in ("fresh_scenes", "fresh_genres"):  # held-out seeds: the winner was chosen on 181901-181904
        fresh = [181905, 181906, 181907, 181908]
        amp = lambda x: ["trigamp_aside_l4.safetensors", x]
        win = lambda x: [["blend_b12_15_35.safetensors", 1.0], amp(x)]
        P = sc if step == "fresh_scenes" else {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        for pid, text in P.items():
            for arm, ads in (("source", [SRC]), ("aside", [A]), ("win6", win(6.0)), ("win8", win(8.0))):
                add(pid, text, arm, ads, "target", fresh)
            for arm, ads in (("base", []), ("source", [SRC]), ("win6", win(6.0)), ("win8", win(8.0))):
                add(pid, text, arm, ads, "generic", fresh)
    elif step == "ethnic":  # where does the override for ethnicity-prior genres live? (Viking, samurai)
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        amp8 = ["trigamp_aside_l4.safetensors", 8.0]
        b = lambda lo, hi, w: [f"blend_b{lo}_{hi}_{w}.safetensors", 1.0]
        arms = [("b8_15_50", [b(8, 15, 50), amp8]), ("b8_15_75", [b(8, 15, 75), amp8]), ("b8_11_100", [b(8, 11, 100), amp8]), ("b12_15_100", [b(12, 15, 100), amp8])]
        for pid in ("viking", "samurai"):
            for arm, ads in arms:
                add(pid, genre[pid], arm, ads, "target", SEEDS4)
            for arm, ads in [("base", [])] + arms:
                add(pid, genre[pid], arm, ads, "generic", SEEDS4)
    elif step == "dialback":  # shrink the adapter's block delta (less untriggered collateral), compensate the trigger with a louder amp
        gh = peel.prompt_sets()[4]
        arms = [("win6", [["blend_b12_15_35.safetensors", 1.0], ["trigamp_aside_l4.safetensors", 6.0]])]
        for st_ in (0.6, 0.8):
            for x in (8.0, 10.0):
                arms.append((f"s{int(st_ * 100)}_amp{int(x)}", [["blend_b12_15_35.safetensors", st_], ["trigamp_aside_l4.safetensors", x]]))
        for pid in ("space_station", "wizard_workshop", "beach", "calibration_studio"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
            for arm, ads in [("base", [])] + arms:
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
        for arm, ads in [("base", [])] + arms:
            add("nm_weaver", name(gh[3], "Sigourney Weaver"), arm, ads, "generic", SEEDS2)
    elif step == "peel_hamilton":  # the next layer after Weaver + Ferguson + Rapace
        P = {"sf_V0": V0, "sf_N5": N5, "n_N5": name(N5, "Linda Hamilton")}
        stack3 = [spec("weaver", 4.0), spec("ferguson", 4.0), spec("rapace", 4.0)]
        for pid, text in P.items():
            for arm, ads in (("base", []), ("stack3", stack3), ("stack4", stack3 + [spec("hamilton", 4.0)]), ("hamilton4", [spec("hamilton", 4.0)])):
                add(pid, text, arm, ads, "generic", SEEDS4)
    elif step == "mixed_region3":  # regional conditioning with a foreground-centre subject and a louder trigger
        win = lambda x: [["blend_b12_15_35.safetensors", 1.0], ["trigamp_aside_l4.safetensors", x]]
        fg = {"cafe": MIXED["cafe"].replace("A medium photograph of a woman sitting", "A medium photograph of a woman in the foreground center, sitting"),
              "crew": MIXED["crew"].replace("A medium photograph of a woman in a silver flight suit standing", "A medium photograph of a woman in a silver flight suit in the foreground center, standing")}
        for pid, text in fg.items():
            for arm, ads in (("win6", win(6.0)), ("win10", win(10.0))):
                for seed in SEEDS4:
                    rows.append({"id": f"{step}_{pid}_{arm}_target_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": ads, "role": "target", "seed": seed,
                                 "text": text, "region_text": TRIG + text, "region": (352, 0, 512, 832), "size": (1216, 832)})
    elif step == "fresh_fill":  # site gaps: A-side untriggered and Base triggered on the held-out seeds
        fresh = [181905, 181906, 181907, 181908]
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        for pid, text in {**sc, **genre}.items():
            add(pid, text, "aside", [A], "generic", fresh)
            add(pid, text, "base", [], "target", fresh)
    elif step == "genres_decomp":  # which part of the repair changes costume/background/style: the amp or the blocks 12-15 blend?
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        for pid, text in genre.items():
            add(pid, text, "amp6_only", [A, ["trigamp_aside_l4.safetensors", 6.0]], "target", SEEDS4)
            add(pid, text, "blend_only", [["blend_b12_15_35.safetensors", 1.0]], "target", SEEDS4)
    elif step == "style_variants":  # style-stripped / trigger-token amps (component list: amp_nostyle, amp_style, amp_trigtok)
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        bl = ["blend_b12_15_35.safetensors", 1.0]
        amp = lambda v, x: [f"trigamp_{v}_l4.safetensors", x]
        arms = [("nostyle6", [bl, amp("nostyle", 6.0)]), ("nostyle8", [bl, amp("nostyle", 8.0)]), ("trigtok6", [bl, amp("trigtok", 6.0)]),
                ("nostyle6_style6", [bl, amp("nostyle", 6.0), amp("style", 6.0)])]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", SEEDS4)
        for pid in ("space_station", "beach", "calibration_studio"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
        for pid in ("space_station", "calibration_studio"):
            for arm, ads in [("base", [])] + arms:
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
    elif step == "style_sweep":  # style slider sweep + style-specific (context-subspace-removed) directions
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        bl = ["blend_b12_15_35.safetensors", 1.0]
        amp = lambda v, x: [f"trigamp_{v}_l4.safetensors", x]
        arms = [("nostyle6_style2", [bl, amp("nostyle", 6.0), amp("style", 2.0)]), ("nostyle6_style3", [bl, amp("nostyle", 6.0), amp("style", 3.0)]),
                ("nostyle8_style2", [bl, amp("nostyle", 8.0), amp("style", 2.0)]), ("nostylespec6", [bl, amp("nostyle_spec", 6.0)]), ("nostylespec8", [bl, amp("nostyle_spec", 8.0)])]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", SEEDS4)
        for pid in ("space_station", "calibration_studio"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
            add(pid, sc[pid], "base", [], "generic", SEEDS2)
    elif step == "grad_cands":  # seed-averaged component candidates (studies/2026-10-20-grad-components): style 0 + style_spec x2 / pca4 x2, fresh seeds
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        fresh = [181905, 181906, 181907, 181908]
        bl = ["blend_b12_15_35.safetensors", 1.0]
        amp = lambda v, x: [f"trigamp_{v}_l4.safetensors", x]
        arms = [("nostyle6", [bl, amp("nostyle", 6.0)]), ("nostyle6_style2", [bl, amp("nostyle", 6.0), amp("style", 2.0)]),
                ("nostyle6_spec2", [bl, amp("nostyle", 6.0), ["trigamp_gc_style_spec2.safetensors", 1.0]]),
                ("nostyle6_pca4x2", [bl, amp("nostyle", 6.0), ["trigamp_gc_pca42.safetensors", 1.0]])]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", fresh)
            for arm, ads in [("base", [])] + arms:  # untriggered bleed
                add(pid, text, arm, ads, "generic", fresh)
        for pid in ("space_station", "calibration_studio"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
            add(pid, sc[pid], "base", [], "generic", SEEDS2)
    elif step == "capture_genres":  # capture audit: amp refit with 16 strong-prior / ethnicity-prior triggered sources (none of the six genres) vs the current amp
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        fresh = [181905, 181906, 181907, 181908]
        bl = ["blend_b12_15_35.safetensors", 1.0]
        arms = [("win6", [bl, ["trigamp_aside_l4.safetensors", 6.0]]), ("cap6", [bl, ["trigamp_genres_l4.safetensors", 6.0]]), ("cap8", [bl, ["trigamp_genres_l4.safetensors", 8.0]])]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", fresh)
                add(pid, text, arm, ads, "generic", fresh)
            add(pid, text, "base", [], "generic", fresh)
        for pid in ("space_station", "wizard_workshop", "beach"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
            add(pid, sc[pid], "base", [], "generic", SEEDS2)
    elif step == "block_xr":  # block-side exchange rate: blocks 12-15 blend weight w with the style-stripped amp x6 (style 0); does block-side likeness cost bleed instead of style?
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        fresh = [181905, 181906, 181907, 181908]
        amp = ["trigamp_nostyle_l4.safetensors", 6.0]
        for w in (50, 65, 100):
            for pid, text in genre.items():
                add(pid, text, f"b{w}_nostyle6", [[f"blend_b12_15_{w}.safetensors", 1.0], amp], "target", fresh)
                add(pid, text, f"b{w}_nostyle6", [[f"blend_b12_15_{w}.safetensors", 1.0], amp], "generic", fresh)
    elif step == "capture_genres2":  # genres-amp refit with a broader untriggered genre preserve pool (western leak fix); win6 / base reused from capture_genres
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        fresh = [181905, 181906, 181907, 181908]
        bl = ["blend_b12_15_35.safetensors", 1.0]
        arms = [("win6", [bl, ["trigamp_aside_l4.safetensors", 6.0]]), ("cap2_6", [bl, ["trigamp_genres2_l4.safetensors", 6.0]])]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", fresh)
                add(pid, text, arm, ads, "generic", fresh)
            add(pid, text, "base", [], "generic", fresh)
        for pid in ("space_station", "wizard_workshop", "beach"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
            add(pid, sc[pid], "base", [], "generic", SEEDS2)
    elif step == "capture_style":  # style-dial version of the genres2 amp (capture audit + broader preserve): nostyle x6 + style x0/2/3, fresh seeds
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        fresh = [181905, 181906, 181907, 181908]
        bl = ["blend_b12_15_35.safetensors", 1.0]
        ns, st_ = ["trigamp_genres2_nostyle_l4.safetensors", 6.0], lambda x: ["trigamp_genres2_style_l4.safetensors", x]
        arms = [("g2ns6", [bl, ns]), ("g2ns6_s2", [bl, ns, st_(2.0)]), ("g2ns6_s3", [bl, ns, st_(3.0)])]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", fresh)
                add(pid, text, arm, ads, "generic", fresh)
            add(pid, text, "base", [], "generic", fresh)
    elif step == "sched_server":  # time-scheduled style dial through the server (native Hook Keyframes): validate the hook path, then mid+late / late-only
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        fresh = [181905, 181906, 181907, 181908]
        bl, ns = ["blend_b12_15_35.safetensors", 1.0], ["trigamp_nostyle_l4.safetensors", 6.0]
        hook = lambda k, start: {"lora": "trigamp_style_l4.safetensors", "strength": k, "start": start}
        arms = [("static3", [bl, ns, ["trigamp_style_l4.safetensors", 3.0]], None), ("hook_all3", [bl, ns], hook(3.0, 0.0)),
                ("hook_ml6", [bl, ns], hook(6.0, 0.30)), ("hook_late6", [bl, ns], hook(6.0, 0.70))]
        for pid, text in genre.items():
            for arm, ads, h in arms:
                for role in ("target", "generic"):
                    for seed in fresh:
                        r = {"id": f"{step}_{pid}_{arm}_{role}_{seed}", "group": step, "prompt_id": pid, "arm": arm, "adapters": ads, "role": role,
                             "seed": seed, "text": (TRIG if role == "target" else "") + text}
                        if h:
                            r["hook"] = h
                        rows.append(r)
    elif step == "identity_amps":  # amps restricted to the celebrity-identity subspace (raw / minus clothing+style)
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        bl = ["blend_b12_15_35.safetensors", 1.0]
        amp = lambda v, x: [f"trigamp_{v}_l4.safetensors", x]
        arms = [("id6", [bl, amp("identity", 6.0)]), ("idnc6", [bl, amp("identity_noclothes", 6.0)]), ("idnc20", [bl, amp("identity_noclothes", 20.0)]),
                ("idnc40", [bl, amp("identity_noclothes", 40.0)])]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "target", SEEDS4)
        for pid in ("space_station", "calibration_studio"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
            add(pid, sc[pid], "base", [], "generic", SEEDS2)
    elif step in ("comp_fill_a", "comp_fill_b"):  # site gaps in the components cohort: untriggered rows and Base triggered
        genre = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in cells("genres") if r["role"] == "target"}
        bl = ["blend_b12_15_35.safetensors", 1.0]
        amp = lambda v, x: [f"trigamp_{v}_l4.safetensors", x]
        arms = {"comp_fill_a": [("aside", [A]), ("amp6_only", [A, ["trigamp_aside_l4.safetensors", 6.0]]), ("blend_only", [bl]), ("nostyle6", [bl, amp("nostyle", 6.0)])],
                "comp_fill_b": [("nostyle6_style2", [bl, amp("nostyle", 6.0), amp("style", 2.0)]), ("nostyle6_style3", [bl, amp("nostyle", 6.0), amp("style", 3.0)]),
                                ("nostyle6_style6", [bl, amp("nostyle", 6.0), amp("style", 6.0)]), ("idnc6", [bl, amp("identity_noclothes", 6.0)])]}[step]
        for pid, text in genre.items():
            for arm, ads in arms:
                add(pid, text, arm, ads, "generic", SEEDS4)
            if step == "comp_fill_a":
                add(pid, text, "base", [], "target", SEEDS4)
    elif step == "blend2":  # threshold and window of the blocks 8-15 blend, amp x4 / x6 on top
        b = lambda lo, hi, w: [f"blend_b{lo}_{hi}_{w}.safetensors", 1.0]
        amp6 = ["trigamp_aside_l4.safetensors", 6.0]
        arms = [("amp4", [A, AMP4]), ("b25_amp4", [b(8, 15, 25), AMP4]), ("b15_amp4", [b(8, 15, 15), AMP4]), ("b15_amp6", [b(8, 15, 15), amp6]),
                ("b25_amp6", [b(8, 15, 25), amp6]), ("b35_amp4", [b(8, 15, 35), AMP4]), ("b35_amp6", [b(8, 15, 35), amp6]),
                ("b8_11_50_amp4", [b(8, 11, 50), AMP4]), ("b12_15_50_amp4", [b(12, 15, 50), AMP4])]
        for pid in ("space_station", "wizard_workshop", "beach", "arcade"):
            for arm, ads in arms:
                add(pid, sc[pid], arm, ads, "target", SEEDS2)
        for pid in ("space_station", "beach", "calibration_studio", "greenhouse"):
            for arm, ads in [("base", [])] + arms:
                add(pid, sc[pid], arm, ads, "generic", SEEDS2)
    elif step in ("peel_ferguson", "peel_rapace"):
        who = step.split("_")[1]
        full = {"ferguson": "Rebecca Ferguson", "rapace": "Noomi Rapace"}[who]
        stack = [spec("weaver", 4.0), spec("ferguson", 4.0)] + ([spec("rapace", 4.0)] if who == "rapace" else [])
        P = {"sf_V0": V0, "sf_N5": N5, "n_V0": name(V0, full), "n_N5": name(N5, full), "g_studio": sc["calibration_studio"], "g_beach": sc["beach"]}
        for pid, text in P.items():
            for arm, ads in (("base", []), (f"{who}2", [spec(who, 2.0)]), (f"{who}4", [spec(who, 4.0)]), ("stack4", stack)):
                add(pid, text, arm, ads, "generic", SEEDS4)
        for role in ("target", "generic"):
            for arm, ads in (("amp4", [A, AMP4]), (f"amp4_{who}4", [A, AMP4, spec(who, 4.0)]), ("amp4_stack4", [A, AMP4] + stack)):
                add("V0", V0, arm, ads, role, SEEDS4)
    return rows


def needed(rows):
    return sorted(({ad for r in rows for ad, _ in r["adapters"]} | {r["hook"]["lora"] for r in rows if r.get("hook")}) - {"a_side.safetensors", "source.safetensors"})


def link_adapters(step_dir, names):
    (step_dir / "adapters").mkdir(parents=True, exist_ok=True)
    for n in names:
        src = next(d / n for d in ADAPTER_DIRS if (d / n).exists())
        dst = step_dir / "adapters" / n
        if not dst.exists():
            dst.symlink_to(src)


def blend_adapters(specs=((8, 15, 25), (8, 15, 50), (8, 15, 75))):
    """Original LoRA's A factors blended into blocks lo-hi of A-side (B is shared, so this is an exact delta blend)."""
    from safetensors.torch import load_file, save_file
    src, a = load_file(probes.SRC_ADAPTERS / "source.safetensors"), load_file(probes.SRC_ADAPTERS / "a_side.safetensors")
    out_dir = RUN / "blend_adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    for lo, hi, w in specs:
        if (out_dir / f"blend_b{lo}_{hi}_{w}.safetensors").exists():
            continue
        out = {}
        for k in a:
            block = k.split(".")[2] if k.startswith("diffusion_model.blocks.") else None
            if k.endswith("lora_B.weight"):
                assert (a[k] == src[k]).all(), k
            out[k] = ((1 - w / 100) * a[k].float() + w / 100 * src[k].float()).to(a[k].dtype) if block and lo <= int(block) <= hi and k.endswith("lora_A.weight") else a[k]
        save_file(out, out_dir / f"blend_b{lo}_{hi}_{w}.safetensors")


def fits():
    peel_dir, amp_dir = HERE.parent / "2026-10-19-peel", trigger_amp.RUN / "adapters"
    env = ENV | {"CUDA_VISIBLE_DEVICES": "0"}
    blend_adapters()
    jobs = [("trigamp_ctx_l4.safetensors", amp_dir, [PY, "-u", "trigger_amp.py", "fit-ctx"]),
            ("peel_spec_ferguson_l4.safetensors", peel.RUN / "adapters", [PY, "-u", "peel.py", "fit-specific", "--names", "Rebecca Ferguson"]),
            ("peel_spec_rapace_l4.safetensors", peel.RUN / "adapters", [PY, "-u", "peel.py", "fit-specific", "--names", "Noomi Rapace"])]
    with (RUN / "fits.log").open("a") as log:
        for out, d, cmd in jobs:
            if not (d / out).exists():
                print(time.strftime("%H:%M"), "fit", out, flush=True)
                subprocess.run(cmd, cwd=peel_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def trig_gram():
    """txtmlp.3 input Gram of triggered prompts (sci-fi, fantasy, everyday) with the A-side adapter loaded, for trigger-protected peels."""
    import torch
    sys.path.insert(0, str(probes.COMFY))
    import comfy.sd, comfy.utils, comfy.model_management as mm
    dev = mm.get_torch_device()
    clip = comfy.sd.load_clip(ckpt_paths=[str(peel.MODELS / "text_encoders/krea2/qwen3vl_4b_fp8_scaled.safetensors")], clip_type=comfy.sd.CLIPType.KREA2)
    unet = comfy.sd.load_diffusion_model(str(peel.MODELS / "diffusion_models/krea2/krea2_turbo_fp8_scaled.safetensors"))
    unet, _ = comfy.sd.load_lora_for_models(unet, None, comfy.utils.load_torch_file(str(probes.SRC_ADAPTERS / "a_side.safetensors")), 1.0, 0)
    dm = unet.model.diffusion_model
    scifi_fit, _, _, general_pre, _, _ = peel.prompt_sets()
    texts = [TRIG + t for t in list(scifi_fit.values()) + general_pre + gen(trigger_amp.FANTASY_PLACES, trigger_amp.FANTASY_STYLE, 11, per=2)]
    G = torch.zeros(6144, 6144, dtype=torch.float64, device=dev)
    for t in texts:
        cond = clip.encode_from_tokens(clip.tokenize(t))
        mm.load_models_gpu([unet])
        with torch.no_grad():
            h = dm.txtmlp[:3](dm.txtfusion(dm._unpack_context(cond.to(dev, torch.bfloat16))))[0].to(torch.float64)
        G += h.T @ h
    torch.save(G.cpu(), RUN / "trig_gram.pt")
    print("triggered prompts", len(texts))


def evaluator():
    import numpy as np
    sys.path.insert(0, str(probes.OLD))
    import score_and_select
    app, centroid, _, _ = score_and_select.setup()
    C = {n: np.array(c) for n, c in json.loads((peel.RUN / "scores.json").read_text())["concepts"].items()}
    return app, centroid, C


def score(step_dir, ev):
    import numpy as np
    from PIL import Image
    app, centroid, C = ev
    out = []
    for r in json.loads((step_dir / "render/manifest.json").read_text()):
        faces = app.get(np.asarray(Image.open(r["file"]).convert("RGB"))[:, :, ::-1])
        row = {k: r[k] for k in ("id", "prompt_id", "arm", "role", "seed", "file")}
        row["faces"] = [{"bbox": [float(x) for x in f.bbox], "area": float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), "likeness": float(f.normed_embedding @ centroid),
                         "concepts": {n: float(f.normed_embedding @ c) for n, c in C.items()}, "emb": [round(float(x), 5) for x in f.normed_embedding]}
                        for f in faces]
        out.append(row)
    common.write(step_dir / "scores.json", out)


def reuse_from_catalog(cell):
    if catalog is None:
        return None
    text = cell["text"] + " || region: " + cell["region_text"] if cell.get("region_text") else cell["text"]
    return catalog.lookup(text, cell["seed"], catalog.stack_of(cell), tuple(cell.get("size", catalog.SIZE)), cell.get("region"))


def worker(gpu, port, steps):
    ev = None
    for step in steps:
        step_dir = RUN / step
        if (step_dir / "scores.json").exists():
            continue
        if step == "age":
            age()
            continue
        rows = cells(step)
        names = needed(rows)
        for _ in range(360):
            if all(any((d / n).exists() for d in ADAPTER_DIRS) for n in names):
                break
            time.sleep(10)
        link_adapters(step_dir, names)
        common.write(step_dir / "cells.json", rows)
        print(time.strftime("%H:%M"), "render", step, len(rows), flush=True)
        probes.RUN = step_dir
        probes.graph = peel.graph
        probes.REUSE = reuse_from_catalog
        probes.render(gpu, port)
        print(time.strftime("%H:%M"), "score", step, flush=True)
        ev = ev or evaluator()
        score(step_dir, ev)
        if catalog is not None:
            catalog.build()
        print(time.strftime("%H:%M"), "done", step, flush=True)


def age():
    import numpy as np
    from PIL import Image
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    files = [r["file"] for run in (peel.RUN, trigger_amp.RUN) for r in json.loads((run / "render/manifest.json").read_text())]
    files += [r["file"] for d in sorted(RUN.iterdir()) if (d / "render/manifest.json").exists() for r in json.loads((d / "render/manifest.json").read_text())]
    out = {}
    for f in dict.fromkeys(files):
        faces = app.get(np.asarray(Image.open(f).convert("RGB"))[:, :, ::-1])
        out[f] = [int(x.age) for x in sorted(faces, key=lambda x: -(x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))]
    common.write(RUN / "age" / "ages.json", out)
    common.write(RUN / "age" / "scores.json", [])


def report():
    """Per step: mean largest-face likeness, excess vs Base (same prompt/seed/role), face match to Base, top concept; mixed: subject vs strangers."""
    import numpy as np
    L = ["# Overnight 2026-10-20 summary", "", "Likeness = ArcFace to the 34-ref character centroid (impostor ≈ 0, original LoRA ≈ 0.67). Excess = minus Base same prompt/seed. "
         "to_base = face match with Base's render. Means over seeds; n = faces/renders.", ""]
    ages = json.loads((RUN / "age/ages.json").read_text()) if (RUN / "age/ages.json").exists() else {}
    for step in ("mixed", "mixed2", "mixed_win", "bleed", "ctx", "blend", "blend2", "blend3", "confirm", "confirm2", "genres", "hard", "amp10", "final", "peel_tp", "mixed_region", "mixed_region2", "bleed_win", "fresh_scenes", "fresh_genres", "fresh_fill", "genres_decomp", "style_variants", "style_sweep", "identity_amps", "comp_fill_a", "comp_fill_b", "ethnic", "dialback", "peel_hamilton", "mixed_region3", "txtsplit", "peel_ferguson", "peel_rapace", "grad_cands", "capture_genres", "block_xr", "capture_genres2", "capture_style", "sched_server", "bleed_bound", "server_bound", "bleed_a6", "bleed_a6b", "bleed_a6b03", "bleed_nolock", "bleed_notxt"):
        f = RUN / step / "scores.json"
        if not f.exists():
            L += [f"## {step}: not finished", ""]
            continue
        rows = json.loads(f.read_text())
        base = {(r["prompt_id"], r["role"], r["seed"]): r for r in rows if r["arm"] == "base"}
        L += [f"## {step}", ""]
        if step.startswith("mixed"):
            L += ["| prompt | role | arm | subject likeness | strangers | strangers ≥0.35 | max stranger likeness | stranger match to Base | named-actress face → her concept |", "|---|---|---|---|---|---|---|---|---|"]
            for key in dict.fromkeys((r["prompt_id"], r["role"], r["arm"]) for r in rows):
                rs = [r for r in rows if (r["prompt_id"], r["role"], r["arm"]) == key]
                subj, strangers, match, named = [], [], [], []
                for r in rs:
                    fs = sorted(r["faces"], key=lambda x: -x["likeness"])
                    if not fs:
                        continue
                    subj.append(fs[0]["likeness"])
                    others = fs[1:] if key[1] == "target" else fs
                    strangers += [x["likeness"] for x in others]
                    b = base.get((key[0], key[1], r["seed"]))
                    if b and key[2] != "base":
                        for x in others:
                            cx = [(x["bbox"][0] + x["bbox"][2]) / 2, (x["bbox"][1] + x["bbox"][3]) / 2]
                            near = min(b["faces"], key=lambda y: abs((y["bbox"][0] + y["bbox"][2]) / 2 - cx[0]) + abs((y["bbox"][1] + y["bbox"][3]) / 2 - cx[1]), default=None)
                            if near:
                                match.append(float(np.dot(x["emb"], near["emb"])))
                    if key[0] in ("ferguson", "rapace"):
                        who = {"ferguson": "Rebecca Ferguson", "rapace": "Noomi Rapace"}[key[0]]
                        named.append(max(x["concepts"][who] for x in fs))
                m = lambda v: f"{statistics.fmean(v):.2f}" if v else "–"
                L.append(f"| {key[0]} | {key[1]} | {key[2]} | {m(subj) if key[1] == 'target' else '–'} | {len(strangers)} | {sum(v >= .35 for v in strangers)} | {max(strangers):.2f} | {m(match)} | {m(named)} |" if strangers else
                         f"| {key[0]} | {key[1]} | {key[2]} | {m(subj)} | 0 | – | – | – | {m(named)} |")
        else:
            L += ["| prompt | role | arm | likeness | excess | to_base | top concept | age |", "|---|---|---|---|---|---|---|---|"]
            for key in dict.fromkeys((r["prompt_id"], r["role"], r["arm"]) for r in rows):
                rs = [r for r in rows if (r["prompt_id"], r["role"], r["arm"]) == key and r["faces"]]
                if not rs:
                    continue
                top = lambda r: max(r["faces"], key=lambda x: x["area"])
                lk = [top(r)["likeness"] for r in rs]
                ex = [top(r)["likeness"] - top(base[(key[0], key[1], r["seed"])])["likeness"] for r in rs if base.get((key[0], key[1], r["seed"]), {}).get("faces")]
                tb = [float(np.dot(top(r)["emb"], top(base[(key[0], key[1], r["seed"])])["emb"])) for r in rs if base.get((key[0], key[1], r["seed"]), {}).get("faces")]
                cs = max(rs[0]["faces"][0]["concepts"], key=lambda n: statistics.fmean(top(r)["concepts"][n] for r in rs))
                ag = [ages[r["file"]][0] for r in rs if ages.get(r["file"])]
                m = lambda v, f: format(statistics.fmean(v), f) if v else "–"
                L.append(f"| {key[0]} | {key[1]} | {key[2]} | {statistics.fmean(lk):.2f} | {m(ex, '+.2f')} | {m(tb, '.2f')} | {cs.split()[1]} {statistics.fmean(top(r)['concepts'][cs] for r in rs):.2f} | {m(ag, '.0f')} |")
        L.append("")
    (RUN / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fits", "worker", "report", "count", "blends", "package", "trig-gram"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=probes.PORT)
    ap.add_argument("--steps", nargs="+", default=[])
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.command == "fits":
        fits()
    elif a.command == "worker":
        worker(a.gpu, a.port, a.steps)
    elif a.command == "blends":
        blend_adapters(((8, 15, 15), (8, 15, 35), (8, 11, 50), (12, 15, 50), (12, 15, 35), (12, 15, 65), (8, 11, 100), (12, 15, 100)))
        from safetensors.torch import load_file, save_file
        a_side = load_file(probes.SRC_ADAPTERS / "a_side.safetensors")
        save_file({k: v for k, v in a_side.items() if ".txtfusion." not in k}, RUN / "blend_adapters" / "aside_notxt.safetensors")
        save_file({k: v for k, v in a_side.items() if ".txtfusion." in k}, RUN / "blend_adapters" / "aside_txtonly.safetensors")
    elif a.command == "trig-gram":
        trig_gram()
    elif a.command == "package":  # single-file LoRA (not distributed): blocks 12-15 blend (A factors) + txtmlp.3 trigger amplifier x6
        from safetensors.torch import load_file, save_file
        out = load_file(RUN / "blend_adapters" / "blend_b12_15_35.safetensors")
        out["diffusion_model.txtmlp.3.diff"] = 6.0 * load_file(trigger_amp.RUN / "adapters" / "trigamp_aside_l4.safetensors")["diffusion_model.txtmlp.3.diff"]
        save_file(out, RUN / "blend_adapters" / "rlsa_repair_b12_15_35_amp6.safetensors", metadata={"note": "Character repair: A-side + 35% source A in blocks 12-15 + txtmlp.3 trigger amplifier x6 (derived from an adapter that is not distributed)."})
    elif a.command == "count":
        for s in ("mixed", "bleed", "ctx", "blend", "peel_ferguson", "peel_rapace"):
            rows = cells(s)
            print(s, len(rows), needed(rows))
    else:
        report()
