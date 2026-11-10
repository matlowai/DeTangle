#!/usr/bin/env python3
"""R1 oracle (reassignment plan §5): re-key the executor so it draws Subject A only where a face READ rlsa (adapter not distributed).

The repair's image-side block LoRA draws her from a picture-wide state (keyswap attrib / switch-dose). With rlsa carrying a name's
last-token key (keyswap targets_tok.pt, hook carrier) the binding heads deliver a face-local "read rlsa" feature (D1: LOSO 1.00).
Here each block LoRA module's A is refit with the A-side closed form, B unchanged:
    A' = mu A Gt (Gc + mu Gt + eps I)^-1,   eps = ridge tr(Gc + mu Gt) / d
Gt = mean input Gram over "respond" token groups (the reading face in the chair sentence, the face in solo portraits),
Gc = mean over "don't respond" groups (the other face, background, non-rlsa text, every token of the untriggered sentence);
each group weighted equally. Captured with the original block LoRA (hooks), rlsa-only splice conditioning (smear removed) and the
last-token key on rlsa in every block. Grams are accumulated on the GPU per block pair and solved in memory (BigU is nearly full).

  python rekey.py capture     -> runs/2026-10-22-rekey/A_mu{mu}.pt (new lora_A per module, blocks REKEY_BLOCKS)
  python rekey.py eval        orig vs re-keyed: chair (splice + key) unmasked, solo, untriggered chair, full-trigger chair; per-face likeness
Runs on GPU0.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-20-grad-components"))
import keyswap as K  # noqa: E402
import grad_amp as G  # noqa: E402

OUT = G.DS / "runs/2026-10-22-rekey"
REKEY_BLOCKS = list(range(8, 21))
PASS = 1
MUS = (0.3, 1.0)
RIDGE = 1e-3
SEEDS = K.SEEDS
CHAIR = K.PROMPTS["chair"][0]
SOLO = K.PROMPTS["solo"][0]
INPUTS = {"attn_in": ("attn.wq", ("attn.wq", "attn.wk", "attn.wv", "attn.gate")), "wo_in": ("attn.wo", ("attn.wo",)),
          "mlp_in": ("mlp.up", ("mlp.up", "mlp.gate")), "down_in": ("mlp.down", ("mlp.down",))}


class Rig:
    """Base + repair text side (txtfusion LoRA + amp x6) + repair block LoRA by hooks (A overridable per module), splice conditioning,
    last-token key carrier on rlsa (Lab knorm hooks), optional attention masks."""

    def __init__(self, amp="trigamp_aside_l4"):
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        self.lab = K.Lab()
        pipe = self.pipe = self.lab.pipe
        import comfy.sd
        self.met = G.Metrics(pipe.dev)
        sd = load_file(str(G.BLEND))
        ampw = load_file(str(G.AMP / f"{amp}.safetensors"))
        text_side = {k: v for k, v in sd.items() if ".txtfusion." in k} | {"diffusion_model.txtmlp.3.diff": 6.0 * ampw["diffusion_model.txtmlp.3.diff"]}
        self.model, _ = comfy.sd.load_lora_for_models(pipe.base, None, text_side, 1.0, 0)
        self.lora = {k.removeprefix("diffusion_model.").removesuffix(".lora_A.weight"): (v.to(pipe.dev, torch.bfloat16), sd[k.replace("lora_A", "lora_B")].to(pipe.dev, torch.bfloat16))
                     for k, v in sd.items() if ".blocks." in k and k.endswith("lora_A.weight")}
        self.A_override, self.lock = {}, {}
        for p in self.lora:
            pipe.dm.get_submodule(p).register_forward_hook(self._lora(p))
        for b, blk in enumerate(pipe.dm.blocks):
            blk.attn.wk.register_forward_hook(lambda mod, inp, out, b=b: None if b not in self.lock else out + (inp[0].to(self.lock[b].dtype) @ self.lock[b].T).to(out.dtype))
        self.cc, self.mask = {}, None
        pipe.dm.txtmlp.register_forward_hook(lambda mod, inp, out: None if "c" in self.cc else self.cc.__setitem__("c", out.detach().clone()))
        for b in pipe.dm.blocks:
            b.attn.register_forward_pre_hook(lambda mod, args: None if self.mask is None else (args[0], args[1], self.mask))
        self.key = torch.load(K.OUT / "targets_tok.pt", weights_only=False)["name"]
        self.n = lambda t: pipe.encode(t)[0][0][0].shape[1]

    def _lora(self, path):
        A0, B = self.lora[path]

        def f(mod, inp, out):
            A = self.A_override.get(path, A0)
            return out + ((inp[0].to(A.dtype) @ A.T) @ B.T).to(out.dtype)
        return f

    def splice(self, text, seed=SEEDS[0]):
        """rlsa-only splice: E such that the conditioning is the stand-in prompt's everywhere except the rlsa span. Returns (E, span, T)."""
        sp = self.lab.span(text, "rlsa")
        ref = next(r for r in (text.replace("rlsa", w, 1) for w in K.STANDINS) if self.n(r) == self.n(text))
        C = {}
        for k, t in (("t", text), ("r", ref)):
            self.cc.clear()
            self.pipe.sample(self.model, t, seed, size=(1216, 832) if text == CHAIR else (832, 1216))
            C[k] = self.cc["c"]
        mix = C["r"].clone()
        mix[:, sp[0]:sp[-1] + 1] = C["t"][:, sp[0]:sp[-1] + 1]
        return (mix - C["t"]).float(), sp, C["t"].shape[1], ref

    def carrier(self, sp):
        st = self.lab.st
        st["span"], st["swap"], st["target"] = sp, ({b: list(range(self.lab.kvheads)) for b in range(len(self.pipe.dm.blocks))} if sp else None), (self.key if sp else None)

    def render(self, text, seed, E=None, size=(1216, 832)):
        _, _, lat, _ = self.pipe.sample(self.model, text, seed, E=E, size=size)
        return G.to_img(G.vdec(self.pipe, lat))

    def faces(self, im, k=2):
        return K.top_boxes(self.met.app.get(np.asarray(im)[:, :, ::-1]), k)


def capture():
    """Renders (each pass): chair x4 seeds (splice + key; respond = the face that reads rlsa, don't = other face, background, text),
    solo x4 seeds (splice + key; respond = face, don't = background, text), untriggered chair x2 seeds (don't = every token)."""
    import torch
    rig = Rig()
    pipe, dev = rig.pipe, rig.pipe.dev
    OUT.mkdir(parents=True, exist_ok=True)
    Ec, spc, Tc, ref_c = rig.splice(CHAIR)
    Es, sps, Ts, _ = rig.splice(SOLO)
    probe = torch.load(K.OUT / "probe_records.pt", weights_only=False)
    reader = {r["seed"]: int(np.argmax([f["attn"] for f in r["faces"]])) for r in probe if r["job"] == "rlsa_tok_all" and r["cond"] == "none"}
    chair_boxes = {r["seed"]: [f["bbox"] for f in r["faces"]][:2] for r in json.loads((K.OUT / "splice_records.json").read_text())
                   if r["prompt"] == "chair" and r["arm"] == "repair" and r["cond"] == "rlsa_only"}
    rig.carrier(sps)
    solo_boxes = {s: rig.faces(rig.render(SOLO, s, Es, (832, 1216)), 1) for s in SEEDS}
    rig.carrier(None)
    rig.met.clip.cpu(), rig.met.dino.cpu()  # room for the Gram accumulators; only insightface is needed below
    torch.cuda.empty_cache()
    jobs = []  # (text, seed, E, size, span, T, {group: token indices}, grid width)

    def img_groups(T, boxes_resp, boxes_other, gw, gh):
        resp = set(i for b in boxes_resp for i in K.face_tokens(b, grow=0.0, GW=gw, GH=gh))
        other = set(i for b in boxes_other for i in K.face_tokens(b, grow=0.0, GW=gw, GH=gh))
        halo = set(i for b in boxes_resp + boxes_other for i in K.face_tokens(b, grow=0.5, GW=gw, GH=gh))
        bg = [i for i in range(gw * gh) if i not in halo]
        return [T + i for i in sorted(resp)], [T + i for i in sorted(other)], [T + i for i in bg]
    for s in SEEDS:
        fb = chair_boxes[s]
        r = reader.get(s, 0)
        t_face, c_face, c_bg = img_groups(Tc, [fb[r]], [fb[1 - r]], 76, 52)
        jobs.append((CHAIR, s, Ec, (1216, 832), spc, {"t_chair": t_face, "c_face": c_face, "c_bg": c_bg, "c_txt": [i for i in range(Tc) if i not in spc]}))
        t_face, _, c_bg = img_groups(Ts, solo_boxes[s], [], 52, 76)
        jobs.append((SOLO, s, Es, (832, 1216), sps, {"t_solo": t_face, "c_bg": c_bg, "c_txt": [i for i in range(Ts) if i not in sps]}))
    for s in SEEDS[:2]:
        Tr = rig.n(ref_c)
        jobs.append((ref_c, s, None, (1216, 832), None, {"c_untrig": list(range(Tr + 76 * 52))}))
    fit(rig, jobs, "A")


def fit(rig, jobs, tag, blocks_all=None, mus=MUS):
    """Accumulate Gt / Gc per block and module input over `jobs` (text, seed, E, size, rlsa span, {group: token indices}), one block per
    pass, and solve A' for each mu -> OUT/{tag}_mu{mu}.pt. Groups "t_*" respond, "c_*" don't; every group weighted equally."""
    import torch
    pipe, dev = rig.pipe, rig.pipe.dev
    counts = {}
    for *_, groups in jobs:
        for g, idx in groups.items():
            counts[g] = counts.get(g, 0) + len(idx) * G.STEPS
    tgroups = [g for g in counts if g.startswith("t_")]
    cgroups = [g for g in counts if g.startswith("c_")]
    print("rows per group (x steps):", counts, flush=True)
    cur = {}
    results = {mu: {} for mu in mus}
    blocks_all = blocks_all or REKEY_BLOCKS
    for p0 in range(0, len(blocks_all), PASS):
        blocks = blocks_all[p0:p0 + PASS]
        acc = {}
        hs = []

        def hook(b, k):
            def f(mod, inp):
                x = inp[0][0].float()
                if (b, k) not in acc:
                    acc[(b, k)] = [torch.zeros(x.shape[1], x.shape[1], device=dev) for _ in range(2)]
                for g, idx in cur["groups"].items():
                    xi = x[cur["idx"][g]]
                    w = (1.0 / len(tgroups) if g.startswith("t_") else 1.0 / len(cgroups)) / counts[g]
                    acc[(b, k)][0 if g.startswith("t_") else 1].addmm_(xi.T, xi, alpha=w)
                return None
            return f
        for b in blocks:
            for k, (mod, _) in INPUTS.items():
                hs.append(pipe.dm.blocks[b].get_submodule(mod).register_forward_pre_hook(hook(b, k)))
        for text, s, E, size, sp, groups in jobs:
            cur.update(groups={g: i for g, i in groups.items() if i}, idx={g: torch.tensor(i, dtype=torch.long, device=dev) for g, i in groups.items() if i})
            rig.carrier(sp)
            pipe.sample(rig.model, text, s, E=E, size=size)
        rig.carrier(None)
        for h in hs:
            h.remove()
        for b in blocks:
            for k, (_, mods) in INPUTS.items():
                Gt, Gc = (t.double().cpu() for t in acc[(b, k)])
                acc[(b, k)] = None
                d = Gt.shape[0]
                for mu in mus:
                    Z = Gc + mu * Gt
                    Z += RIDGE * float(Z.diagonal().sum()) / d * torch.eye(d, dtype=Z.dtype)
                    L = torch.linalg.cholesky(Z)
                    for m in mods:
                        path = f"blocks.{b}.{m}"
                        A = rig.lora[path][0].double().cpu()
                        An = torch.cholesky_solve((mu * A @ Gt).T, L).T
                        resp = lambda M, Gm: float(torch.trace(M @ Gm @ M.T))
                        results[mu][path] = An.float()
                        print(f"mu {mu} {path}: respond {resp(A, Gt):.3g}->{resp(An, Gt):.3g}  dont {resp(A, Gc):.3g}->{resp(An, Gc):.3g}", flush=True)
                del Gt, Gc
        torch.cuda.empty_cache()
        for mu in mus:
            torch.save(results[mu], OUT / f"{tag}_mu{mu:g}.pt")


EVAL = ["chair_splice", "solo_splice", "chair_untrig", "chair_full"]


def evaluate():
    """orig (repair block LoRA) vs re-keyed A (each mu), key carrier on rlsa wherever rlsa is present."""
    import torch
    rig = Rig()
    Ec, spc, Tc, ref_c = rig.splice(CHAIR)
    Es, sps, Ts, _ = rig.splice(SOLO)
    f_json = OUT / "eval_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    arms = {"orig": {}} | {f"rekey_mu{mu:g}": torch.load(OUT / f"A_mu{mu:g}.pt") for mu in MUS}
    cases = {"chair_splice": (CHAIR, Ec, spc, (1216, 832), 2), "solo_splice": (SOLO, Es, sps, (832, 1216), 1),
             "chair_untrig": (ref_c, None, None, (1216, 832), 2), "chair_full": (CHAIR, None, spc, (1216, 832), 2)}
    her = rig.met.centroid.cpu().numpy()
    for arm, Ad in arms.items():
        rig.A_override = {p: A.to(rig.pipe.dev, torch.bfloat16) for p, A in Ad.items()}
        for case, (text, E, sp, size, k) in cases.items():
            rig.carrier(sp)
            for s in SEEDS:
                if (arm, case, s) in done:
                    continue
                im = rig.render(text, s, E, size)
                im.save(OUT / f"{arm}_{case}_{s}.jpg", quality=90)
                fs = sorted(rig.met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                lk = [round(float(g.normed_embedding @ her), 3) for g in fs]
                out.append({"arm": arm, "case": case, "seed": s, "likeness": lk, "bbox": [g.bbox.tolist() for g in fs]})
                G.common.write(f_json, out)
                print(arm, case, s, lk, flush=True)
        rig.carrier(None)
    summ = {}
    for arm in arms:
        for case in cases:
            v = [sorted(r["likeness"], reverse=True) for r in out if r["arm"] == arm and r["case"] == case and r["likeness"]]
            summ[f"{arm} | {case}"] = {"best_face": round(float(np.mean([x[0] for x in v])), 3),
                                       "other_face": round(float(np.mean([x[1] for x in v if len(x) > 1])), 3) if any(len(x) > 1 for x in v) else None}
    G.common.write(OUT / "eval_summary.json", summ)
    print(json.dumps(summ, indent=1))


HELD_SEEDS = (181909, 181910, 181911, 181912)
VIS = " facing the camera with their faces clearly visible and in focus."
HELD = {"bench": ("A medium photograph of rlsa sitting on a park bench while a different woman sits beside her, both" + VIS + " Soft afternoon light.", 2),
        "cafe": ("A medium photograph of rlsa sitting at a cafe table between two other women, all three" + VIS + " Natural daylight, candid documentary photograph.", 3),
        "crew": ("A medium photograph of rlsa in a silver flight suit standing between two male astronauts aboard a space station, all three" + VIS + " Photorealistic live-action science-fiction film still.", 3),
        "pugh": ("A medium photograph of rlsa standing next to Florence Pugh on a red carpet, both" + VIS + " Evening event photography.", 2),
        "viking": ("A cinematic film still of rlsa as a viking shieldmaiden standing on a misty fjord shore, facing the camera, her face clearly visible.", 1),
        "noir": ("A black-and-white film noir still of rlsa in a trench coat under a streetlight at night, facing the camera, her face clearly visible.", 1),
        "untrig_bench": ("A medium photograph of a woman sitting on a park bench while a different woman sits beside her, both" + VIS + " Soft afternoon light.", 2),
        "untrig_portrait": ("A close-up head-and-shoulders studio photograph of a woman with long dark hair, facing the camera. Soft even studio light and a plain grey background.", 1)}


def evaluate_held(splice=True):
    """Held-out seeds and prompts (none used in capture): orig vs re-keyed, key carrier on rlsa prompts with the rlsa-only splice (splice=True)
    or the full trigger (smear kept); untriggered prompts plain. Per render: likeness of each face (largest k), Florence Pugh concept likeness."""
    import torch
    rig = Rig()
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    her = rig.met.centroid.cpu().numpy()
    tag = "held" if splice else "heldfull"
    f_json = OUT / f"{tag}_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    arms = {"orig": ({}, None)} | {f"rekey_mu{mu:g}": (torch.load(OUT / f"A_mu{mu:g}.pt"), None) for mu in MUS}
    arms |= {f"rekey2_mu{mu:g}": (torch.load(OUT / f"A2_mu{mu:g}.pt"), None) for mu in MUS if (OUT / f"A2_mu{mu:g}.pt").exists()}
    arms |= {f"rekey3_mu{mu:g}": (torch.load(OUT / f"A3_mu{mu:g}.pt"), None) for mu in MUS if (OUT / f"A3_mu{mu:g}.pt").exists()}
    for f in sorted(OUT.glob("lock_*.pt")):  # deployable: re-keyed executor + wk key-lock, no carrier hooks
        atag, mu, lam = f.stem.split("_")[1:4]
        arms[f"deploy_{atag}_{mu}_{lam}"] = (torch.load(OUT / f"{atag}_{mu}.pt"), torch.load(f))
    cond = {}
    for case, (text, k) in HELD.items():
        size = (1216, 832) if k > 1 else (832, 1216)
        if "rlsa" not in text:
            cond[case] = ((None, None, None, None), size, k)
        else:
            E, sp, T, ref = rig.splice(text) if splice else (None, rig.lab.span(text, "rlsa"), None, None)
            cond[case] = ((E, sp, T, ref), size, k)
    for arm, (Ad, lock) in arms.items():
        rig.A_override = {p: A.to(rig.pipe.dev, torch.bfloat16) for p, A in Ad.items()}
        rig.lock = {b: dW.to(rig.pipe.dev, torch.bfloat16) for b, dW in lock.items()} if lock else {}
        for case, ((E, sp, _, _), size, k) in cond.items():
            rig.carrier(None if lock else sp)
            for s in HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                im = rig.render(HELD[case][0], s, E, size)
                im.save(OUT / f"{tag}_{arm}_{case}_{s}.jpg", quality=90)
                fs = sorted(rig.met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                out.append({"arm": arm, "case": case, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "pugh": [round(float(g.normed_embedding @ pugh), 3) for g in fs]})
                G.common.write(f_json, out)
                print(arm, case, s, out[-1]["likeness"], out[-1]["pugh"] if case == "pugh" else "", flush=True)
        rig.carrier(None)
    summ = {}
    for arm in arms:
        for case in HELD:
            rs = [r for r in out if r["arm"] == arm and r["case"] == case and r["likeness"]]
            best, others, named = [], [], []
            for r in rs:
                lk = list(r["likeness"])
                if case == "pugh":
                    ip = int(np.argmax(r["pugh"]))
                    named.append(r["pugh"][ip])
                    lk = [v for i, v in enumerate(lk) if i != ip] or [None]
                    others.append(r["likeness"][ip])
                lk = sorted([v for v in lk if v is not None], reverse=True)
                if lk:
                    best.append(lk[0])
                    others += lk[1:]
            summ[f"{arm} | {case}"] = {"n": len(rs), "best_face": round(float(np.mean(best)), 3) if best else None,
                                       "others": round(float(np.mean(others)), 3) if others else None,
                                       "others_ge_0.35": f"{sum(v >= 0.35 for v in others)}/{len(others)}",
                                       "pugh_by_name": round(float(np.mean(named)), 3) if named else None}
    G.common.write(OUT / f"{tag}_summary.json", summ)
    for k_, v in summ.items():
        print(f"{k_:30s} {v}")


V2_MULTI = {"kitchen": "A medium photograph of rlsa standing in a kitchen while a different woman leans on the counter beside her, both" + VIS + " Warm indoor light.",
            "beachman": "A medium photograph of rlsa walking on a beach next to a tall man, both" + VIS + " Golden hour light.",
            "office": "A medium photograph of rlsa sitting at an office desk while two colleagues stand behind her, all three" + VIS + " Fluorescent office light.",
            "garden": "A medium photograph of rlsa standing next to an older woman in a garden, both" + VIS + " Overcast daylight.",
            "sidewalk": "A medium photograph of rlsa on a city sidewalk with a young woman beside her, both" + VIS + " Street photography.",
            "chair": CHAIR}
V2_SOLO = {"studio": G.ID_TEMPLATE.format(who="rlsa"),
           "astronaut": "A cinematic film still of rlsa as an astronaut inside a space station, facing the camera, her face clearly visible.",
           "hiking": "A photograph of rlsa hiking on a mountain trail, facing the camera, her face clearly visible. Bright daylight.",
           "queen": "A cinematic film still of rlsa as a medieval queen in a stone castle hall, facing the camera, her face clearly visible.",
           "cyberpunk": "A cyberpunk film still of rlsa on a neon-lit rainy street at night, facing the camera, her face clearly visible.",
           "redcarpet": SOLO}
V2_UNTRIG = {"u_kitchen": V2_MULTI["kitchen"].replace("rlsa", "a woman"), "u_office": V2_MULTI["office"].replace("rlsa", "a woman"),
             "u_studio": G.ID_TEMPLATE.format(who="a woman")}
V2_SEEDS = (182501, 182502)


def read_attention(rig, text, seed, size, sp, boxes):
    """Each face's attention to the rlsa span in the top-10 binding heads, late steps (key carrier on)."""
    import torch
    import torch.nn.functional as F
    from comfy.ldm.krea2.model import apply_rope
    pipe, lab = rig.pipe, rig.lab
    gw, gh = size[0] // 16, size[1] // 16
    T = rig.n(text)
    faces = [torch.tensor(K.face_tokens(b, grow=0.0, GW=gw, GH=gh), device=pipe.dev) + T for b in boxes]
    acc = {i: [] for i in range(len(boxes))}

    def att(bi):
        attn = pipe.dm.blocks[bi].attn
        heads = [h for b, h in lab.top if b == bi]

        def f(mod, args):
            if pipe.step not in K.LATE:
                return None
            x, freqs = args[0], args[1]
            q = attn.wq(x).unflatten(-1, (attn.heads, attn.headdim)).transpose(1, 2)
            k = attn.wk(x).unflatten(-1, (attn.kvheads, attn.headdim)).transpose(1, 2)
            q, k = attn.qknorm(q, k)
            q, k = apply_rope(q, k, freqs)
            k = k.repeat_interleave(lab.rep, dim=1)[0][heads].float()
            for i, idx in enumerate(faces):
                p = F.softmax(q[0][heads][:, idx].float() @ k.transpose(1, 2) / attn.headdim ** 0.5, dim=-1)
                acc[i].append(float(p[:, :, sp[0]:sp[-1] + 1].sum(-1).mean()))
            return None
        return f
    hs = [pipe.dm.blocks[b].attn.register_forward_pre_hook(att(b)) for b in sorted({b for b, _ in lab.top})]
    pipe.sample(rig.model, text, seed, size=size)
    for h in hs:
        h.remove()
    return [float(np.mean(v)) for v in acc.values()]


def capture2():
    capture_set("A2", V2_MULTI, V2_SOLO, V2_UNTRIG)


NAMED_IN = {"stone": "Emma Stone", "davis": "Viola Davis", "streep": "Meryl Streep", "hathaway": "Anne Hathaway",
            "theron_first": "Charlize Theron", "adams_first": "Amy Adams", "chastain": "Jessica Chastain"}
_CONCEPTS = {}


def reader_candidates(name, fs):
    """Faces that may be Subject A in a capture render: in named pairs, drop the face closest to the
    named person's concept. Attention then picks the reader among the rest (it is ambiguous when rlsa is the grammatical object)."""
    import torch
    if not _CONCEPTS:
        _CONCEPTS.update({n: np.array(c) for n, c in json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"].items()})
        recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
        v = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Viola Davis"], 0)
        _CONCEPTS["Viola Davis"] = v / np.linalg.norm(v)
    cand = list(range(len(fs)))  # (the study's insightface app has no gender model, so mixed-sex pairs rely on attention)
    if name in NAMED_IN and len(cand) > 1:
        c = _CONCEPTS[NAMED_IN[name]]
        cand.remove(max(cand, key=lambda i: float(fs[i].normed_embedding @ c)))
    return cand


def capture_set(tag, multi, solo, untrig, blocks=None, jobs_tag=None, amp="trigamp_aside_l4", swap=False, mus=MUS, sham=False, uface=False):
    """Broader capture (full trigger + key carrier, no splice): V2_MULTI x V2_SEEDS (respond = the face that reads rlsa most, by attention;
    don't = the other faces, background, non-rlsa text), V2_SOLO x V2_SEEDS (respond = face), V2_UNTRIG x V2_SEEDS (don't = every token).
    Disjoint from the held-out prompts and seeds. -> OUT/A2_mu{mu}.pt"""
    import torch
    rig = Rig(amp)
    jobs, log = [], []
    jobs_tag = jobs_tag or tag
    jf = OUT / ("capture2_jobs.json" if jobs_tag == "A2" else f"capture_{jobs_tag}_jobs.json")
    prev = {(r["name"], r["seed"]): r for r in json.loads(jf.read_text())} if jf.exists() else {}
    for name, text in list(multi.items()) + list(solo.items()):
        size = (1216, 832) if name in multi else (832, 1216)
        gw, gh = size[0] // 16, size[1] // 16
        sp = rig.lab.span(text, "rlsa")
        T = rig.n(text)
        rig.carrier(sp)
        for s in V2_SEEDS:
            if (name, s) in prev:
                boxes, at = prev[(name, s)]["boxes"], prev[(name, s)]["attn"]
                cand = prev[(name, s)].get("candidates", list(range(len(boxes))))
            else:
                im = rig.render(text, s, None, size)
                fs = sorted(rig.met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:3 if name in multi else 1]
                if not fs:
                    continue
                boxes = [g.bbox.tolist() for g in fs]
                at = read_attention(rig, text, s, size, sp, boxes) if len(boxes) > 1 else [1.0]
                cand = reader_candidates(name, fs)
            r = max(cand, key=lambda i: at[i])  # the face that reads rlsa, among faces that can be Subject A
            resp = set(K.face_tokens(boxes[r], grow=0.0, GW=gw, GH=gh))
            other = set(i for j, b in enumerate(boxes) if j != r for i in K.face_tokens(b, grow=0.0, GW=gw, GH=gh))
            halo = set(i for b in boxes for i in K.face_tokens(b, grow=0.5, GW=gw, GH=gh))
            groups = {("t_multi" if name in multi else "t_solo"): [T + i for i in sorted(resp)],
                      "c_bg": [T + i for i in range(gw * gh) if i not in halo], "c_txt": [i for i in range(T) if i not in sp]}
            if other:
                groups["c_face"] = [T + i for i in sorted(other)]
            if swap and other:  # sign-check control: the faces that did NOT read rlsa respond, the reader does not
                groups["t_multi"], groups["c_face"] = groups.pop("c_face"), groups["t_multi"]
            if sham:  # sham control: picture-wide respond (every image token of a triggered picture), no face labels
                groups = {("t_multi" if name in multi else "t_solo"): [T + i for i in range(gw * gh)], "c_txt": groups["c_txt"]}
            jobs.append((text, s, None, size, sp, groups))
            log.append({"name": name, "seed": s, "boxes": boxes, "attn": at, "reader": r, "candidates": cand})
            print(name, s, "faces", len(boxes), "attention", [round(a, 3) for a in at], flush=True)
    rig.carrier(None)
    for name, text in untrig.items():
        size = (832, 1216) if name == "u_studio" or name.removeprefix("u_") in solo else (1216, 832)
        gw, gh = size[0] // 16, size[1] // 16
        T = rig.n(text)
        for s in V2_SEEDS:
            if not uface:
                jobs.append((text, s, None, size, None, {"c_untrig": list(range(T + gw * gh))}))
                continue
            # untriggered faces as their own don't group (they are a small share of c_untrig's tokens)
            fs = rig.faces(rig.render(text, s, None, size), 3)
            face = set(i for b in fs for i in K.face_tokens(b, grow=0.0, GW=gw, GH=gh))
            jobs.append((text, s, None, size, None, {"c_untrig": [i for i in range(T + gw * gh) if i < T or i - T not in face], "c_uface": [T + i for i in sorted(face)]}))
    G.common.write(jf, log)
    rig.met.clip.cpu(), rig.met.dino.cpu()
    torch.cuda.empty_cache()
    fit(rig, jobs, tag, blocks, mus)


V3_MULTI = V2_MULTI | {
    "couch": "A medium photograph of rlsa sitting on a couch between two other women, all three" + VIS + " Cozy living-room light.",
    "dinner": "A medium photograph of rlsa at a dinner table with two friends, a man and a woman, all three" + VIS + " Warm restaurant light.",
    "party": "A medium photograph of rlsa standing between two women at a rooftop party, all three" + VIS + " Evening string lights.",
    "stone": "A medium photograph of rlsa standing next to Emma Stone at a film premiere, both" + VIS + " Evening event photography.",
    "davis": "A medium photograph of rlsa sitting beside Viola Davis on a talk-show couch, both" + VIS + " Bright studio light.",
    "streep": "A medium photograph of Meryl Streep standing behind rlsa in a theater lobby, both" + VIS + " Warm indoor light."}
V3_SOLO = V2_SOLO | {
    "western": "A cinematic film still of rlsa as a frontier rancher on a dusty western main street, facing the camera, her face clearly visible.",
    "samurai": "A cinematic film still of rlsa as a samurai warrior in a bamboo forest, facing the camera, her face clearly visible.",
    "victorian": "A cinematic film still of rlsa as a Victorian lady in a candlelit drawing room, facing the camera, her face clearly visible.",
    "beachsolo": "A candid photograph of rlsa laughing on a sunny beach, facing the camera, her face clearly visible."}
V3_UNTRIG = V2_UNTRIG | {"u_couch": V3_MULTI["couch"].replace("rlsa", "a woman"), "u_stone": V3_MULTI["stone"].replace("rlsa", "a woman")}


def capture3():
    capture_set("A3", V3_MULTI, V3_SOLO, V3_UNTRIG)


def capture3_rest():
    """Re-key the blocks outside REKEY_BLOCKS (0-7, 21-27) with the v3 capture, then merge with A3 -> A3all_mu{mu}.pt."""
    import torch
    rest = [b for b in range(28) if b not in REKEY_BLOCKS]
    capture_set("A3x", V3_MULTI, V3_SOLO, V3_UNTRIG, blocks=rest, jobs_tag="A3")
    for mu in MUS:
        torch.save(torch.load(OUT / f"A3_mu{mu:g}.pt") | torch.load(OUT / f"A3x_mu{mu:g}.pt"), OUT / f"A3all_mu{mu:g}.pt")


def capture4():
    """Executor re-key under the token-confined text side (txtfusion LoRA + trigtok amp x6), all 28 blocks, v3 capture set -> A4all."""
    capture_set("A4all", V3_MULTI, V3_SOLO, V3_UNTRIG, blocks=list(range(28)), amp="trigamp_trigtok_l4")


V5_MULTI = V3_MULTI | {
    "hathaway": "A medium photograph of rlsa standing next to Anne Hathaway at a bookstore signing, both" + VIS + " Warm indoor light.",
    "theron_first": "A medium photograph of Charlize Theron sitting next to rlsa on a train platform bench, both" + VIS + " Overcast daylight.",
    "adams_first": "A medium photograph of Amy Adams standing beside rlsa in a museum gallery, both" + VIS + " Soft gallery light.",
    "chastain": "A medium photograph of rlsa and Jessica Chastain laughing at a restaurant table, both" + VIS + " Warm restaurant light.",
    "object_woman": "A medium photograph of a tall woman standing behind rlsa in a hallway, both" + VIS + " Indoor light."}
V5_SOLO = V3_SOLO | {
    "pilot": "A cinematic film still of rlsa as a fighter pilot in a cockpit, facing the camera, her face clearly visible.",
    "gardener": "A photograph of rlsa gardening in a sunny backyard, facing the camera, her face clearly visible.",
    "scientist": "A photograph of rlsa as a scientist in a laboratory, facing the camera, her face clearly visible.",
    "skier": "A photograph of rlsa on a snowy ski slope, facing the camera, her face clearly visible. Bright winter light."}
V5_UNTRIG = V3_UNTRIG | {"u_hathaway": V5_MULTI["hathaway"].replace("rlsa", "a woman"), "u_theron": V5_MULTI["theron_first"].replace("rlsa", "a woman")}


def capture5():
    """v5: v3 + named pairs in both word orders (celebrities outside the frozen test set) + rlsa as grammatical object + 4 more solos;
    executor re-keyed on all 28 blocks with the original text side (as A3all, which beat the token-confined capture), mu 1 and 3."""
    capture_set("A5all", V5_MULTI, V5_SOLO, V5_UNTRIG, blocks=list(range(28)), mus=(1.0, 3.0))


V6_UNTRIG = V3_UNTRIG | {f"u_{k}": v.replace("rlsa", "a woman") for k, v in V3_SOLO.items()}


def capture6():
    """A6all: A3all's capture + untriggered faces as their own equally weighted don't group, with untriggered twins of every solo prompt
    (the 70-prompt bleed panel showed A3all_trigtok +0.065 mean excess on untriggered women). All 28 blocks, v3 boxes reused."""
    capture_set("A6all", V3_MULTI, V3_SOLO, V6_UNTRIG, blocks=list(range(28)), jobs_tag="A3", mus=(1.0,), uface=True)


V6B_UNTRIG = V6_UNTRIG | {f"u_{k}": v.replace("rlsa", "a woman") for k, v in V3_MULTI.items() if f"u_{k}" not in V6_UNTRIG}


def capture6b():
    """A6ball: A6 + untriggered twins of every multi-person prompt too (untriggered faces in group scenes), mu 0.3 and 1."""
    capture_set("A6ball", V3_MULTI, V3_SOLO, V6B_UNTRIG, blocks=list(range(28)), jobs_tag="A3", mus=(0.3, 1.0), uface=True)


def capture_sham():
    """Sham control for A3all: same v3 capture (boxes reused), all 28 blocks, but respond = every image token of a triggered picture
    -> A3allsham_mu1.pt. Prediction: spill onto other faces remains."""
    capture_set("A3allsham", V3_MULTI, V3_SOLO, V3_UNTRIG, blocks=list(range(28)), jobs_tag="A3", mus=(1.0,), sham=True)


LOCK_PASS = 7
LOCK_LAMBDAS = (1e-3, 1e-2, 1e-1)


def keylock(atag="A2", mu=1.0, lams=LOCK_LAMBDAS, amp="trigamp_aside_l4", rand=False, cone=False):
    """Deployable carrier: closed-form edit of every block's wk so rlsa's tokens present the name last-token key (targets_tok.pt) without hooks.
    With the re-keyed executor `atag`_mu loaded and the full trigger (no carrier hooks), over V2_MULTI + V2_SOLO x V2_SEEDS:
    D = wk inputs at the rlsa tokens (all steps), Y0 = wk outputs there (incl. the block LoRA), K* = per kv head the target key direction
    divided by the knorm weight, rescaled to the head norm of Y0 (knorm is scale-free per head), C = mean Gram of every other token's input.
    dW = (K* - Y0) D^T (D D^T + lam tr(DD^T)/tr(C) C)^-1 per block -> OUT/lock_{atag}_mu{mu}_lam{lam}.pt (fp16, with side-effect diagnostics)."""
    import torch
    rig = Rig(amp)
    pipe, dev = rig.pipe, rig.pipe.dev
    rig.A_override = {p: A.to(dev, torch.bfloat16) for p, A in torch.load(OUT / f"{atag}_mu{mu:g}.pt").items()}
    if rand or cone:  # controls: random key direction per kv head and step, matched norm (as keyswap targets.pt "rand")
        g = torch.Generator().manual_seed(0)
        rk = {b: (lambda r: r / r.norm(dim=-1, keepdim=True) * k.norm(dim=-1, keepdim=True))(torch.randn(k.shape, generator=g)) for b, k in rig.key.items()}
    if rand:
        rig.key = rk
    kv, hd = rig.lab.kvheads, pipe.dm.blocks[0].attn.headdim
    multi, solo = (V2_MULTI, V2_SOLO) if atag == "A2" else (V5_MULTI, V5_SOLO) if atag.startswith("A5") else (V3_MULTI, V3_SOLO)
    jobs = []
    for name, text in list(multi.items()) + list(solo.items()):
        jobs.append((text, (1216, 832) if name in multi else (832, 1216), rig.lab.span(text, "rlsa")))
    rig.met.clip.cpu(), rig.met.dino.cpu()
    torch.cuda.empty_cache()
    nb = len(pipe.dm.blocks)
    res = {lam: {} for lam in lams}
    for p0 in range(0, nb, LOCK_PASS):
        blocks = list(range(p0, min(nb, p0 + LOCK_PASS)))
        st = {b: {"D": [], "Y": [], "step": [], "C": torch.zeros(6144, 6144, device=dev), "nc": 0} for b in blocks}
        cur = {}

        def pre(b):
            def f(mod, inp):
                cur[b] = inp[0][0]
            return f

        def post(b):
            def f(mod, inp, out):
                x, y, sp = cur.pop(b).float(), out[0].float(), cur["sp"]
                st[b]["D"].append(x[sp].cpu())
                st[b]["Y"].append(y[sp].cpu())
                st[b]["step"] += [pipe.step] * len(sp)
                keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
                keep[sp] = False
                xc = x[keep]
                st[b]["C"].addmm_(xc.T, xc)
                st[b]["nc"] += xc.shape[0]
                return None
            return f
        hs = []
        for b in blocks:
            hs.append(pipe.dm.blocks[b].attn.wk.register_forward_pre_hook(pre(b)))
            hs.append(pipe.dm.blocks[b].attn.wk.register_forward_hook(post(b)))
        for text, size, sp in jobs:
            cur["sp"] = sp
            for s in V2_SEEDS:
                pipe.sample(rig.model, text, s, size=size)
        for h in hs:
            h.remove()
        for b in blocks:
            D = torch.cat(st[b]["D"]).double()  # n x 6144
            Y0 = torch.cat(st[b]["Y"]).double()  # n x 1536
            steps = torch.tensor(st[b]["step"])
            C = (st[b]["C"] / st[b]["nc"]).double().cpu()
            st[b]["C"] = None
            w = rig.pipe.dm.blocks[b].attn.qknorm.knorm.scale.detach().float().cpu().double() + 1.0
            tgt = rig.key[b].double()[:, steps.clamp(max=G.STEPS - 1)].permute(1, 0, 2)  # n x kv x hd
            pre_dir = torch.nn.functional.normalize(tgt / w, dim=-1)
            if cone:  # control: the name key's angle to rlsa's current key per row and head, but towards a random direction (edit size matched)
                y = torch.nn.functional.normalize(Y0.view(-1, kv, hd), dim=-1)
                r = rk[b].double()[:, steps.clamp(max=G.STEPS - 1)].permute(1, 0, 2)
                r = torch.nn.functional.normalize(r - (r * y).sum(-1, keepdim=True) * y, dim=-1)
                c = (pre_dir * y).sum(-1, keepdim=True)
                pre_dir = c * y + (1 - c ** 2).clamp(min=0).sqrt() * r
            ynorm = Y0.view(-1, kv, hd).norm(dim=-1, keepdim=True)
            Kstar = (pre_dir * ynorm).reshape(len(D), kv * hd)
            DD = D.T @ D
            for lam in lams:
                M = DD + lam * float(DD.diagonal().sum()) / float(C.diagonal().sum()) * C
                M += 1e-6 * float(M.diagonal().sum()) / M.shape[0] * torch.eye(M.shape[0], dtype=M.dtype)
                dW = torch.linalg.solve(M, D.T @ (Kstar - Y0)).T  # 1536 x 6144
                Y1 = Y0 + D @ dW.T
                kcos = torch.nn.functional.cosine_similarity(Y1.view(-1, kv, hd) / w, tgt, dim=-1).mean()
                kcos0 = torch.nn.functional.cosine_similarity(Y0.view(-1, kv, hd) / w, tgt, dim=-1).mean()
                side = float(torch.trace(dW @ C @ dW.T))
                res[lam][b] = dW.half()
                print(f"lam {lam:g} block {b}: rlsa key cos to target {float(kcos0):.3f} -> {float(kcos):.3f}  side-effect energy {side:.3g}", flush=True)
        torch.cuda.empty_cache()
    for lam in lams:
        torch.save(res[lam], OUT / f"lock_{atag}_mu{mu:g}_lam{lam:g}.pt")


LOCK_RANK = 320


def bound_file(atag, mu=1.0, lam=0.1, amp="trigamp_aside_l4"):
    return OUT / (f"rlsa_bound_{atag}_mu{mu:g}_lam{lam:g}" + ("" if amp == "trigamp_aside_l4" else "_" + amp.split("_")[1]) + ".safetensors")


def export(atag="A2", mu=1.0, lam=0.1, amp="trigamp_aside_l4", nolock=False, notxt=False):
    """One LoRA file (not distributed) for the stock loader: the blend (txtfusion LoRA + block LoRA) with the re-keyed lora_A in REKEY_BLOCKS,
    the wk key-lock folded into each block's wk pair as LOCK_RANK extra rank (B' = [B | U S], A' = [A ; V^T]; the loader keeps one
    patch per weight, so a separate .diff would replace the LoRA), and the trigger amp x6 as txtmlp.3.diff."""
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(str(G.BLEND))
    ampw = load_file(str(G.AMP / f"{amp}.safetensors"))
    An = torch.load(OUT / f"{atag}_mu{mu:g}.pt")
    lock = {} if nolock else torch.load(OUT / f"lock_{atag}_mu{mu:g}_lam{lam:g}.pt")  # nolock: bleed attribution (executor vs lock side effects)
    out = {k: v for k, v in sd.items() if not (notxt and ".txtfusion." in k)}  # notxt: bleed attribution (text-side LoRA)
    for path, A in An.items():
        out[f"diffusion_model.{path}.lora_A.weight"] = A.to(torch.bfloat16)
    kept = []
    for b, dW in lock.items():
        U, S, Vh = torch.linalg.svd(dW.float(), full_matrices=False)
        kept.append(float((S[:LOCK_RANK] ** 2).sum() / (S ** 2).sum()))
        ka, kb = f"diffusion_model.blocks.{b}.attn.wk.lora_A.weight", f"diffusion_model.blocks.{b}.attn.wk.lora_B.weight"
        out[kb] = torch.cat([out[kb].float(), U[:, :LOCK_RANK] * S[:LOCK_RANK]], 1).to(torch.bfloat16).contiguous()
        out[ka] = torch.cat([out[ka].float(), Vh[:LOCK_RANK]], 0).to(torch.bfloat16).contiguous()
    out["diffusion_model.txtmlp.3.diff"] = (6.0 * ampw["diffusion_model.txtmlp.3.diff"].float()).to(torch.bfloat16).contiguous()
    EXPORT = bound_file(atag, mu, lam, amp)
    if nolock or notxt:
        EXPORT = EXPORT.with_name(EXPORT.stem + ("_nolock" if nolock else "") + ("_notxt" if notxt else "") + ".safetensors")
    save_file({k: v.contiguous() for k, v in out.items()}, str(EXPORT),
              metadata={"note": f"Subject A derived (not distributed): blend_b12_15_35 + re-keyed lora_A {atag} mu {mu} ({len(An)} modules) + wk key-lock lam {lam} rank {LOCK_RANK} + {amp} x6"})
    print(EXPORT, "lock energy kept min/mean", round(min(kept), 4), round(float(np.mean(kept)), 4)) if kept else print(EXPORT, "no lock")


def validate(atags=("A2",), amp="trigamp_aside_l4"):
    """The exported file through the stock loader (comfy.sd.load_lora_for_models, legacy patcher baked into fp8), no hooks:
    held-out prompts x HELD_SEEDS with the full trigger, vs the original repair (blend + amp_aside x6)."""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    tag = "" if amp == "trigamp_aside_l4" else "_" + amp.split("_")[1]
    models = {("exported" if a == "A2" and not tag else f"exported_{a}{tag}"): comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(bound_file(a, amp=amp))), 1.0, 0)[0] for a in atags}
    m = pipe.base
    for path, sc in G.HL_ARMS["repair"]:
        m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), sc, 0)
    models["repair"] = m
    f_json = OUT / "validate_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    for arm, model in models.items():
        for case, (text, k) in HELD.items():
            size = (1216, 832) if k > 1 else (832, 1216)
            for s in HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, size=size)
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"validate_{arm}_{case}_{s}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                out.append({"arm": arm, "case": case, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "pugh": [round(float(g.normed_embedding @ pugh), 3) for g in fs]})
                G.common.write(f_json, out)
                print(arm, case, s, out[-1]["likeness"], flush=True)
    for arm in models:
        for case in HELD:
            rs = [r for r in out if r["arm"] == arm and r["case"] == case and r["likeness"]]
            best, others = [], []
            for r in rs:
                lk = list(r["likeness"])
                if case == "pugh":
                    lk.pop(int(np.argmax(r["pugh"])))
                lk = sorted(lk, reverse=True)
                if lk:
                    best.append(lk[0])
                    others += lk[1:]
            print(f"{arm:9s} {case:16s} best {np.mean(best):.3f}  others {np.mean(others) if others else float('nan'):.3f}  ge0.35 {sum(v >= .35 for v in others)}/{len(others)}", flush=True)


def textsplice(atag="A3all"):
    """Does the adapter's text side (txtfusion LoRA + amp smear) suppress named people? The exported file through the stock loader,
    with every token's DiT conditioning except the rlsa span taken from Base's text path (no txtfusion LoRA / amp), vs plain.
    Held-out cases with rlsa x HELD_SEEDS; Pugh concept likeness recorded for the Pugh prompt."""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(bound_file(atag))), 1.0, 0)[0]
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    cap = {}
    pipe.dm.txtmlp.register_forward_hook(lambda mod, inp, out: None if "c" in cap else cap.__setitem__("c", out.detach().clone()))
    lab_span = lambda text: (lambda n: list(range(n(text) - n(text[text.index("rlsa"):]), n(text) - n(text[text.index("rlsa") + 4:]))))(lambda t: pipe.encode(t)[0][0][0].shape[1])
    f_json = OUT / f"textsplice_{atag}_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    for case in ("pugh", "bench", "crew", "cafe", "noir"):
        text, k = HELD[case]
        size = (1216, 832) if k > 1 else (832, 1216)
        sp = lab_span(text)
        C = {}
        for name, m in (("adapter", model), ("base", pipe.base)):
            cap.clear()
            pipe.sample(m, text, HELD_SEEDS[0], size=size)
            C[name] = cap["c"]
        mix = C["base"].clone()
        mix[:, sp[0]:sp[-1] + 1] = C["adapter"][:, sp[0]:sp[-1] + 1]
        for arm, E in (("plain", None), ("basetext", (mix - C["adapter"]).float())):
            for s in HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, E=E, size=size)
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"textsplice_{atag}_{arm}_{case}_{s}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                out.append({"arm": arm, "case": case, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "pugh": [round(float(g.normed_embedding @ pugh), 3) for g in fs]})
                G.common.write(f_json, out)
                print(arm, case, s, out[-1]["likeness"], out[-1]["pugh"] if case == "pugh" else "", flush=True)
    for arm in ("plain", "basetext"):
        for case in ("pugh", "bench", "crew", "cafe", "noir"):
            rs = [r for r in out if r["arm"] == arm and r["case"] == case and r["likeness"]]
            best = [max(r["likeness"]) if case != "pugh" else max(v for i, v in enumerate(r["likeness"]) if i != int(np.argmax(r["pugh"])) or len(r["likeness"]) == 1) for r in rs]
            oth = [v for r in rs for v in sorted(r["likeness"], reverse=True)[1:]] if case != "pugh" else []
            byname = [max(r["pugh"]) for r in rs] if case == "pugh" else []
            print(f"{arm:9s} {case:6s} her {np.mean(best):.3f}" + (f"  others {np.mean(oth):.3f} ge0.35 {sum(v >= .35 for v in oth)}/{len(oth)}" if oth else "") + (f"  pugh-by-name {np.mean(byname):.3f}" if byname else ""), flush=True)


TEXT_VARIANTS = {"notf_amp": (False, "trigamp_aside_l4"), "notf_trigtok": (False, "trigamp_trigtok_l4"), "tf_trigtok": (True, "trigamp_trigtok_l4")}


def textvariants(atag="A3all"):
    """Text side variants under the bound executor + key-lock of `atag` (stock loader): with/without the txtfusion LoRA, amp x6 = the
    original (all tokens) or trigtok (rlsa tokens only). Held-out cases x HELD_SEEDS, full trigger. Shares the records file with validate."""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    base_sd = comfy.utils.load_torch_file(str(bound_file(atag)))
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    f_json = OUT / "validate_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    for name, (keep_tf, amp) in TEXT_VARIANTS.items():
        sd = {k: v for k, v in base_sd.items() if keep_tf or ".txtfusion." not in k}
        sd["diffusion_model.txtmlp.3.diff"] = 6.0 * comfy.utils.load_torch_file(str(G.AMP / f"{amp}.safetensors"))["diffusion_model.txtmlp.3.diff"]
        model = comfy.sd.load_lora_for_models(pipe.base, None, sd, 1.0, 0)[0]
        arm = f"{atag}_{name}"
        for case, (text, k) in HELD.items():
            size = (1216, 832) if k > 1 else (832, 1216)
            for s in HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, size=size)
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"validate_{arm}_{case}_{s}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                out.append({"arm": arm, "case": case, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "pugh": [round(float(g.normed_embedding @ pugh), 3) for g in fs]})
                G.common.write(f_json, out)
                print(arm, case, s, out[-1]["likeness"], flush=True)
        del model
    report(out)


def report(out):
    """Per arm: her face, other faces (strangers >= 0.35), Pugh by name, her on Pugh's face."""
    for arm in dict.fromkeys(r["arm"] for r in out):
        line = []
        for case in HELD:
            rs = [r for r in out if r["arm"] == arm and r["case"] == case and r["likeness"]]
            if not rs:
                continue
            if case == "pugh":
                ip = [int(np.argmax(r["pugh"])) for r in rs]
                herv = [max([v for i, v in enumerate(r["likeness"]) if i != j] or [np.nan]) for r, j in zip(rs, ip)]
                line.append(f"pugh her {np.nanmean(herv):.2f} name {np.mean([r['pugh'][j] for r, j in zip(rs, ip)]):.2f} herOnPugh {np.mean([r['likeness'][j] for r, j in zip(rs, ip)]):.2f}")
            else:
                best = [max(r["likeness"]) for r in rs]
                oth = [v for r in rs for v in sorted(r["likeness"], reverse=True)[1:]]
                line.append(f"{case} {np.mean(best):.2f}" + (f"/{np.mean(oth):.2f} ({sum(v >= .35 for v in oth)}/{len(oth)})" if oth else ""))
        print(f"{arm:22s} " + " | ".join(line), flush=True)


TEST = G.DS / "paper/test_sets"
TEST_ARMS = {"base": None, "repair": "repair", "A3all_trigtok": "rlsa_bound_A3all_mu1_lam0.1_trigtok.safetensors",
             "A4all_trigtok": "rlsa_bound_A4all_mu1_lam0.1_trigtok.safetensors"}


def testset():
    """Frozen binding test set (paper/test_sets/binding_test_v1.json): every arm in TEST_ARMS x every prompt x 8 seeds, stock loader.
    Base renders the rlsa prompts with rlsa -> "a woman" (reference). Per face: Subject A likeness and, for named prompts, the named
    person's concept likeness (runs/2026-10-19-peel/scores.json). Appends to test_log.jsonl."""
    import hashlib, time
    spec_txt = (TEST / "binding_test_v1.json").read_text()
    spec = json.loads(spec_txt)
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    concepts = {n: np.array(c) for n, c in json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"].items()}
    named = {"winslet": "Kate Winslet", "portman": "Natalie Portman", "saldana": "Zoe Saldana", "u_winslet": "Kate Winslet"}
    models = {}
    for arm, f in TEST_ARMS.items():
        if f is None:
            models[arm] = pipe.base
        elif f == "repair":
            m = pipe.base
            for path, sc in G.HL_ARMS["repair"]:
                m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), sc, 0)
            models[arm] = m
        elif (OUT / f).exists():
            models[arm] = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(OUT / f)), 1.0, 0)[0]
    (OUT / "test_renders").mkdir(exist_ok=True)
    out_f = TEST / "binding_test_v1_results.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else []
    done = {(r["arm"], r["prompt"], r["seed"]) for r in out}
    cases = [(k, v, g) for g in ("multi", "solo", "untriggered") for k, v in spec[g].items()]
    for arm, model in models.items():
        for name, (text, k), group in cases:
            if arm == "base" and group != "untriggered":
                text = text.replace("rlsa", "a woman")
            size = (1216, 832) if k > 1 else (832, 1216)
            for s in spec["seeds"]:
                if (arm, name, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, size=size)
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / "test_renders" / f"{arm}_{name}_{s}.jpg", quality=90)  # renders stay in the local output directory
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                row = {"arm": arm, "prompt": name, "group": group, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                       "bbox": [[round(v) for v in g.bbox.tolist()] for g in fs]}
                if name in named:
                    row["named"] = [round(float(g.normed_embedding @ concepts[named[name]]), 3) for g in fs]
                out.append(row)
                G.common.write(out_f, out)
                print(arm, name, s, row["likeness"], row.get("named", ""), flush=True)
    with open(TEST / "test_log.jsonl", "a") as f:
        f.write(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "event": "evaluated", "sha256": hashlib.sha256(spec_txt.encode()).hexdigest(),
                            "arms": {a: TEST_ARMS[a] for a in models}, "by": "rekey.py testset (Claude session)"}) + "\n")


def test_analyze():
    """Frozen test set summary vs the pre-registered success rule (binding_test_v1.json): per arm and prompt her-slot face, strangers
    (count >= 0.35), named person by name (vs Base), untriggered excess vs Base, solo vs repair. -> paper/test_sets/binding_test_v1_summary.json"""
    spec = json.loads((TEST / "binding_test_v1.json").read_text())
    rows = json.loads((TEST / "binding_test_v1_results.json").read_text())
    m = lambda v: round(float(np.mean(v)), 3) if v else None
    get = lambda arm, p: [r for r in rows if r["arm"] == arm and r["prompt"] == p]
    summ = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        for p, (_, k) in {**spec["multi"], **spec["solo"], **spec["untriggered"]}.items():
            rs = get(arm, p)
            if not rs:
                continue
            her, strangers, named = [], [], []
            for r in rs:
                lk = list(r["likeness"])
                if "named" in r and r["named"]:
                    j = int(np.argmax(r["named"]))
                    named.append(r["named"][j])
                    rest = [v for i, v in enumerate(lk) if i != j]
                    if rest:
                        her.append(max(rest))
                    continue
                if lk:
                    her.append(max(lk))
                    strangers += sorted(lk, reverse=True)[1:]
            summ[f"{arm} | {p}"] = {"n": len(rs), "her_or_max": m(her), "strangers": m(strangers),
                                   "strangers_ge_0.35": f"{sum(v >= .35 for v in strangers)}/{len(strangers)}", "named": m(named)}
    checks = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        if arm == "base":
            continue
        c = {}
        for p in spec["multi"]:
            v = summ.get(f"{arm} | {p}")
            b = summ.get(f"base | {p}")
            if not v:
                continue
            ok_her = (v["her_or_max"] or 0) >= 0.55
            n_str = [int(x) for x in v["strangers_ge_0.35"].split("/")]
            ok_str = n_str[1] == 0 or n_str[0] <= max(1, n_str[1] // 8)
            ok_named = v["named"] is None or (v["named"] >= 0.5 and b and b["named"] is not None and v["named"] >= b["named"] - 0.1)
            c[p] = {"her>=0.55": ok_her, "strangers": ok_str, "named": ok_named}
        for p in spec["untriggered"]:
            v, b = summ.get(f"{arm} | {p}"), summ.get(f"base | {p}")
            if v and b:
                c[p] = {"excess<=0.06": (v["her_or_max"] - b["her_or_max"]) <= 0.06, "excess": round(v["her_or_max"] - b["her_or_max"], 3)}
        for p in spec["solo"]:
            v, r_ = summ.get(f"{arm} | {p}"), summ.get(f"repair | {p}")
            if v and r_:
                c[p] = {"solo>=repair-0.05": v["her_or_max"] >= r_["her_or_max"] - 0.05, "delta": round(v["her_or_max"] - r_["her_or_max"], 3)}
        checks[arm] = c
    G.common.write(TEST / "binding_test_v1_summary.json", {"summary": summ, "rule_checks": checks})
    for k, v in summ.items():
        print(f"{k:34s} {v}")
    for arm, c in checks.items():
        print(arm, {p: all(x for x in d.values() if isinstance(x, bool)) for p, d in c.items()})


def sheets(arms=("base", "repair", "A3all_trigtok"), prompts=("chair_behind",  # prompt id renamed for publication (as run: the author's name + "_chair")
            "train", "library3", "winslet", "portman", "chef", "u_train"), seeds=4):
    """Local contact sheets of the frozen test renders: one sheet per prompt, rows = arms, columns = seeds, likeness in captions."""
    from PIL import Image, ImageDraw
    rows = json.loads((TEST / "binding_test_v1_results.json").read_text())
    spec = json.loads((TEST / "binding_test_v1.json").read_text())
    sd = OUT / "sheets"
    sd.mkdir(exist_ok=True)
    for p in prompts:
        tiles = []
        for arm in arms:
            line = []
            for s in spec["seeds"][:seeds]:
                f = OUT / "test_renders" / f"{arm}_{p}_{s}.jpg"
                if not f.exists():
                    continue
                im = Image.open(f).convert("RGB")
                im.thumbnail((480, 480))
                r = next((r for r in rows if r["arm"] == arm and r["prompt"] == p and r["seed"] == s), None)
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
                d.text((4, 4), f"{arm} {s}  her {r['likeness'] if r else ''}" + (f" named {r['named']}" if r and 'named' in r else ""), fill=(255, 255, 255))
                line.append(im)
            if line:
                tiles.append(line)
        if not tiles:
            continue
        W = max(sum(t.width for t in line) for line in tiles)
        H = sum(max(t.height for t in line) for line in tiles)
        sheet = Image.new("RGB", (W, H), (20, 20, 20))
        y = 0
        for line in tiles:
            x = 0
            for t in line:
                sheet.paste(t, (x, y))
                x += t.width
            y += max(t.height for t in line)
        sheet.save(sd / f"test_{p}.jpg", quality=88)
        print(sd / f"test_{p}.jpg")


def signcheck():
    """Control: re-key with respond/don't swapped on multi-person captures (v2 set, REKEY_BLOCKS; job boxes reused from capture v2),
    then render dev multi-person prompts with the A2 key-lock (no carrier hooks). Per render the reader face (most attention to rlsa)
    vs the other faces: the method predicts Subject A moves to the non-reading face. Compared with A2 (unswapped)."""
    import torch
    capture_set("A2swap", V2_MULTI, V2_SOLO, V2_UNTRIG, jobs_tag="A2", swap=True)
    rig = Rig()
    her = rig.met.centroid.cpu().numpy()
    lock = torch.load(OUT / "lock_A2_mu1_lam0.1.pt")
    rig.lock = {b: dW.to(rig.pipe.dev, torch.bfloat16) for b, dW in lock.items()}
    f_json = OUT / "signcheck_records.json"
    out = []
    for arm in ("A2", "A2swap"):
        rig.A_override = {p: A.to(rig.pipe.dev, torch.bfloat16) for p, A in torch.load(OUT / f"{arm}_mu1.pt").items()}
        for case in ("bench", "cafe", "crew", "pugh"):
            text, k = HELD[case]
            sp = rig.lab.span(text, "rlsa")
            for s in HELD_SEEDS:
                im = rig.render(text, s, None, (1216, 832))
                fs = sorted(rig.met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                if len(fs) < 2:
                    continue
                at = read_attention(rig, text, s, (1216, 832), sp, [g.bbox.tolist() for g in fs])
                r = int(np.argmax(at))
                lk = [float(g.normed_embedding @ her) for g in fs]
                out.append({"arm": arm, "case": case, "seed": s, "attn": at, "likeness": lk, "reader": r})
                G.common.write(f_json, out)
                print(arm, case, s, "reader", r, [round(v, 3) for v in lk], [round(a, 2) for a in at], flush=True)
    for arm in ("A2", "A2swap"):
        rs = [r for r in out if r["arm"] == arm]
        rd = [r["likeness"][r["reader"]] for r in rs]
        ot = [v for r in rs for i, v in enumerate(r["likeness"]) if i != r["reader"]]
        print(f"{arm}: reader face {np.mean(rd):.3f}  non-reader faces {np.mean(ot):.3f}  (n={len(rs)})", flush=True)


def validate5():
    """Dev-set (HELD x 181909-12) stock-loader comparison: A3all_trigtok (current best) vs A5all mu1 / mu3 (trigtok)."""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    f_json = OUT / "validate_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    for mu in (1.0, 3.0):
        arm = f"A5all_mu{mu:g}_trigtok"
        model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(bound_file("A5all", mu, amp="trigamp_trigtok_l4"))), 1.0, 0)[0]
        for case, (text, k) in HELD.items():
            size = (1216, 832) if k > 1 else (832, 1216)
            for s in HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, size=size)
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"validate_{arm}_{case}_{s}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                out.append({"arm": arm, "case": case, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "pugh": [round(float(g.normed_embedding @ pugh), 3) for g in fs]})
                G.common.write(f_json, out)
                print(arm, case, s, out[-1]["likeness"], flush=True)
        del model
    report([r for r in out if r["arm"] in ("repair", "A3all_tf_trigtok", "A5all_mu1_trigtok", "A5all_mu3_trigtok")])


GENDER_PROMPTS = {"plain": "A medium photograph of rlsa standing next to Kate Winslet at a charity gala, both" + VIS + " Evening event photography.",
                  "rlsa_woman": "A medium photograph of rlsa woman standing next to Kate Winslet at a charity gala, both" + VIS + " Evening event photography.",
                  "rlsa_a_woman": "A medium photograph of rlsa, a woman, standing next to Kate Winslet at a charity gala, both" + VIS + " Evening event photography.",
                  "she": "A medium photograph of rlsa standing next to Kate Winslet at a charity gala, rlsa smiling as she poses, both" + VIS + " Evening event photography.",
                  "chair_rlsa_woman": "rlsa woman is sitting on the chair and a different woman is standing behind her."}


def gender_check():
    """Usage check (dev seeds 181909-12, not the frozen set): does writing rlsa as "rlsa woman" (the training-caption form) or "rlsa, a woman,"
    stop the slot from rendering as a man, and does binding still hold? Bound file vs repair; per face Subject A and Winslet likeness."""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    win = np.array(json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"]["Kate Winslet"])
    models = {"bound": comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(OUT / "rlsa_bound_A3all_mu1_lam0.1_trigtok.safetensors")), 1.0, 0)[0]}
    m = pipe.base
    for path, sc in G.HL_ARMS["repair"]:
        m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), sc, 0)
    models["repair"] = m
    out = []
    for arm, model in models.items():
        for name, text in GENDER_PROMPTS.items():
            for s in HELD_SEEDS:
                _, _, lat, _ = pipe.sample(model, text, s, size=(1216, 832))
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"gender_{arm}_{name}_{s}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:2]
                out.append({"arm": arm, "prompt": name, "seed": s, "her": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "winslet": [round(float(g.normed_embedding @ win), 3) for g in fs]})
                G.common.write(OUT / "gender_records.json", out)
                print(arm, name, s, out[-1]["her"], out[-1]["winslet"], flush=True)


NAMED_ORDER = {"theron_first": "Charlize Theron", "adams_first": "Amy Adams", "hathaway": "Anne Hathaway", "chastain": "Jessica Chastain"}


def named_order():
    """Usage check on dev seeds (181909-12; v5 capture prompts, not the frozen set): named pairs in both word orders, rlsa plain vs "rlsa woman",
    best file. Per face Subject A and the named person's concept likeness -> named_order_records.json (gender via gender-audit)."""
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    concepts = {n: np.array(c) for n, c in json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"].items()}
    model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(OUT / "rlsa_bound_A3all_mu1_lam0.1_trigtok.safetensors")), 1.0, 0)[0]
    out = []
    for name, who in NAMED_ORDER.items():
        for phr, text in (("plain", V5_MULTI[name]), ("woman", V5_MULTI[name].replace("rlsa", "rlsa woman", 1))):
            for s in HELD_SEEDS:
                _, _, lat, _ = pipe.sample(model, text, s, size=(1216, 832))
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"named_bound_{name}_{phr}_{s}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:2]
                out.append({"prompt": name, "phrasing": phr, "seed": s, "her": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "named": [round(float(g.normed_embedding @ concepts[who]), 3) for g in fs]})
                G.common.write(OUT / "named_order_records.json", out)
                print(name, phr, s, out[-1]["her"], out[-1]["named"], flush=True)
    for name in NAMED_ORDER:
        for phr in ("plain", "woman"):
            rs = [r for r in out if r["prompt"] == name and r["phrasing"] == phr and len(r["her"]) == 2]
            j = [int(np.argmax(r["named"])) for r in rs]
            print(f"{name:13s} {phr:6s} her {np.mean([r['her'][1 - i] for r, i in zip(rs, j)]):.2f}  named {np.mean([r['named'][i] for r, i in zip(rs, j)]):.2f}"
                  f"  her-on-named {np.mean([r['her'][i] for r, i in zip(rs, j)]):.2f}  (n={len(rs)})", flush=True)

CTRL = ("A3allrand", "A3allsham", "A3allcone")


def validate_ctrl(atags=CTRL):
    """Controls on the dev set (HELD x HELD_SEEDS), stock loader, trigtok amp: random-direction key lock of matched norm on the A3all executor
    (A3allrand), the same with the name key's angle to rlsa's current key (A3allcone) and the sham re-key (A3allsham). One records file per
    control (controls and validate5 run concurrently on two GPUs); reported next to repair and A3all_tf_trigtok."""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    rows = []
    for atag in atags:
        f = bound_file(atag, amp="trigamp_trigtok_l4")
        if not f.exists():
            continue
        f_json = OUT / f"validate_{atag}_records.json"
        out = json.loads(f_json.read_text()) if f_json.exists() else []
        done = {(r["arm"], r["case"], r["seed"]) for r in out}
        arm = f"{atag}_trigtok"
        model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(f)), 1.0, 0)[0]
        for case, (text, k) in HELD.items():
            size = (1216, 832) if k > 1 else (832, 1216)
            for s in HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, size=size)
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"validate_{arm}_{case}_{s}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                out.append({"arm": arm, "case": case, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                            "pugh": [round(float(g.normed_embedding @ pugh), 3) for g in fs]})
                G.common.write(f_json, out)
                print(arm, case, s, out[-1]["likeness"], flush=True)
        del model
        rows += out
    ref = [r for r in json.loads((OUT / "validate_records.json").read_text()) if r["arm"] in ("repair", "A3all_tf_trigtok")]
    report(ref + rows)


def gender_audit():
    """Per-face gender and age (insightface buffalo_l genderage, CPU) on every saved render of this run: dev (validate_*), gender_*, frozen
    test (test_renders/). Her slot = the face most like Subject A. -> gender_audit.json; prints per set / arm / prompt how often her slot
    and any face read as male."""
    import cv2
    from insightface.app import FaceAnalysis
    sys.path.insert(0, str(G.probes.OLD))
    import score_and_select
    _, her, _, _ = score_and_select.setup()
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "genderage"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=.5)
    spec = json.loads((TEST / "binding_test_v1.json").read_text())
    sets = {"dev": (sorted(OUT.glob("validate_*.jpg")), "validate_", set(HELD)), "gender": (sorted(OUT.glob("gender_*.jpg")), "gender_", set(GENDER_PROMPTS)),
            "named": (sorted(OUT.glob("named_*.jpg")), "named_", {f"{n}_{p}" for n in NAMED_ORDER for p in ("plain", "woman")}),
            "test": (sorted((OUT / "test_renders").glob("*.jpg")), "", {k for g in ("multi", "solo", "untriggered") for k in spec[g]})}
    rows = []
    for sname, (files, prefix, cases) in sets.items():
        for f in files:
            stem, seed = f.stem.removeprefix(prefix).rsplit("_", 1)
            case = max((c for c in cases if stem.endswith("_" + c)), key=len, default=None)
            if case is None:
                continue
            fs = sorted(app.get(cv2.imread(str(f))), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:3]
            rows.append({"set": sname, "arm": stem.removesuffix("_" + case), "case": case, "seed": int(seed),
                         "faces": [{"her": round(float(g.normed_embedding @ her), 3), "male": int(g.gender), "age": int(g.age)} for g in fs]})
    G.common.write(OUT / "gender_audit.json", rows)
    for key in dict.fromkeys((r["set"], r["arm"], r["case"]) for r in rows):
        rs = [r for r in rows if (r["set"], r["arm"], r["case"]) == key and r["faces"]]
        if not rs:
            continue
        slot = [max(r["faces"], key=lambda x: x["her"]) for r in rs]
        print(f"{key[0]:6s} {key[1]:24s} {key[2]:16s} her-slot male {sum(x['male'] for x in slot)}/{len(slot)} (her {np.mean([x['her'] for x in slot]):.2f})"
              f"  any face male {sum(any(x['male'] for x in r['faces']) for r in rs)}/{len(rs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["capture", "capture2", "eval", "held", "held-full", "keylock", "export", "validate", "capture3", "keylock3", "capture3-rest", "keylock3all", "export3", "validate3", "textsplice", "textvariants", "report", "export3tok", "capture4", "keylock4", "export4", "validate4", "testset", "test-analyze", "sheets", "signcheck", "capture5", "keylock5", "export5", "validate5", "gender", "capture-sham", "keylock-rand", "keylock-sham", "capture6", "keylock6", "export6", "export-nolock", "export-notxt", "keylock6b03", "export6b03", "validate6b03", "validate6", "capture6b", "keylock6b", "export6b", "validate6b", "keylock-cone", "export-ctrl", "validate-ctrl", "validate-cone", "named-order", "gender-audit"])
    {"capture": capture, "capture2": capture2, "keylock": keylock, "export": export, "validate": validate, "capture3": capture3, "keylock3": lambda: keylock("A3", lams=(0.1,)), "capture3-rest": capture3_rest, "keylock3all": lambda: keylock("A3all", lams=(0.1,)), "export3": lambda: (export("A3"), export("A3all")), "validate3": lambda: validate(("A2", "A3", "A3all")), "textsplice": textsplice, "textvariants": textvariants, "export3tok": lambda: export("A3all", amp="trigamp_trigtok_l4"), "capture4": capture4, "keylock4": lambda: keylock("A4all", lams=(0.1,), amp="trigamp_trigtok_l4"), "export4": lambda: export("A4all", amp="trigamp_trigtok_l4"), "validate4": lambda: validate(("A4all",), amp="trigamp_trigtok_l4"), "testset": testset, "test-analyze": test_analyze, "signcheck": signcheck, "capture5": capture5, "keylock5": lambda: (keylock("A5all", 1.0, (0.1,)), keylock("A5all", 3.0, (0.1,))), "export5": lambda: (export("A5all", 1.0, amp="trigamp_trigtok_l4"), export("A5all", 3.0, amp="trigamp_trigtok_l4")), "validate5": lambda: validate5(), "gender": gender_check, "capture-sham": capture_sham, "keylock-rand": lambda: keylock("A3allrand", lams=(0.1,), rand=True), "keylock-sham": lambda: keylock("A3allsham", lams=(0.1,)), "capture6": capture6, "keylock6": lambda: keylock("A6all", lams=(0.1,)), "export-nolock": lambda: export("A3all", amp="trigamp_trigtok_l4", nolock=True), "export-notxt": lambda: export("A3all", amp="trigamp_trigtok_l4", notxt=True), "keylock6b03": lambda: keylock("A6ballm03", lams=(0.1,)), "export6b03": lambda: export("A6ballm03", amp="trigamp_trigtok_l4"), "validate6b03": lambda: validate_ctrl(("A6ballm03",)), "export6": lambda: export("A6all", amp="trigamp_trigtok_l4"), "validate6": lambda: validate_ctrl(("A6all",)), "capture6b": capture6b, "keylock6b": lambda: keylock("A6ball", lams=(0.1,)), "export6b": lambda: export("A6ball", amp="trigamp_trigtok_l4"), "validate6b": lambda: validate_ctrl(("A6ball",)), "keylock-cone": lambda: keylock("A3allcone", lams=(0.1,), cone=True), "export-ctrl": lambda: [export(a, amp="trigamp_trigtok_l4") for a in CTRL if (OUT / f"lock_{a}_mu1_lam0.1.pt").exists() and not bound_file(a, amp="trigamp_trigtok_l4").exists()], "validate-ctrl": lambda: validate_ctrl(("A3allrand", "A3allsham")), "validate-cone": lambda: validate_ctrl(("A3allcone",)), "named-order": named_order, "gender-audit": gender_audit, "sheets": sheets, "report": lambda: report(json.loads((OUT / "validate_records.json").read_text())), "eval": evaluate, "held": evaluate_held, "held-full": lambda: evaluate_held(False)}[ap.parse_args().command]()
