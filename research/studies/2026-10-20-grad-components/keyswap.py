#!/usr/bin/env python3
"""Key-swap oracle (renders of the undistributed adapter stay local): does rlsa bind to one face once the binding heads see a name-like key?

The binding heads (blocks 7-11) read "this phrase is a person" from the key; identity rides in the values. rlsa's key is off the
person manifold (repair 0.64, original 0.82 vs names 0.97), so faces give it 0.01-0.03 attention. Here the post-qknorm key of the
rlsa span is replaced by the mean celebrity-name key (same block, kv group and step; Base, KV_TEMPLATE), values untouched.
Text tokens sit at RoPE position 0, so a key taken from another prompt is position-free.

  python keyswap.py targets    mean name key per (block 7-11, kv group, step) -> keyswap/targets.pt
  python keyswap.py run        conditions x prompts x seeds 181905-08: per-face likeness + face->span attention in the top-10 heads
  python keyswap.py summary    keyswap/summary.json
  python keyswap.py splice | splice-summary   per-token splice of the txtmlp conditioning (rlsa tokens vs the smear on the other words)
  python keyswap.py mask | mask-pugh          in-DiT attention masks on the phrase keys (rlsa, repair) and the Base + Florence Pugh control
  python keyswap.py attrib | attrib-summary   the same masks with the repair's block LoRA routed to all / image / text / no tokens
Conditions: none / name_top (only the kv groups of the top-10 binding heads) / name_b7_11 (every kv group in blocks 7-11) /
rand_b7_11 (random key, same norm) for the repair; none / name_b7_11 for the original LoRA. Runs on GPU0.
"""
import argparse, json, sys
import numpy as np
import grad_amp as G
from grad_amp import Pipe, Metrics, HL_ARMS, OP_STACK, RUN, common, vdec, to_img
import overnight

OUT = RUN / "keyswap"
BLOCKS = [7, 8, 9, 10, 11]
SEEDS = (181905, 181906, 181907, 181908)
LATE = (5, 7)
PROMPTS = {"chair": (G.BIND_RLSA["chair"], {"other": "a different woman"}),
           "pugh": (G.BIND_RLSA["pugh"], {"Pugh": "Florence Pugh"}),
           "cafe": (G.BIND_RLSA["cafe"], {}),
           "crew": (G.TRIG + overnight.MIXED["crew"], {}),
           "solo": (G.KV_TEMPLATE.format(who="rlsa"), {})}
CONDS = [("repair", "none"), ("repair", "name_top"), ("repair", "name_b7_11"), ("repair", "rand_b7_11"), ("naive", "none"), ("naive", "name_b7_11")]


class Lab:
    def __init__(self):
        import torch
        self.torch = torch
        self.pipe = pipe = Pipe(OP_STACK)
        import comfy.sd, comfy.utils  # on sys.path once Pipe has added the pinned checkout
        self.models = {"base": pipe.base}
        for arm in ("repair", "naive"):
            m = pipe.base
            for path, s in HL_ARMS[arm]:
                m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), s, 0)
            self.models[arm] = m
        a0 = pipe.dm.blocks[0].attn
        self.rep, self.kvheads = a0.heads // a0.kvheads, a0.kvheads
        self.top = [(h["block"], h["head"]) for h in json.loads((RUN / "bind" / "bind_analysis.json").read_text())["top_binding_heads"][:10]]
        self.st = {"span": None, "rec": None, "swap": None, "vtag": None}
        for b, blk in enumerate(pipe.dm.blocks):
            blk.attn.qknorm.knorm.register_forward_hook(self._knorm(b))
            blk.attn.wv.register_forward_hook(self._vtag(b))

    def _knorm(self, b):
        def f(mod, inp, out):
            st, step = self.st, self.pipe.step
            if st["span"] is None:
                return None
            sp = st["span"]
            if st["rec"] is not None:
                st["rec"][(b, step)] = out[0, :, sp].float().mean(1).cpu()
            if st["swap"] is not None and b in st["swap"]:
                groups, T = st["swap"][b], st["target"]
                out = out.clone()
                out[0, groups, sp[0]:sp[-1] + 1] = T[b][groups, step].to(out.device, out.dtype).unsqueeze(1)
                return out
            return None
        return f

    def _vtag(self, b):
        """Adds st["vtag"][b] (kvheads*headdim) to the phrase span's values: a strong, clean tag the faces that read it carry away."""
        def f(mod, inp, out):
            st = self.st
            if st["span"] is None or st["vtag"] is None or b not in st["vtag"]:
                return None
            out = out.clone()
            out[0, st["span"][0]:st["span"][-1] + 1] += st["vtag"][b].to(out.device, out.dtype)
            return out
        return f

    def span(self, text, phrase):
        n = lambda t: self.pipe.encode(t)[0][0][0].shape[1]
        i = text.index(phrase)
        full = n(text)
        tail = n(text[i + len(phrase):]) if text[i + len(phrase):].strip() else 0
        return list(range(full - n(text[i:]), full - tail))


def targets():
    import torch
    lab = Lab()
    keys = {}
    for who in G.ID_YOUNG + G.ID_OLD:
        text = G.KV_TEMPLATE.format(who=who)
        lab.st["span"], lab.st["rec"] = lab.span(text, who), {}
        lab.pipe.sample(lab.models["base"], text, 182401, size=(832, 1216))
        keys[who] = lab.st["rec"]
        print(who, len(lab.st["span"]), flush=True)
    lab.st["span"] = lab.st["rec"] = None
    T = {b: torch.stack([torch.stack([keys[w][(b, s)] for s in range(G.STEPS)], 1) for w in keys]).mean(0) for b in BLOCKS}  # kvH x steps x D
    g = torch.Generator().manual_seed(0)
    R = {b: (lambda r: r / r.norm(dim=-1, keepdim=True) * T[b].norm(dim=-1, keepdim=True))(torch.randn(T[b].shape, generator=g)) for b in BLOCKS}
    cos = {b: float(torch.nn.functional.cosine_similarity(torch.stack([keys[w][(b, 7)] for w in keys]), T[b][:, 7].unsqueeze(0).expand(len(keys), -1, -1), dim=-1).mean()) for b in BLOCKS}
    OUT.mkdir(exist_ok=True)
    torch.save({"name": T, "rand": R, "names": list(keys)}, OUT / "targets.pt")
    print("mean cos of each name key to the mean name key (step 7):", {b: round(c, 3) for b, c in cos.items()})


def targets_inslot():
    """In-slot name keys: the post-qknorm key of each celebrity's name span in the chair sentence itself (Base, seed 182401), every block,
    kv group and step, averaged over names -> keyswap/targets_inslot.pt (the key that binds in this slot, not the red-carpet mean)."""
    import torch
    lab = Lab()
    keys = {}
    nb = len(lab.pipe.dm.blocks)
    for who in G.ID_YOUNG + G.ID_OLD:
        text = PROMPTS["chair"][0].replace("rlsa", who)
        lab.st["span"], lab.st["rec"] = lab.span(text, who), {}
        lab.pipe.sample(lab.models["base"], text, 182401, size=(1216, 832))
        keys[who] = lab.st["rec"]
        print(who, len(lab.st["span"]), flush=True)
    lab.st["span"] = lab.st["rec"] = None
    T = {b: torch.stack([torch.stack([keys[w][(b, s)] for s in range(G.STEPS)], 1) for w in keys]).mean(0) for b in range(nb)}
    cos = {b: round(float(torch.nn.functional.cosine_similarity(torch.stack([keys[w][(b, 7)] for w in keys]), T[b][:, 7].unsqueeze(0).expand(len(keys), -1, -1), dim=-1).mean()), 3) for b in range(nb)}
    torch.save({"name": T, "names": list(keys)}, OUT / "targets_inslot.pt")
    print("mean cos of each in-slot name key to the mean (step 7):", cos)


def targets_tok():
    """Which token of a name binds? For each celebrity in the chair sentence (Base, seed 182401): per-token post-qknorm keys of the name span
    (every block, kv group, step) and, in a second pass with the named face's box, that face's attention to each span token in the top-10
    binding heads (late steps). Target = per name the key of its most-attended token, averaged over names -> keyswap/targets_tok.pt."""
    import torch
    import torch.nn.functional as F
    lab = Lab()
    pipe, st = lab.pipe, lab.st
    from comfy.ldm.krea2.model import apply_rope
    met = Metrics(pipe.dev)
    recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
    cent = {}
    for r in recs:
        if "celebrity" in r["group"]:
            cent.setdefault(r["who"], []).append(np.array(r["arcface"]))
    cent = {k: (lambda v: v / np.linalg.norm(v))(np.mean(v, 0)) for k, v in cent.items()}
    keys, cap = {}, {"face": None}

    def knorm(b):
        def f(mod, inp, out):
            if cap.get("keys") is not None:
                cap["keys"][(b, pipe.step)] = out[0, :, cap["sp"]].float().cpu()  # kvH x span x D
        return f
    for b, blk in enumerate(pipe.dm.blocks):
        blk.attn.qknorm.knorm.register_forward_hook(knorm(b))

    def att(bi):
        attn = pipe.dm.blocks[bi].attn
        heads = [h for b, h in lab.top if b == bi]

        def f(mod, args):
            if cap["face"] is None or pipe.step not in LATE:
                return None
            x, freqs = args[0], args[1]
            q = attn.wq(x).unflatten(-1, (attn.heads, attn.headdim)).transpose(1, 2)
            k = attn.wk(x).unflatten(-1, (attn.kvheads, attn.headdim)).transpose(1, 2)
            q, k = attn.qknorm(q, k)
            q, k = apply_rope(q, k, freqs)
            k = k.repeat_interleave(lab.rep, dim=1)[0][heads].float()
            p = F.softmax(q[0][heads][:, cap["face"] + cap["T"]].float() @ k.transpose(1, 2) / attn.headdim ** 0.5, dim=-1)
            cap["att"].append(p[:, :, cap["sp"]].mean((0, 1)).cpu())
            return None
        return f
    for b in sorted({b for b, _ in lab.top}):
        pipe.dm.blocks[b].attn.register_forward_pre_hook(att(b))
    best = {}
    for who in G.ID_YOUNG + G.ID_OLD:
        text = PROMPTS["chair"][0].replace("rlsa", who)
        sp = lab.span(text, who)
        cap.update(sp=sp, keys={}, T=pipe.encode(text)[0][0][0].shape[1])
        _, _, lat, _ = pipe.sample(lab.models["base"], text, 182401, size=(1216, 832))
        keys[who] = cap["keys"]
        cap["keys"] = None
        fs = met.app.get(np.asarray(to_img(vdec(pipe, lat)))[:, :, ::-1])
        if not fs:
            continue
        g = max(fs, key=lambda g: float(g.normed_embedding @ cent[who]))
        cap.update(face=torch.tensor(face_tokens(g.bbox.tolist(), grow=0.0), device=pipe.dev), att=[])
        pipe.sample(lab.models["base"], text, 182401, size=(1216, 832))
        a = torch.stack(cap["att"]).mean(0)
        cap["face"] = None
        best[who] = int(a.argmax())
        print(who, "likeness", round(float(g.normed_embedding @ cent[who]), 3), "attention per token", [round(float(v), 3) for v in a], flush=True)
    nb = len(pipe.dm.blocks)
    T = {b: torch.stack([torch.stack([keys[w][(b, s)][:, best[w]] for s in range(G.STEPS)], 1) for w in best]).mean(0) for b in range(nb)}
    torch.save({"name": T, "names": list(best), "best_token": best}, OUT / "targets_tok.pt")


def run():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    lab = Lab()
    from comfy.ldm.krea2.model import apply_rope
    pipe, st = lab.pipe, lab.st
    met = Metrics(pipe.dev)
    tg = torch.load(OUT / "targets.pt", weights_only=False)
    recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    her = met.centroid.cpu().numpy()
    top_groups = {}
    for b, h in lab.top:
        top_groups.setdefault(b, set()).add(h // lab.rep)
    swaps = {"none": None, "name_top": ({b: sorted(g) for b, g in top_groups.items()}, "name"),
             "name_b7_11": ({b: list(range(lab.kvheads)) for b in BLOCKS}, "name"), "rand_b7_11": ({b: list(range(lab.kvheads)) for b in BLOCKS}, "rand")}
    GW, GH = 1216 // 16, 832 // 16
    meas = {"faces": None}

    def pre(bi):
        attn = pipe.dm.blocks[bi].attn
        heads = [h for b, h in lab.top if b == bi]

        def f(mod, args):
            if meas["faces"] is None or pipe.step not in LATE:
                return None
            x, freqs = args[0], args[1]
            with torch.no_grad():
                q = attn.wq(x).unflatten(-1, (attn.heads, attn.headdim)).transpose(1, 2)
                k = attn.wk(x).unflatten(-1, (attn.kvheads, attn.headdim)).transpose(1, 2)
                q, k = attn.qknorm(q, k)  # the knorm hook applies the swap here too
                q, k = apply_rope(q, k, freqs)
                k = k.repeat_interleave(lab.rep, dim=1)
                T = meas["T"]
                res = []
                for fi in meas["faces"]:
                    idx = torch.tensor(fi, device=x.device) + T
                    p = F.softmax((q[0][heads][:, idx].float() @ k[0][heads].float().transpose(1, 2)) / attn.headdim ** 0.5, dim=-1)
                    res.append({n: p[:, :, sp].sum(-1).mean(-1).cpu().tolist() for n, sp in meas["spans"].items()})
                meas["rows"].append({"block": bi, "step": pipe.step, "heads": heads, "faces": res})
            return None
        return f
    for b in sorted({b for b, _ in lab.top}):
        pipe.dm.blocks[b].attn.register_forward_pre_hook(pre(b))
    f_json = OUT / "records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["prompt"], r["arm"], r["cond"], r["seed"]) for r in out}
    for pid, (text, extra) in PROMPTS.items():
        spans = {"rlsa": lab.span(text, "rlsa")} | {k: lab.span(text, p) for k, p in extra.items()}
        size = (832, 1216) if pid == "solo" else (1216, 832)
        for arm, cond in CONDS:
            sw = swaps[cond]
            st["swap"], st["target"] = (sw[0], tg[sw[1]]) if sw else (None, None)
            for seed in SEEDS:
                if (pid, arm, cond, seed) in done:
                    continue
                st["span"] = spans["rlsa"] if sw else None
                _, _, lat, _ = pipe.sample(lab.models[arm], text, seed, size=size)
                im = to_img(vdec(pipe, lat))
                im.save(OUT / f"{pid}_{arm}_{cond}_{seed}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:1 if pid == "solo" else 3]
                info = [{"bbox": g.bbox.tolist(), "her": float(g.normed_embedding @ her), "pugh": float(g.normed_embedding @ pugh)} for g in fs]
                if pid != "solo" and len(fs) >= 2:
                    meas["faces"] = [[r_ * GW + c for r_ in range(max(0, int(g.bbox[1] // 16)), min(GH, int(g.bbox[3] // 16) + 1))
                                      for c in range(max(0, int(g.bbox[0] // 16)), min(GW, int(g.bbox[2] // 16) + 1))] for g in fs]
                    meas["T"], meas["spans"], meas["rows"] = pipe.encode(text)[0][0][0].shape[1], spans, []
                    pipe.sample(lab.models[arm], text, seed, size=size)
                    meas["faces"] = None
                    for i, d in enumerate(info):
                        for n in spans:
                            d["attn_" + n] = float(np.mean([np.mean(r["faces"][i][n]) for r in meas["rows"]]))
                st["span"] = None
                out.append({"prompt": pid, "arm": arm, "cond": cond, "seed": seed, "faces": info})
                common.write(f_json, out)
                print(pid, arm, cond, seed, [{k: round(v, 3) for k, v in d.items() if k != "bbox"} for d in info], flush=True)


def summary(name="", conds=CONDS):
    rows = json.loads((OUT / f"{name}records.json").read_text())
    m = lambda v: round(float(np.mean(v)), 3) if v else None
    summ = {}
    for pid in PROMPTS:
        for arm, cond in conds:
            rs = [r for r in rows if r["prompt"] == pid and r["arm"] == arm and r["cond"] == cond and r["faces"]]
            subj, strangers, a_her, a_str, pugh_face = [], [], [], [], []
            for r in rs:
                fs = r["faces"]
                if pid == "pugh":  # Pugh = the face closest to her concept; Subject A = the best of the rest
                    ip = int(np.argmax([f["pugh"] for f in fs]))
                    pugh_face.append(fs[ip]["pugh"])
                    rest = [f for i, f in enumerate(fs) if i != ip]
                    if not rest:
                        continue
                    hf = max(rest, key=lambda f: f["her"])
                    others = [f for f in fs if f is not hf]
                else:
                    hf = max(fs, key=lambda f: f["her"])
                    others = [f for f in fs if f is not hf]
                subj.append(hf["her"])
                strangers += [f["her"] for f in others]
                if "attn_rlsa" in hf:
                    a_her.append(hf["attn_rlsa"])
                    a_str += [f["attn_rlsa"] for f in others]
            summ[f"{pid} | {arm} | {cond}"] = {"n": len(rs), "subject": m(subj), "stranger_likeness": m(strangers),
                                               "strangers_ge_0.35": f"{sum(v >= .35 for v in strangers)}/{len(strangers)}",
                                               "attn_rlsa_subject": m(a_her), "attn_rlsa_strangers": m(a_str), "pugh_face_pugh": m(pugh_face)}
    common.write(OUT / f"{name}summary.json", summ)
    print(json.dumps(summ, indent=1))

STANDINS = ["a woman", "Anna", "Mira", "Nora", "Vera", "Tess"]
SPLICE = [(arm, c) for arm in ("repair", "naive") for c in ("trig", "ref", "rlsa_only", "smear_only")]


def splice():
    """Which conditioning tokens carry her to every face? The DiT's text conditioning (txtmlp output, constant over steps) of the
    triggered prompt (trig) vs the same prompt with rlsa replaced by a stand-in of equal token count (ref), spliced per token:
    rlsa_only = ref everywhere except the rlsa span; smear_only = trig everywhere except the rlsa span (rlsa's mixing into
    the other words through txtfusion). Same arm weights in every condition."""
    import torch
    lab = Lab()
    pipe = lab.pipe
    met = Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    cap = {}
    pipe.dm.txtmlp.register_forward_hook(lambda mod, inp, out: None if "c" in cap else cap.__setitem__("c", out.detach().clone()))
    n = lambda t: pipe.encode(t)[0][0][0].shape[1]
    f_json = OUT / "splice_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["prompt"], r["arm"], r["cond"], r["seed"]) for r in out}
    for pid, (text, _) in PROMPTS.items():
        sp = lab.span(text, "rlsa")
        ref = next(r for r in (text.replace("rlsa", w, 1) for w in STANDINS) if n(r) == n(text))
        print(pid, "span", len(sp), "ref:", ref[:60], flush=True)
        size = (832, 1216) if pid == "solo" else (1216, 832)
        for arm, cond in SPLICE:
            for seed in SEEDS:
                if (pid, arm, cond, seed) in done:
                    continue
                if cond in ("trig", "ref"):
                    cap.clear()
                    _, _, lat, _ = pipe.sample(lab.models[arm], text if cond == "trig" else ref, seed, size=size)
                    C = cap["c"]
                    if cond == "trig":
                        Ct = C
                    else:
                        Cr = C
                else:
                    mix = Ct.clone() if cond == "smear_only" else Cr.clone()
                    mix[:, sp[0]:sp[-1] + 1] = (Cr if cond == "smear_only" else Ct)[:, sp[0]:sp[-1] + 1]
                    _, _, lat, _ = pipe.sample(lab.models[arm], text, seed, E=(mix - Ct).float(), size=size)
                im = to_img(vdec(pipe, lat))
                im.save(OUT / f"splice_{pid}_{arm}_{cond}_{seed}.jpg", quality=90)
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:1 if pid == "solo" else 3]
                info = [{"bbox": g.bbox.tolist(), "her": float(g.normed_embedding @ her), "pugh": float(g.normed_embedding @ pugh)} for g in fs]
                out.append({"prompt": pid, "arm": arm, "cond": cond, "seed": seed, "faces": info})
                common.write(f_json, out)
                print(pid, arm, cond, seed, [{k: round(v, 3) for k, v in d.items() if k != "bbox"} for d in info], flush=True)


def face_tokens(bbox, grow=0.3, GW=1216 // 16, GH=832 // 16):
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) * grow, (y1 - y0) * grow
    return [r * GW + c for r in range(max(0, int((y0 - dy) // 16)), min(GH, int((y1 + dy) // 16) + 1))
            for c in range(max(0, int((x0 - dx) // 16)), min(GW, int((x1 + dx) // 16) + 1))]


def phrase_mask(cond, T, sp, fb, dev):
    """Boolean attention mask (True = may attend) removing chosen queries' access to the phrase keys `sp`; None for full / none.
    txt = other text tokens, img = all image tokens, block_face{i} = face i, only_face{i} = text and all image tokens outside face i."""
    import torch
    if cond in ("full", "none"):
        return None
    M = torch.ones(1, 1, T + 76 * 52, T + 76 * 52, dtype=torch.bool, device=dev)
    rl = slice(sp[0], sp[-1] + 1)
    if "txt" in cond or cond.startswith("only_face"):
        M[0, 0, [i for i in range(T) if i not in sp], rl] = False
    if "img" in cond:
        M[0, 0, T:, rl] = False
    if "block_face" in cond:
        M[0, 0, [T + i for i in face_tokens(fb[int(cond[-1])])], rl] = False
    if cond.startswith("only_face"):
        keep = set(face_tokens(fb[int(cond[-1])]))
        M[0, 0, [T + i for i in range(76 * 52) if i not in keep], rl] = False
    return M


def top_boxes(fs, k=2):
    return [g.bbox.tolist() for g in sorted(fs, key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]]


def score_boxes(fs, fb, target, key):
    """Score the detected face nearest each reference box, so face i keeps its identity across conditions."""
    info = []
    for i, b in enumerate(fb or [g.bbox.tolist() for g in fs]):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        g = min(fs, key=lambda g: abs((g.bbox[0] + g.bbox[2]) / 2 - cx) + abs((g.bbox[1] + g.bbox[3]) / 2 - cy), default=None)
        info.append({"face": i, key: float(g.normed_embedding @ target) if g is not None else None,
                     "shift": float(abs((g.bbox[0] + g.bbox[2]) / 2 - cx) + abs((g.bbox[1] + g.bbox[3]) / 2 - cy)) if g is not None else None})
    return info


MASK_JOBS = {"rlsa": {"arm": "repair", "phrase": "rlsa", "score": "her", "prompts": (("chair", PROMPTS["chair"][0], (0, 1)), ("cafe", PROMPTS["cafe"][0], (0,)))},
             "pugh": {"arm": "base", "phrase": "Florence Pugh", "score": "pugh", "prompts": (("pugh_chair", PROMPTS["chair"][0].replace("rlsa", "Florence Pugh"), (0, 1)),)}}


def mask(job="rlsa"):
    """Where does a phrase's identity spread inside the DiT? Conditioning = phrase-only splice (only the phrase span from the real
    prompt, every other token from the stand-in prompt); a boolean attention mask in every block and step removes chosen query
    rows' access to the phrase keys: txt = other text tokens, img = all image tokens, face i = image tokens of face i (box from
    the unmasked splice render "none", +30 %). only_face{i}: text and every image token outside face i blocked.
    Jobs: rlsa (repair weights, Subject A likeness) and pugh (Base, "Florence Pugh" in the chair sentence, Pugh likeness) as the
    control for whether a named identity stays on the face that reads it. "full" = unspliced prompt, no mask."""
    import torch
    J = MASK_JOBS[job]
    lab = Lab()
    pipe = lab.pipe
    met = Metrics(pipe.dev)
    if job == "rlsa":
        target = met.centroid.cpu().numpy()
    else:
        recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
        target = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
        target /= np.linalg.norm(target)
    cap, st = {}, {"mask": None}
    pipe.dm.txtmlp.register_forward_hook(lambda mod, inp, out: None if "c" in cap else cap.__setitem__("c", out.detach().clone()))
    for b in pipe.dm.blocks:
        b.attn.register_forward_pre_hook(lambda mod, args: None if st["mask"] is None else (args[0], args[1], st["mask"]))
    n = lambda t: pipe.encode(t)[0][0][0].shape[1]
    boxes = {(r["prompt"], r["seed"]): [f["bbox"] for f in r["faces"]] for r in json.loads((OUT / "splice_records.json").read_text())
             if r["arm"] == "repair" and r["cond"] == "rlsa_only"}
    f_json = OUT / "mask_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["prompt"], r["cond"], r["seed"]) for r in out}
    model = lab.models[J["arm"]]
    for pid, text, faces_tested in J["prompts"]:
        sp = lab.span(text, J["phrase"])
        ref = next(r for r in (text.replace(J["phrase"], w, 1) for w in STANDINS + ["a young woman", "the young woman", "a dark-haired woman"]) if n(r) == n(text))
        print(pid, "span", len(sp), "ref:", ref[:70], flush=True)
        C = {}
        for k, t in (("t", text), ("r", ref)):
            cap.clear()
            pipe.sample(model, t, SEEDS[0], size=(1216, 832))
            C[k] = cap["c"]
        mix = C["r"].clone()
        mix[:, sp[0]:sp[-1] + 1] = C["t"][:, sp[0]:sp[-1] + 1]
        E = (mix - C["t"]).float()
        T = C["t"].shape[1]
        conds = ([] if job == "rlsa" else ["full", "none"]) + ["txt", "img", "txt+img"] + [f"txt+block_face{i}" for i in faces_tested] + [f"only_face{i}" for i in faces_tested]
        for seed in SEEDS:
            for cond in conds:
                if (pid, cond, seed) in done:
                    continue
                fb = boxes.get((pid, seed))
                st["mask"] = phrase_mask(cond, T, sp, fb, pipe.dev)
                _, _, lat, _ = pipe.sample(model, text, seed, E=None if cond == "full" else E, size=(1216, 832))
                st["mask"] = None
                im = to_img(vdec(pipe, lat))
                im.save(OUT / f"mask_{pid}_{cond}_{seed}.jpg", quality=90)
                fs = met.app.get(np.asarray(im)[:, :, ::-1])
                if cond == "none":
                    boxes[(pid, seed)] = fb = top_boxes(fs)
                info = score_boxes(fs, fb, target, J["score"])
                out.append({"prompt": pid, "cond": cond, "seed": seed, "faces": info})
                common.write(f_json, out)
                print(pid, cond, seed, [(d["face"], None if d[J["score"]] is None else round(d[J["score"]], 3), None if d["shift"] is None else round(d["shift"])) for d in info], flush=True)

ATTRIB_ROUTES = ["all", "image", "text", "none"]
ATTRIB_CONDS = ["full", "none", "img", "txt", "only_face0", "only_face1"]


def attrib():
    """Which adapter part makes rlsa a broadcast? Text side of the repair (txtfusion LoRA + amp x6) always on; the repair's block LoRA
    (blocks 0-27) added by hooks at all / image / text / no token positions (as grad_amp.routed). Per route, the chair sentence under the
    full trigger and under the rlsa-only splice with the mask conditions of mask(): img = relay only, txt = direct reads only,
    only_face{i} = one face reads rlsa. A route without the text-token LoRA that still relays or copies clears the in-DiT text processing."""
    import torch
    from safetensors.torch import load_file
    lab = Lab()
    pipe = lab.pipe
    import comfy.sd
    met = Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    sd = load_file(str(G.BLEND))
    amp = load_file(str(G.AMP / "trigamp_aside_l4.safetensors"))
    text_side = {k: v for k, v in sd.items() if ".txtfusion." in k} | {"diffusion_model.txtmlp.3.diff": 6.0 * amp["diffusion_model.txtmlp.3.diff"]}
    model, _ = comfy.sd.load_lora_for_models(pipe.base, None, text_side, 1.0, 0)
    lora = {k.removeprefix("diffusion_model.").removesuffix(".lora_A.weight"): (v.to(pipe.dev, torch.bfloat16), sd[k.replace("lora_A", "lora_B")].to(pipe.dev, torch.bfloat16))
            for k, v in sd.items() if ".blocks." in k and k.endswith("lora_A.weight")}
    st = {"route": "all", "T": 0, "mask": None}

    def hook(path):
        A, B = lora[path]

        def f(mod, inp, out):
            if st["route"] == "none":
                return None
            d = (inp[0].to(A.dtype) @ A.T) @ B.T
            if st["route"] == "text":
                d[:, st["T"]:] = 0
            elif st["route"] == "image":
                d[:, :st["T"]] = 0
            return out + d.to(out.dtype)
        return f
    for p in lora:
        pipe.dm.get_submodule(p).register_forward_hook(hook(p))
    cap = {}
    pipe.dm.txtmlp.register_forward_hook(lambda mod, inp, out: None if "c" in cap else cap.__setitem__("c", out.detach().clone()))
    for b in pipe.dm.blocks:
        b.attn.register_forward_pre_hook(lambda mod, args: None if st["mask"] is None else (args[0], args[1], st["mask"]))
    n = lambda t: pipe.encode(t)[0][0][0].shape[1]
    text = PROMPTS["chair"][0]
    sp = lab.span(text, "rlsa")
    ref = next(r for r in (text.replace("rlsa", w, 1) for w in STANDINS) if n(r) == n(text))
    C = {}
    for k, t in (("t", text), ("r", ref)):
        cap.clear()
        pipe.sample(model, t, SEEDS[0], size=(1216, 832))
        C[k] = cap["c"]
    mix = C["r"].clone()
    mix[:, sp[0]:sp[-1] + 1] = C["t"][:, sp[0]:sp[-1] + 1]
    E = (mix - C["t"]).float()
    T = st["T"] = C["t"].shape[1]
    f_json = OUT / "attrib_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["route"], r["cond"], r["seed"]) for r in out}
    boxes = {}
    for route in ATTRIB_ROUTES:
        st["route"] = route
        for seed in SEEDS:
            for cond in ATTRIB_CONDS:
                if (route, cond, seed) in done:
                    continue
                fb = boxes.get((route, seed))
                st["mask"] = phrase_mask(cond, T, sp, fb, pipe.dev)
                _, _, lat, _ = pipe.sample(model, text, seed, E=None if cond == "full" else E, size=(1216, 832))
                st["mask"] = None
                im = to_img(vdec(pipe, lat))
                im.save(OUT / f"attrib_{route}_{cond}_{seed}.jpg", quality=90)
                fs = met.app.get(np.asarray(im)[:, :, ::-1])
                if cond == "none":
                    boxes[(route, seed)] = fb = top_boxes(fs)
                info = score_boxes(fs, fb, her, "her")
                out.append({"route": route, "cond": cond, "seed": seed, "faces": info})
                common.write(f_json, out)
                print(route, cond, seed, [(d["face"], None if d["her"] is None else round(d["her"], 3), None if d["shift"] is None else round(d["shift"])) for d in info], flush=True)
    st["route"] = "all"


def attrib_summary():
    rows = json.loads((OUT / "attrib_records.json").read_text())
    summ = {}
    for route in ATTRIB_ROUTES:
        for cond in ATTRIB_CONDS:
            a = np.array([[np.nan if f["her"] is None else f["her"] for f in r["faces"][:2]] + [np.nan] * (2 - len(r["faces"][:2]))
                          for r in rows if r["route"] == route and r["cond"] == cond], dtype=float)
            if len(a):
                summ[f"{route} | {cond}"] = {"face0": round(float(np.nanmean(a[:, 0])), 3), "face1": round(float(np.nanmean(a[:, 1])), 3),
                                             "per_seed": [[None if np.isnan(v) else round(float(v), 2) for v in x] for x in a]}
    common.write(OUT / "attrib_summary.json", summ)
    for k, v in summ.items():
        print(f"{k:22s} {v['face0']:.3f} / {v['face1']:.3f}   {v['per_seed']}")


def switch_dose():
    """Can a strong, local switch bind her to one face? Repair weights, chair sentence, rlsa-only splice with the rlsa-span conditioning
    change scaled by k (span = ref + k (trig - ref)); only_face{i} = only face i may read rlsa (as mask()); none = no mask at the same k.
    If face i climbs with k while the other face stays low, the executor is local and binding needs a strong local read; if both climb
    together, the image-side LoRA copies her across faces and binding needs an executor change."""
    lab = Lab()
    pipe = lab.pipe
    met = Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    cap, st = {}, {"mask": None}
    pipe.dm.txtmlp.register_forward_hook(lambda mod, inp, out: None if "c" in cap else cap.__setitem__("c", out.detach().clone()))
    for b in pipe.dm.blocks:
        b.attn.register_forward_pre_hook(lambda mod, args: None if st["mask"] is None else (args[0], args[1], st["mask"]))
    n = lambda t: pipe.encode(t)[0][0][0].shape[1]
    boxes = {r["seed"]: [f["bbox"] for f in r["faces"]] for r in json.loads((OUT / "splice_records.json").read_text())
             if r["prompt"] == "chair" and r["arm"] == "repair" and r["cond"] == "rlsa_only"}
    model = lab.models["repair"]
    text = PROMPTS["chair"][0]
    sp = lab.span(text, "rlsa")
    ref = next(r for r in (text.replace("rlsa", w, 1) for w in STANDINS) if n(r) == n(text))
    C = {}
    for k, t in (("t", text), ("r", ref)):
        cap.clear()
        pipe.sample(model, t, SEEDS[0], size=(1216, 832))
        C[k] = cap["c"]
    T = C["t"].shape[1]
    f_json = OUT / "dose_records.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["k"], r["cond"], r["seed"]) for r in out}
    for k in (2.0, 4.0, 8.0):
        mix = C["r"].clone()
        mix[:, sp[0]:sp[-1] + 1] += k * (C["t"] - C["r"])[:, sp[0]:sp[-1] + 1]
        E = (mix - C["t"]).float()
        for seed in SEEDS:
            for cond in ("none", "only_face0", "only_face1"):
                if (k, cond, seed) in done:
                    continue
                st["mask"] = phrase_mask(cond, T, sp, boxes[seed], pipe.dev)
                _, _, lat, _ = pipe.sample(model, text, seed, E=E, size=(1216, 832))
                st["mask"] = None
                im = to_img(vdec(pipe, lat))
                im.save(OUT / f"dose_k{k:g}_{cond}_{seed}.jpg", quality=90)
                info = score_boxes(met.app.get(np.asarray(im)[:, :, ::-1]), boxes[seed], her, "her")
                out.append({"k": k, "cond": cond, "seed": seed, "faces": info})
                common.write(f_json, out)
                print(f"k{k:g}", cond, seed, [(d["face"], None if d["her"] is None else round(d["her"], 3), None if d["shift"] is None else round(d["shift"])) for d in info], flush=True)


PROBE_BLOCKS = [8, 10, 12, 15, 20]
PROBE_STEPS = [3, 5, 7]
PROBE_CONDS = ["only_face0", "only_face1", "txt+img", "none"]


CHAIR_PUGH = PROMPTS["chair"][0].replace("rlsa", "Florence Pugh")
# job: (arm, text, phrase, key-swap blocks, key target file, value-tag scale)
PROBE_JOBS = {"rlsa": ("repair", PROMPTS["chair"][0], "rlsa", BLOCKS, "targets.pt", 0.0),
              "pugh": ("base", CHAIR_PUGH, "Florence Pugh", None, None, 0.0),
              "rlsa_inslot7_11": ("repair", PROMPTS["chair"][0], "rlsa", BLOCKS, "targets_inslot.pt", 0.0),
              "rlsa_inslot_all": ("repair", PROMPTS["chair"][0], "rlsa", list(range(28)), "targets_inslot.pt", 0.0),
              "rlsa_inslot_all_vtag": ("repair", PROMPTS["chair"][0], "rlsa", list(range(28)), "targets_inslot.pt", 3.0),
              "rlsa_tok_all": ("repair", PROMPTS["chair"][0], "rlsa", list(range(28)), "targets_tok.pt", 0.0)}


def probe(jobs=("rlsa", "pugh")):
    """D1 (reassignment plan §4): does a face that read the phrase carry a linearly separable "read" feature?
    Chair sentence, phrase-only splice; PROBE_JOBS sets the weights, the key swap (blocks, target file: red-carpet mean or in-slot mean)
    and an optional value tag (a fixed random direction per block/kv group, scale x the mean norm of the span's values). Per condition
    (only_face0 / only_face1 / txt+img / none) the mean block output over each face's tokens and the image mean at PROBE_BLOCKS x
    PROBE_STEPS, plus each face's attention to the span in the top-10 binding heads (late steps), captured in the masked render itself.
    Boxes: rlsa from the splice rlsa_only renders, Pugh from a fresh unmasked splice render."""
    import torch
    import torch.nn.functional as F
    lab = Lab()
    pipe = lab.pipe
    from comfy.ldm.krea2.model import apply_rope
    met = Metrics(pipe.dev)
    st, cap, cc = lab.st, {}, {}
    pipe.dm.txtmlp.register_forward_hook(lambda mod, inp, out: None if "c" in cc else cc.__setitem__("c", out.detach().clone()))
    for b in pipe.dm.blocks:
        b.attn.register_forward_pre_hook(lambda mod, args: None if cap.get("mask") is None else (args[0], args[1], cap["mask"]))

    def blk(i):
        def f(mod, inp, out):
            if cap.get("faces") is None or pipe.step not in PROBE_STEPS:
                return None
            img = out[0, cap["T"]:].float()
            cap["rec"][(i, pipe.step)] = (torch.stack([img[idx].mean(0) for idx in cap["faces"]]).cpu(), img.mean(0).cpu())
            return None
        return f
    for i in PROBE_BLOCKS:
        pipe.dm.blocks[i].register_forward_hook(blk(i))

    def att(bi):
        attn = pipe.dm.blocks[bi].attn
        heads = [h for b, h in lab.top if b == bi]

        def f(mod, args):
            if cap.get("faces") is None or pipe.step not in LATE:
                return None
            x, freqs = args[0], args[1]
            q = attn.wq(x).unflatten(-1, (attn.heads, attn.headdim)).transpose(1, 2)
            k = attn.wk(x).unflatten(-1, (attn.kvheads, attn.headdim)).transpose(1, 2)
            q, k = attn.qknorm(q, k)  # the knorm hook applies the key swap here too
            q, k = apply_rope(q, k, freqs)
            k = k.repeat_interleave(lab.rep, dim=1)[0][heads].float()
            for fi, idx in enumerate(cap["faces"]):
                p = F.softmax(q[0][heads][:, idx + cap["T"]].float() @ k.transpose(1, 2) / attn.headdim ** 0.5, dim=-1)
                cap["att"].setdefault(fi, []).append(float(p[:, :, cap["sp"][0]:cap["sp"][-1] + 1].sum(-1).mean()))
            return None
        return f
    for b in sorted({b for b, _ in lab.top}):
        pipe.dm.blocks[b].attn.register_forward_pre_hook(att(b))
    recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    boxes = {("rlsa", r["seed"]): [f["bbox"] for f in r["faces"]][:2] for r in json.loads((OUT / "splice_records.json").read_text())
             if r["prompt"] == "chair" and r["arm"] == "repair" and r["cond"] == "rlsa_only"}
    n = lambda t: pipe.encode(t)[0][0][0].shape[1]
    f_pt = OUT / "probe_records.pt"
    out = torch.load(f_pt, weights_only=False) if f_pt.exists() else []
    done = {(r["job"], r["cond"], r["seed"]) for r in out}
    for job in jobs:
        arm, text, phrase, swap_blocks, tfile, vscale = PROBE_JOBS[job]
        bkey = "rlsa" if phrase == "rlsa" else job
        target = met.centroid.cpu().numpy() if phrase == "rlsa" else pugh
        model = lab.models[arm]
        sp = lab.span(text, phrase)
        ref = next(r for r in (text.replace(phrase, w, 1) for w in STANDINS + ["a young woman", "the young woman"]) if n(r) == n(text))
        C = {}
        for k, t in (("t", text), ("r", ref)):
            cc.clear()
            pipe.sample(model, t, SEEDS[0], size=(1216, 832))
            C[k] = cc["c"]
        mix = C["r"].clone()
        mix[:, sp[0]:sp[-1] + 1] = C["t"][:, sp[0]:sp[-1] + 1]
        E = (mix - C["t"]).float()
        T = C["t"].shape[1]
        st["span"] = sp
        if swap_blocks:
            st["swap"], st["target"] = {b: list(range(lab.kvheads)) for b in swap_blocks}, torch.load(OUT / tfile, weights_only=False)["name"]
        if vscale:
            vn = {}
            hv = [pipe.dm.blocks[b].attn.wv.register_forward_hook(lambda mod, inp, o, b=b: None if b in vn else vn.__setitem__(b, float(o[0, sp[0]:sp[-1] + 1].float().norm(dim=-1).mean()))) for b in swap_blocks]
            pipe.sample(model, text, SEEDS[0], E=E, size=(1216, 832))
            for h in hv:
                h.remove()
            g = torch.Generator().manual_seed(1)
            st["vtag"] = {b: vscale * vn[b] * torch.nn.functional.normalize(torch.randn(lab.kvheads * 128, generator=g), dim=0) for b in swap_blocks}
        for seed in SEEDS:
            if (bkey, seed) not in boxes:  # boxes from an unmasked splice render
                _, _, lat, _ = pipe.sample(model, text, seed, E=E, size=(1216, 832))
                boxes[(bkey, seed)] = top_boxes(met.app.get(np.asarray(to_img(vdec(pipe, lat)))[:, :, ::-1]))
            fb = boxes[(bkey, seed)]
            for cond in PROBE_CONDS:
                if (job, cond, seed) in done:
                    continue
                cap.update(mask=phrase_mask(cond, T, sp, fb, pipe.dev), T=T, sp=sp, rec={}, att={},
                           faces=[torch.tensor(face_tokens(b, grow=0.0), device=pipe.dev) for b in fb])
                with torch.no_grad():
                    _, _, lat, _ = pipe.sample(model, text, seed, E=E, size=(1216, 832))
                rec, at = cap["rec"], cap["att"]
                cap.update(mask=None, faces=None)
                im = to_img(vdec(pipe, lat))
                im.save(OUT / f"probe_{job}_{cond}_{seed}.jpg", quality=90)
                info = score_boxes(met.app.get(np.asarray(im)[:, :, ::-1]), fb, target, "id")
                for i, d in enumerate(info):
                    d["attn"] = float(np.mean(at[i])) if at.get(i) else None
                out.append({"job": job, "cond": cond, "seed": seed, "faces": info, "rec": rec})
                torch.save(out, f_pt)
                print(job, cond, seed, [(d["face"], None if d["id"] is None else round(d["id"], 3), None if d["attn"] is None else round(d["attn"], 3)) for d in info], flush=True)
        st["span"] = st["swap"] = st["target"] = st["vtag"] = None


def probe_analyze():
    """LOSO read-vs-not probe per (block, step): samples = faces in only_face0/only_face1 (reader = 1, the other face = 0) and txt+img (0).
    Direction = mean(read) - mean(not) on the training seeds; paired accuracy = share of held-out only_face renders where the reader
    projects above the non-reader; Fisher ratio of the 1-D projection. Controls: labels shuffled within each render, random direction.
    Features: face mean minus image mean ("face-whole") and the raw face mean."""
    import torch
    rows = torch.load(OUT / "probe_records.pt", weights_only=False)
    rng = np.random.default_rng(0)
    rep = {}
    for job in dict.fromkeys(r["job"] for r in rows):
        R = [r for r in rows if r["job"] == job and r["cond"] != "none"]
        for feat in ("face-whole", "face"):
            for b in PROBE_BLOCKS:
                for s in PROBE_STEPS:
                    X, y, seed, pair = [], [], [], []
                    for r in R:
                        fa, wh = r["rec"][(b, s)]
                        for i in range(len(fa)):
                            X.append((fa[i] - wh if feat == "face-whole" else fa[i]).numpy())
                            y.append(int(r["cond"] == f"only_face{i}"))
                            seed.append(r["seed"])
                            pair.append((r["cond"], r["seed"]) if r["cond"].startswith("only_face") else None)
                    X, y, seed = np.array(X), np.array(y), np.array(seed)

                    def evaluate(labels, direction=None):
                        hits, tot, proj_all, lab_all = 0, 0, [], []
                        for sd in SEEDS:
                            tr, te = seed != sd, seed == sd
                            d = X[tr & (labels == 1)].mean(0) - X[tr & (labels == 0)].mean(0) if direction is None else direction
                            p = X[te] @ d / (np.linalg.norm(d) + 1e-9)
                            proj_all += list(p)
                            lab_all += list(labels[te])
                            idx = np.where(te)[0]
                            for key in {pair[j] for j in idx if pair[j] is not None}:
                                k = [j for j in idx if pair[j] == key]
                                rd = [j for j in k if labels[j] == 1]
                                nr = [j for j in k if labels[j] == 0]
                                if rd and nr:
                                    hits += float(X[rd[0]] @ d > X[nr[0]] @ d)
                                    tot += 1
                        p, l = np.array(proj_all), np.array(lab_all)
                        fisher = (p[l == 1].mean() - p[l == 0].mean()) ** 2 / (p[l == 1].var() + p[l == 0].var() + 1e-9)
                        return hits / max(tot, 1), float(fisher)
                    acc, fis = evaluate(y)
                    shuf = []
                    for _ in range(20):
                        lab = y.copy()
                        for key in {p_ for p_ in pair if p_ is not None}:
                            k = [j for j in range(len(y)) if pair[j] == key]
                            if rng.random() < 0.5:
                                lab[k] = 1 - lab[k]
                        shuf.append(evaluate(lab))
                    rnd = [evaluate(y, rng.standard_normal(X.shape[1])) for _ in range(20)]
                    rep[f"{job} | {feat} | b{b} s{s}"] = {"paired_acc": round(acc, 3), "fisher": round(fis, 3),
                                                         "shuffled_acc": round(float(np.mean([a for a, _ in shuf])), 3), "shuffled_fisher": round(float(np.mean([f for _, f in shuf])), 3),
                                                         "random_acc": round(float(np.mean([a for a, _ in rnd])), 3), "random_fisher": round(float(np.mean([f for _, f in rnd])), 3)}
    common.write(OUT / "probe_summary.json", rep)
    for k, v in rep.items():
        print(f"{k:28s} acc {v['paired_acc']:.2f} F {v['fisher']:.2f} | shuffled {v['shuffled_acc']:.2f} F {v['shuffled_fisher']:.2f} | random {v['random_acc']:.2f} F {v['random_fisher']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["targets", "run", "summary", "splice", "splice-summary", "mask", "mask-pugh", "attrib", "attrib-summary", "switch-dose", "probe", "probe2", "probe-analyze", "targets-inslot", "targets-tok", "probe3"])
    {"targets": targets, "run": run, "summary": summary, "splice": splice, "splice-summary": lambda: summary("splice_", SPLICE), "mask": mask, "mask-pugh": lambda: mask("pugh"), "attrib": attrib, "attrib-summary": attrib_summary, "switch-dose": switch_dose, "probe": probe, "probe2": lambda: probe(("rlsa_inslot7_11", "rlsa_inslot_all", "rlsa_inslot_all_vtag")), "probe-analyze": probe_analyze, "targets-inslot": targets_inslot, "targets-tok": targets_tok, "probe3": lambda: probe(("rlsa_tok_all", "rlsa_inslot_all_vtag"))}[ap.parse_args().command]()
