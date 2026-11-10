#!/usr/bin/env python3
"""Does Base treat a trigger candidate like a name? (2026-10-25)

Two-person scene on Base (no adapter), fresh seeds. Per face: attention in the top-10 binding heads (late steps) to the
candidate's whole span and to its last token (names bind through their last token, keyswap targets-tok). A name binds
locally: one face reads it (0.6+), the other does not. Known names are the positive controls, "rlsa woman" the
current trigger, "a woman" the class baseline.
  CUDA_VISIBLE_DEVICES=0 python probe.py   -> runs/2026-10-25-trigger-probe/probe.json (+ renders)
  python probe.py prebind                  closed-form wk key lock fitted on Base alone for "Rlsa Porol" (its last token gets a
                                           name's last-token key, rekey.py keylock maths), then the probe again with the lock on
"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-22-rekey"))
import rekey as R  # noqa: E402

K, G = R.K, R.G
OUT = G.DS / "runs/2026-10-25-trigger-probe"
SEEDS = (190001, 190002, 190003, 190004)
SCENE = "A medium photograph of {x} standing next to a different woman at a charity gala, both" + R.VIS + " Evening event photography."
CANDIDATES = {"rlsa_woman": "rlsa woman", "rlsenwoman": "rlsenwoman", "actress_rlsa_porol": "the actress Rlsa Porol",
              "rlsa_porol": "Rlsa Porol", "actress_rlsa_porol_woman": "the actress Rlsa Porol, a woman,",
              "a_woman": "a woman", "pugh": "Florence Pugh", "winslet": "Kate Winslet"}


NAME = "Rlsa Porol"
LOCK_SEEDS = (182501, 182502)


def fit_lock(lab, model, lam=0.1):
    """dW per block so the name's last token presents the mean celebrity last-token key (targets_tok.pt), fitted on Base with the
    v2 capture prompts (rlsa -> NAME); C = Gram of every other token's wk input (side effects held down)."""
    import torch
    pipe = lab.pipe
    key = torch.load(K.OUT / "targets_tok.pt", weights_only=False)["name"]
    kv, hd = lab.kvheads, pipe.dm.blocks[0].attn.headdim
    jobs = [(t.replace("rlsa", NAME), (1216, 832) if n in R.V2_MULTI else (832, 1216)) for n, t in list(R.V2_MULTI.items()) + list(R.V2_SOLO.items())]
    jobs = [(t, size, [lab.span(t, NAME)[-1]]) for t, size in jobs]
    lock = {}
    for p0 in range(0, len(pipe.dm.blocks), R.LOCK_PASS):
        blocks = list(range(p0, min(len(pipe.dm.blocks), p0 + R.LOCK_PASS)))
        st = {b: {"D": [], "Y": [], "step": [], "C": torch.zeros(6144, 6144, device=pipe.dev), "nc": 0} for b in blocks}
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
                st[b]["C"].addmm_(x[keep].T, x[keep])
                st[b]["nc"] += int(keep.sum())
            return f
        hs = [h for b in blocks for h in (pipe.dm.blocks[b].attn.wk.register_forward_pre_hook(pre(b)), pipe.dm.blocks[b].attn.wk.register_forward_hook(post(b)))]
        for text, size, sp in jobs:
            cur["sp"] = sp
            for s in LOCK_SEEDS:
                pipe.sample(model, text, s, size=size)
        for h in hs:
            h.remove()
        for b in blocks:
            D, Y0 = torch.cat(st[b]["D"]).double(), torch.cat(st[b]["Y"]).double()
            steps = torch.tensor(st[b]["step"])
            C = (st[b]["C"] / st[b]["nc"]).double().cpu()
            st[b]["C"] = None
            w = pipe.dm.blocks[b].attn.qknorm.knorm.scale.detach().float().cpu().double() + 1.0
            tgt = key[b].double()[:, steps.clamp(max=G.STEPS - 1)].permute(1, 0, 2)
            Kstar = (torch.nn.functional.normalize(tgt / w, dim=-1) * Y0.view(-1, kv, hd).norm(dim=-1, keepdim=True)).reshape(len(D), kv * hd)
            DD = D.T @ D
            M = DD + lam * float(DD.diagonal().sum()) / float(C.diagonal().sum()) * C
            M += 1e-6 * float(M.diagonal().sum()) / M.shape[0] * torch.eye(M.shape[0], dtype=M.dtype)
            dW = torch.linalg.solve(M, D.T @ (Kstar - Y0)).T
            cos = lambda Y: float(torch.nn.functional.cosine_similarity(Y.view(-1, kv, hd) / w, tgt, dim=-1).mean())
            print(f"block {b}: name last-token key cos to celebrity key {cos(Y0):.3f} -> {cos(Y0 + D @ dW.T):.3f}  side-effect energy {float(torch.trace(dW @ C @ dW.T)):.3g}", flush=True)
            lock[b] = dW.half()
        torch.cuda.empty_cache()
    return lock


def main(prebind=False, blocks=None):
    lab = K.Lab()
    pipe = lab.pipe
    sys.path.insert(0, str(G.probes.OLD))
    import score_and_select
    app = score_and_select.setup()[0]  # face detector only (CPU)
    rig = SimpleNamespace(pipe=pipe, lab=lab, model=lab.models["base"], n=lambda t: pipe.encode(t)[0][0][0].shape[1])
    OUT.mkdir(parents=True, exist_ok=True)
    out, tag, cands = [], "", CANDIDATES
    if prebind:
        import torch
        f = OUT / "prebind_lock.pt"
        lock = torch.load(f) if f.exists() else {b: dW.to(pipe.dev, torch.bfloat16) for b, dW in fit_lock(lab, rig.model).items()}
        torch.save(lock, f)
        lock = {b: dW.to(pipe.dev, torch.bfloat16) for b, dW in lock.items() if blocks is None or b in blocks}  # per-block fits are independent
        for b, dW in lock.items():
            pipe.dm.blocks[b].attn.wk.register_forward_hook(lambda mod, inp, o, dW=dW: o + (inp[0].to(dW.dtype) @ dW.T).to(o.dtype))
        tag = "prebind_" if blocks is None else f"prebind{max(blocks) + 1}_"
        cands = {k: v for k, v in CANDIDATES.items() if k in ("rlsa_porol", "actress_rlsa_porol", "actress_rlsa_porol_woman", "a_woman", "pugh")}
    for name, x in cands.items():
        text = SCENE.format(x=x)
        sp = lab.span(text, x.rstrip(","))
        for s in SEEDS:
            im = G.to_img(G.vdec(pipe, pipe.sample(rig.model, text, s, size=(1216, 832))[2]))
            im.save(OUT / f"{tag}{name}_{s}.jpg", quality=90)
            boxes = K.top_boxes(app.get(np.asarray(im)[:, :, ::-1]), 2)
            if len(boxes) < 2:
                continue
            span = R.read_attention(rig, text, s, (1216, 832), sp, boxes)
            last = R.read_attention(rig, text, s, (1216, 832), [sp[-1]], boxes)
            out.append({"cand": name, "seed": s, "span_attn": span, "last_attn": last, "boxes": boxes})
            G.common.write(OUT / f"{tag}probe.json", out)
            print(name, s, "span", [round(v, 3) for v in span], "last", [round(v, 3) for v in last], flush=True)
    for name in cands:
        rs = [r for r in out if r["cand"] == name]
        if rs:
            hi = [max(r["last_attn"]) for r in rs]
            lo = [min(r["last_attn"]) for r in rs]
            sh = [max(r["span_attn"]) for r in rs]
            print(f"{name:26s} last token: reading face {np.mean(hi):.3f} other {np.mean(lo):.3f} | whole span reading face {np.mean(sh):.3f} (n={len(rs)})", flush=True)


if __name__ == "__main__":
    main(prebind=sys.argv[1:2] == ["prebind"], blocks=range(int(sys.argv[2])) if len(sys.argv) > 2 else None)  # prebind 16 = blocks 0-15 only
