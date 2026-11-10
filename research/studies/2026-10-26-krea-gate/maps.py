#!/usr/bin/env python3
"""(1) Name-reading maps for the talk (numbers only): per image patch, attention mass on a name's tokens averaged over the
top-10 binding heads (blocks 7–11), every sampling step — Base with two celebrities (one map per name), and the chair
sentence ("rlsa … a different woman behind her") under repair / bound (key lock) / no lock / no lock + pre-bind. Face
boxes from the detector are kept as numbers (which face is hers = highest likeness); no pixels leave runs/.
(2) Dose sweep of the best pre-bind offset (×0.5 / 0.75 / 1.25 / 1.5) on dev prompts × 4 seeds.

  python maps.py maps    -> runs/2026-10-26-krea-gate/maps.npz + maps.json
  python maps.py dose    -> runs/2026-10-26-krea-gate/dose.json (+ SUMMARY lines)
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prebind as PB  # noqa: E402

G, R, GT, OUT = PB.G, PB.R, PB.GT, PB.OUT
PAIR = "A medium photograph of Meryl Streep standing next to Florence Pugh on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography."
CHAIR = PB.FIT[0].replace("Natural indoor light.", "Natural indoor light.")
SEED, SIZE = 181909, (1216, 832)


def phrase_span(pipe, text, phrase):
    n = lambda t: pipe.encode(t)[0][0][0].shape[1]
    i, T = text.index(phrase), n(text)
    tail = n(text[i + len(phrase):]) if text[i + len(phrase):].strip() else 0
    return list(range(T - n(text[i:]), T - tail)), T


class Reader:
    """Pre-hooks on the binding blocks: per step, image-token attention mass on each registered span (top-10 heads)."""

    def __init__(self, pipe):
        from comfy.ldm.krea2.model import apply_rope
        self.pipe, self.spans, self.acc, self.T, self.on = pipe, {}, {}, 0, False
        heads = {}
        for b, h in GT.TOP:
            heads.setdefault(b, []).append(h)

        def pre(b):
            def f(mod, args):
                if not self.on:
                    return None
                import torch
                with torch.no_grad():
                    x, freqs = args[0], args[1]
                    q = mod.wq(x).unflatten(-1, (mod.heads, mod.headdim)).transpose(1, 2)
                    k = mod.wk(x).unflatten(-1, (mod.kvheads, mod.headdim)).transpose(1, 2)
                    q, k = mod.qknorm(q, k)
                    q, k = apply_rope(q, k, freqs)
                    k = k.repeat_interleave(mod.heads // mod.kvheads, dim=1)[0][heads[b]].float()
                    lg = q[0][heads[b]][:, self.T:].float() @ k.transpose(1, 2) / mod.headdim ** 0.5
                    lse = lg.logsumexp(-1)
                    for name, sp in self.spans.items():
                        m = (lg[..., sp[0]:sp[-1] + 1].logsumexp(-1) - lse).exp().sum(0).cpu().numpy()  # summed over this block's top heads
                        a = self.acc.setdefault(name, {}).setdefault(self.pipe.step, [np.zeros_like(m), 0])
                        a[0] += m
                        a[1] += len(heads[b])
                return None
            return f
        for b in heads:
            pipe.dm.blocks[b].attn.register_forward_pre_hook(pre(b))

    def run(self, model, text, spans, E=None):
        self.spans, self.acc, self.on = spans, {}, True
        self.T = self.pipe.encode(text)[0][0][0].shape[1]
        _, _, lat, _ = self.pipe.sample(model, text, SEED, E=E, size=SIZE)
        self.on = False
        gw, gh = SIZE[0] // 16, SIZE[1] // 16
        maps = {n: np.stack([v[s][0] / max(v[s][1], 1) for s in sorted(v)]).reshape(-1, gh, gw) for n, v in self.acc.items()}
        return maps, G.to_img(G.vdec(self.pipe, lat))


def maps():
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    rd = Reader(pipe)
    delta = torch.load(OUT / "prebind.pt", weights_only=False)["delta"].to(pipe.dev)
    load = lambda f: comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(f)), 1.0, 0)[0]
    repair = pipe.base
    for path, sc in G.HL_ARMS["repair"]:
        repair, _ = comfy.sd.load_lora_for_models(repair, None, comfy.utils.load_torch_file(str(path)), sc, 0)
    sp_r, _ = phrase_span(pipe, CHAIR, "rlsa")
    sp_o, _ = phrase_span(pipe, CHAIR, "a different woman")
    runs = {"base_pair": (pipe.base, PAIR, {"Meryl Streep": phrase_span(pipe, PAIR, "Meryl Streep")[0], "Florence Pugh": phrase_span(pipe, PAIR, "Florence Pugh")[0]}, None),
            "repair_chair": (repair, CHAIR, {"rlsa": sp_r, "a different woman": sp_o}, None),
            "bound_chair": (load(GT.BOUND), CHAIR, {"rlsa": sp_r, "a different woman": sp_o}, None),
            "nolock_chair": (load(PB.NOLOCK), CHAIR, {"rlsa": sp_r, "a different woman": sp_o}, None),
            "prebind_chair": (load(PB.NOLOCK), CHAIR, {"rlsa": sp_r, "a different woman": sp_o}, PB.offset_fn(delta, sp_r))}
    arrays, meta = {}, {"seed": SEED, "size": SIZE, "grid": [SIZE[1] // 16, SIZE[0] // 16], "steps": 8, "heads": GT.TOP, "texts": {"pair": PAIR, "chair": CHAIR}, "runs": {}}
    for name, (model, text, spans, E) in runs.items():
        mp, im = rd.run(model, text, spans, E)
        im.save(OUT / f"maps_{name}.jpg", quality=88)  # reference render, stays in the local output directory
        fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: g.bbox[0])
        faces = [{"bbox": [round(v) for v in g.bbox.tolist()], "her": round(float(g.normed_embedding @ her), 3)} for g in fs]
        for n, a in mp.items():
            arrays[f"{name}|{n}"] = a.astype(np.float32)
        meta["runs"][name] = {"spans": list(spans), "faces": faces, "peak": {n: round(float(a.max()), 4) for n, a in mp.items()}}
        print("MAP", name, meta["runs"][name], flush=True)
    np.savez_compressed(OUT / "maps.npz", **arrays)
    (OUT / "maps.json").write_text(json.dumps(meta, indent=1))
    print("MAPS_DONE", flush=True)


def maps_first():
    """Celebrity-first diagnostic: where does rlsa land when a celebrity is named first (lock and pre-bind both fail there)?"""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    concepts = {n: np.array(c) for n, c in json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"].items()}
    rd = Reader(pipe)
    delta = torch.load(OUT / "prebind.pt", weights_only=False)["delta"].to(pipe.dev)
    load = lambda f: comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(f)), 1.0, 0)[0]
    arrays, meta = {}, {"runs": {}}
    for case, who in (("theron_first", "Charlize Theron"), ("hathaway", "Anne Hathaway")):
        text = R.V5_MULTI[case]
        sp_r, _ = phrase_span(pipe, text, "rlsa")
        spans = {"rlsa": sp_r, who: phrase_span(pipe, text, who)[0]}
        for arm, model, E in (("bound", load(GT.BOUND), None), ("prebind", load(PB.NOLOCK), PB.offset_fn(delta, sp_r))):
            mp, im = rd.run(model, text, spans, E)
            im.save(OUT / f"mapsfirst_{case}_{arm}.jpg", quality=88)  # local only
            fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: g.bbox[0])
            faces = [{"bbox": [round(v) for v in g.bbox.tolist()], "her": round(float(g.normed_embedding @ her), 3), "named": round(float(g.normed_embedding @ concepts[who]), 3)} for g in fs]
            gw = SIZE[0] // 16
            per = {n: [round(float(a[5:].mean(0)[int(f["bbox"][1] // 16):int(f["bbox"][3] // 16) + 1, int(f["bbox"][0] // 16):int(f["bbox"][2] // 16) + 1].mean()), 4) for f in faces] for n, a in mp.items()}
            for n, a in mp.items():
                arrays[f"{case}|{arm}|{n}"] = a.astype(np.float32)
            meta["runs"][f"{case}|{arm}"] = {"faces": faces, "late_attention_per_face": per}
            print("FIRSTMAP", case, arm, meta["runs"][f"{case}|{arm}"], flush=True)
    np.savez_compressed(OUT / "maps_first.npz", **arrays)
    (OUT / "maps_first.json").write_text(json.dumps(meta, indent=1))
    print("MAPS_FIRST_DONE", flush=True)


DOSES = (0.5, 0.75, 1.25, 1.5)
DOSE_CASES = ("bench", "cafe", "crew", "pugh", "noir")


def dose():
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    concepts = {n: np.array(c) for n, c in json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"].items()}
    delta = torch.load(OUT / "prebind.pt", weights_only=False)["delta"].to(pipe.dev)
    model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(PB.NOLOCK)), 1.0, 0)[0]
    cases = {c: R.HELD[c] for c in DOSE_CASES} | {"chastain": (R.V5_MULTI["chastain"], 2)}
    f_json = OUT / "dose.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["dose"], r["case"], r["seed"]) for r in out}
    for d in DOSES:
        for case, (text, k) in cases.items():
            sp, _ = PB.span(pipe, text)
            for s in R.HELD_SEEDS:
                if (d, case, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, E=PB.offset_fn(delta * d, sp), size=(1216, 832) if k > 1 else (832, 1216))
                im = G.to_img(G.vdec(pipe, lat))
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                row = {"dose": d, "case": case, "seed": s, "her": [round(float(g.normed_embedding @ her), 3) for g in fs]}
                if case == "chastain":
                    row["named"] = [round(float(g.normed_embedding @ concepts["Jessica Chastain"]), 3) for g in fs]
                out.append(row)
                G.common.write(f_json, out)
                print("dose", d, case, s, row["her"], row.get("named", ""), flush=True)
    for d in DOSES:
        for case in cases:
            rs = [r for r in out if r["dose"] == d and r["case"] == case and r["her"]]
            hers, others = [], []
            for r in rs:
                h = list(r["her"])
                if "named" in r and len(h) == 2:
                    hers.append(h[1 - int(np.argmax(r["named"]))])
                else:
                    h = sorted(h, reverse=True)
                    hers.append(h[0])
                    others += h[1:]
            print("SUMMARY dose", d, case, {"her": round(float(np.mean(hers)), 3), "others": round(float(np.mean(others)), 3) if others else None,
                                            "ge035": f"{sum(v >= .35 for v in others)}/{len(others)}"}, flush=True)


if __name__ == "__main__":
    {"maps": maps, "dose": dose, "maps-first": maps_first}[sys.argv[1]]()
