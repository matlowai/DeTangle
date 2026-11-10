#!/usr/bin/env python3
"""Name gate on the Krea 2 bound file (Subject A; adapter not distributed). MOSS lesson (from a shelved MOSS-Image study, not included):
scale the image-side block LoRA, per image patch, by how strongly that patch reads the trigger in the binding heads.

The bound file (rlsa_bound_A3all_mu1_lam0.1_trigtok) = text side (txtfusion LoRA + token-only amp x6 on txtmlp.3) + block
LoRA in all 28 blocks (re-keyed lora_A; wk pair carries the +320-rank key lock). Here the text side loads through the stock
loader; the block LoRA runs as forward hooks with a per-token multiplier:
  text tokens   1 when the prompt has rlsa (the lock gives rlsa its name key; the block LoRA needs words AND image, DeTangle
                "Two parts make her")
  image tokens  gate g in [0, 1]: mean attention mass of the patch on the rlsa span over the binding heads (top-10, blocks
                7-11) of this and earlier blocks, / its 99th percentile in the image, clamped, squared; blocks 0-6 use the
                previous sampling step's final gate (0 at step 0)
A prompt without rlsa renders through Base (prompt-level switch): exactly Base by construction — that part is trivial and is
NOT the gate's contribution; the gate is judged on triggered scenes (her likeness, strangers ≥ 0.35, named people).

  python gate.py parity   hooks without gate vs the stock-loaded file (px), 2 cases
  python gate.py dev      arms x rekey HELD dev prompts x HELD_SEEDS -> runs/2026-10-26-krea-gate/dev.json + summary
Env: CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=<ComfyUI-eval-2026-07-30>/pinned_site
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-20-grad-components"))
sys.path.insert(0, str(HERE.parent / "2026-10-22-rekey"))
import grad_amp as G  # noqa: E402
import rekey as R  # noqa: E402

OUT = G.DS / "runs/2026-10-26-krea-gate"
BOUND = R.OUT / "rlsa_bound_A3all_mu1_lam0.1_trigtok.safetensors"
TOP = [tuple(h) for h in [(h["block"], h["head"]) for h in json.loads((G.RUN / "bind" / "bind_analysis.json").read_text())["top_binding_heads"][:10]]]


class GatedBound:
    def __init__(self, pipe, gate=True, power=2.0):
        import torch
        import comfy.sd
        from safetensors.torch import load_file
        self.torch, self.pipe, self.gate, self.power = torch, pipe, gate, power
        sd = load_file(str(BOUND))
        self.model = comfy.sd.load_lora_for_models(pipe.base, None, {k: v for k, v in sd.items() if ".blocks." not in k}, 1.0, 0)[0]
        self.lora = {k.removeprefix("diffusion_model.").removesuffix(".lora_A.weight"): (v.to(pipe.dev, torch.bfloat16), sd[k.replace("lora_A", "lora_B")].to(pipe.dev, torch.bfloat16))
                     for k, v in sd.items() if ".blocks." in k and k.endswith("lora_A.weight")}
        self.heads = {}
        for b, h in TOP:
            self.heads.setdefault(b, []).append(h)
        self.sp, self.T, self.on = None, 0, False
        self.mass, self.g, self.g_prev, self.g_last = {}, {}, None, None
        self.hooks = [pipe.dm.get_submodule(p).register_forward_hook(self._lora(p)) for p in self.lora]
        self.hooks += [blk.attn.register_forward_pre_hook(self._attn_pre(b)) for b, blk in enumerate(pipe.dm.blocks)]
        self.hooks.append(pipe.dm.register_forward_pre_hook(lambda mod, args: self.mass.clear()))
        self.hooks.append(pipe.dm.register_forward_hook(lambda mod, args, out: setattr(self, "g_prev", self.g_last)))

    def _mult(self, b, n):
        """(n,) token multiplier for block b: text tokens 1, image tokens the gate (or 1 ungated)."""
        torch = self.torch
        m = torch.ones(n, device=self.pipe.dev)
        if self.gate:
            g = self.g.get(b)
            m[self.T:] = g if g is not None else 0.0
        return m

    def _lora(self, path):
        A, B = self.lora[path]
        b = int(path.split(".")[1])

        def f(mod, inp, out):
            if not self.on:
                return None
            d = (inp[0].to(A.dtype) @ A.T) @ B.T
            return out + (d * self._mult(b, d.shape[1])[None, :, None].to(d.dtype)).to(out.dtype)
        return f

    def _combine(self, b):
        src = [w for l, w in self.mass.items() if l <= b]
        if not src:
            return self.g_prev
        w = self.torch.stack(src).mean(0)
        return (w / w.quantile(0.99).clamp(min=1e-8)).clamp(max=1).pow(self.power)

    def _attn_pre(self, b):
        def f(mod, args):
            if not (self.on and self.gate):
                return None
            torch = self.torch
            from comfy.ldm.krea2.model import apply_rope
            self.g[b] = self._combine(b - 1)
            if b in self.heads:
                x, freqs = args[0], args[1]
                q = mod.wq(x).unflatten(-1, (mod.heads, mod.headdim)).transpose(1, 2)
                k = mod.wk(x).unflatten(-1, (mod.kvheads, mod.headdim)).transpose(1, 2)
                q, k = mod.qknorm(q, k)
                q, k = apply_rope(q, k, freqs)
                hs = self.heads[b]
                k = k.repeat_interleave(mod.heads // mod.kvheads, dim=1)[0][hs].float()
                lg = q[0][hs][:, self.T:].float() @ k.transpose(1, 2) / mod.headdim ** 0.5
                self.mass[b] = (lg[..., self.sp[0]:self.sp[-1] + 1].logsumexp(-1) - lg.logsumexp(-1)).exp().mean(0)
            self.g[b] = self._combine(b)
            if self.g[b] is not None:
                self.g_last = self.g[b]
            return None
        return f

    def render(self, text, seed, size):
        """Prompt without rlsa -> Base (switch); with rlsa -> text side + hooked block LoRA (gated or not)."""
        if "rlsa" in text:
            n = lambda t: self.pipe.encode(t)[0][0][0].shape[1]
            i, self.T = text.index("rlsa"), n(text)
            tail = n(text[i + 4:]) if text[i + 4:].strip() else 0
            self.sp = list(range(self.T - n(text[i:]), self.T - tail))  # as keyswap.Lab.span
            self.on, self.g_prev, self.g_last = True, None, None
            model = self.model
        else:
            self.on, model = False, self.pipe.base
        self.g.clear()
        _, _, lat, _ = self.pipe.sample(model, text, seed, size=size)
        self.on = False
        return G.to_img(G.vdec(self.pipe, lat))


def parity():
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    stock = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(BOUND)), 1.0, 0)[0]
    gb = GatedBound(pipe, gate=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for case in ("bench", "noir"):
        text, k = R.HELD[case]
        size = (1216, 832) if k > 1 else (832, 1216)
        b = np.asarray(gb.render(text, R.HELD_SEEDS[0], size)).astype(float)  # hooks first: a stock render bakes its patches into the shared weights
        _, _, lat, _ = pipe.sample(stock, text, R.HELD_SEEDS[0], size=size)
        a = np.asarray(G.to_img(G.vdec(pipe, lat))).astype(float)
        b2 = np.asarray(gb.render(text, R.HELD_SEEDS[0], size)).astype(float)
        print("PARITY", case, "hooks vs stock mean abs px", round(float(np.abs(a - b).mean()), 2), "| hooks after a stock render vs before", round(float(np.abs(b2 - b).mean()), 2), flush=True)


ARMS = {"stock": None, "hooks": dict(gate=False), "gate": dict(gate=True, power=2.0), "gate_p1": dict(gate=True, power=1.0)}


def dev():
    import torch
    pipe = G.Pipe([])
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    pugh /= np.linalg.norm(pugh)
    OUT.mkdir(parents=True, exist_ok=True)
    f_json = OUT / "dev.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    import comfy.sd, comfy.utils
    stock = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(BOUND)), 1.0, 0)[0]
    gb = GatedBound(pipe)
    for arm, kw in ARMS.items():
        if kw is not None:
            gb.gate, gb.power = kw["gate"], kw.get("power", 2.0)
        for case, (text, k) in R.HELD.items():
            if "rlsa" not in text and arm != "hooks":
                continue  # untriggered prompts render Base under every arm (switch)
            size = (1216, 832) if k > 1 else (832, 1216)
            for s in R.HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                if kw is None:  # the bound file through the stock loader (baked into the fp8 weights), as rekey.py validate
                    gb.g_last = None
                    im = G.to_img(G.vdec(pipe, pipe.sample(stock, text, s, size=size)[2]))
                else:
                    im = gb.render(text, s, size)
                im.save(OUT / f"{arm}_{case}_{s}.jpg", quality=90)  # renders stay in the local output directory
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                row = {"arm": arm, "case": case, "seed": s, "likeness": [round(float(g.normed_embedding @ her), 3) for g in fs],
                       "pugh": [round(float(g.normed_embedding @ pugh), 3) for g in fs]}
                if gb.g_last is not None and fs:
                    gw, gh = size[0] // 16, size[1] // 16
                    gm = gb.g_last.float().cpu().numpy()
                    row["gate_faces"] = [round(float(gm[R.K.face_tokens(g.bbox.tolist(), grow=0.0, GW=gw, GH=gh)].mean()), 3) for g in fs]
                    row["gate_mean"] = round(float(gm.mean()), 3)
                out.append(row)
                G.common.write(f_json, out)
                print(arm, case, s, row["likeness"], row.get("gate_faces", ""), flush=True)
    summary(out)


def summary(out=None):
    out = out or json.loads((OUT / "dev.json").read_text())
    res = {}
    for arm in ARMS:
        for case in R.HELD:
            rs = [r for r in out if r["arm"] == arm and r["case"] == case and r["likeness"]]
            if not rs:
                continue
            best, others, pg = [], [], []
            for r in rs:
                lk = list(r["likeness"])
                if case == "pugh":
                    i = int(np.argmax(r["pugh"]))
                    pg.append(r["pugh"][i])
                    lk.pop(i)
                lk = sorted(lk, reverse=True)
                if lk:
                    best.append(lk[0])
                    others += lk[1:]
            res[f"{arm}/{case}"] = {"her": round(float(np.mean(best)), 3), "others": round(float(np.mean(others)), 3) if others else None,
                                    "ge035": f"{sum(v >= .35 for v in others)}/{len(others)}", **({"pugh_by_name": round(float(np.mean(pg)), 3)} if pg else {})}
            print("SUMMARY", arm, case, res[f"{arm}/{case}"], flush=True)
    (OUT / "dev_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"parity": parity, "dev": dev, "summary": summary}[sys.argv[1]]()
