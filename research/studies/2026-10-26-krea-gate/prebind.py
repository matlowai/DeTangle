#!/usr/bin/env python3
"""Embedding pre-bind on Krea 2 (MOSS lesson; Subject A, adapter not distributed): replace the bound file's wk key lock by an offset on
rlsa's own conditioning rows (txtmlp.3 output, the pipeline's additive hook), fitted so rlsa's post-qknorm key in blocks 7–11
matches the same name targets the lock used (keyswap/targets_tok.pt: per block, kv group, step). Prompts without rlsa have no
such rows, so they are untouched by construction; other tokens' keys are not edited (the lock's wk edit applies to every
token) — the hypothesis is that this helps celebrity-first sentences, where the lock fails.

  python prebind.py fit    nolock export + trajectories of FIT prompts (1 seed) -> Adam on the offset -> runs/.../prebind.pt
  python prebind.py eval   arms bound (lock) / nolock / nolock+prebind × rekey HELD dev prompts + NAMED_ORDER (both phrasings) × 4 seeds
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-20-grad-components"))
sys.path.insert(0, str(HERE.parent / "2026-10-22-rekey"))
import grad_amp as G  # noqa: E402
import rekey as R  # noqa: E402
import gate as GT  # noqa: E402

OUT = GT.OUT
NOLOCK = R.OUT / "rlsa_bound_A3all_mu1_lam0.1_trigtok_nolock.safetensors"
FIT_BLOCKS = range(7, 12)
VIS = R.VIS
FIT = ["A medium photograph of rlsa sitting on a chair while a different woman stands behind her, both" + VIS + " Natural indoor light.",
       "A medium photograph of rlsa sitting at a cafe table between two other women, all three" + VIS + " Natural daylight.",
       "A medium photograph of rlsa standing between two men at a train station, all three" + VIS + " Overcast daylight.",
       "A medium photograph of rlsa walking a dog in a park next to an older woman, both" + VIS + " Golden hour light.",
       "A close-up head-and-shoulders studio photograph of rlsa, facing the camera. Soft even studio light and a plain grey background.",
       "A medium photograph of rlsa standing on a red carpet, facing the camera with her face clearly visible and in focus. Evening event photography.",
       "A candid photograph of rlsa reading in a library, facing the camera, her face clearly visible. Warm lamp light.",
       "A cinematic film still of rlsa on the deck of a sailing ship, facing the camera, her face clearly visible."]


HELD_FIRST = {"johansson_first": ("A medium photograph of Scarlett Johansson standing next to rlsa at an art gallery opening, both" + VIS + " Soft gallery light.", "Scarlett Johansson"),
              "weaver_first": ("A medium photograph of Sigourney Weaver walking beside rlsa on a city street, both" + VIS + " Overcast daylight.", "Sigourney Weaver")}


def span(pipe, text):
    n = lambda t: pipe.encode(t)[0][0][0].shape[1]
    i, T = text.index("rlsa"), n(text)
    tail = n(text[i + 4:]) if text[i + 4:].strip() else 0
    return list(range(T - n(text[i:]), T - tail)), T


def offset_fn(delta, sp):
    """E for the txtmlp.3 hook: delta on rlsa's rows, zero elsewhere (differentiable in delta)."""
    def E(h, step):
        e = h.new_zeros(h.shape[-2], delta.shape[-1], dtype=delta.dtype)
        return e.index_add(0, delta.new_tensor(sp, dtype=__import__("torch").long), delta).unsqueeze(0)
    return E


def fit(iters=200, lr=0.05, blocks=FIT_BLOCKS, tag="prebind", model_file=None):
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(model_file or NOLOCK)), 1.0, 0)[0]
    tgt = torch.load(G.RUN / "keyswap/targets_tok.pt", weights_only=False)["name"]
    cap = {}
    for b in blocks:
        pipe.dm.blocks[b].attn.qknorm.knorm.register_forward_hook(lambda mod, inp, out, b=b: cap.__setitem__(b, out) if cap.get("on") else None)

    class Stop(Exception):
        pass

    def stop(mod, inp, out):
        if cap.get("on"):
            raise Stop
    pipe.dm.blocks[max(blocks)].register_forward_hook(stop)
    t0 = time.time()
    data = []
    for i, text in enumerate(FIT):
        sp, T = span(pipe, text)
        size = (1216, 832) if "two" in text or "both" in text or "three" in text else (832, 1216)
        xs, _, _, cond = pipe.sample(model, text, 182501 + i, size=size)
        data.append({"text": text, "sp": sp, "cond": cond, "xs": [x.cpu() for x in xs]})
    lens = {len(d["sp"]) for d in data}
    assert len(lens) == 1, f"rlsa tokenises differently across FIT prompts: {lens}"
    print("trajectories", round(time.time() - t0), "s; rlsa tokens", lens, flush=True)
    delta = torch.zeros(lens.pop(), pipe.dm.txtmlp[3].out_features, device=pipe.dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    G.checkpoint_blocks(pipe.dm, True)
    pipe.mm.load_models_gpu([model])
    log = []
    for it in range(iters):
        d = data[it % len(data)]
        step = int(np.random.randint(0, len(pipe.sigmas) - 1))
        pipe.step, pipe.E = step, offset_fn(delta, d["sp"])
        cap.clear()
        cap["on"] = True
        try:
            pipe.denoise(d["xs"][step].to(pipe.dev), pipe.sigmas[step], d["cond"])
        except Stop:
            pass
        cap["on"] = False
        loss = 0
        for b in blocks:
            k = cap[b][0, :, d["sp"]].float()  # kvH x span x D
            t = tgt[b][:, step].to(pipe.dev).float().unsqueeze(1)  # kvH x 1 x D
            loss = loss + (1 - torch.nn.functional.cosine_similarity(k, t, dim=-1)).mean()
        loss = loss / len(blocks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pipe.E = None
        log.append(round(float(loss), 4))
        if it % 20 == 0 or it == iters - 1:
            print(f"it {it} key-cos loss {np.mean(log[-20:]):.4f} |delta| {float(delta.norm()):.1f} {time.time() - t0:.0f}s", flush=True)
    G.checkpoint_blocks(pipe.dm, False)
    torch.save({"delta": delta.detach().cpu(), "log": log, "fit": FIT, "blocks": list(blocks), "iters": iters, "lr": lr,
                "sec": round(time.time() - t0)}, OUT / f"{tag}.pt")
    print("PREBIND_FIT_DONE", OUT / f"{tag}.pt", flush=True)


def evaluate(tags=("prebind",), model_file=None, seeds=None, suffix=""):
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    concepts = {n: np.array(c) for n, c in json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"].items()}
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    concepts["Florence Pugh"] = pugh / np.linalg.norm(pugh)
    deltas = {t: torch.load(OUT / f"{t}.pt", weights_only=False)["delta"].to(pipe.dev) for t in tags}
    models = {"bound": GT.BOUND, "nolock": NOLOCK, **{t: (model_file or NOLOCK) for t in tags}}
    models = {a: comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(f)), 1.0, 0)[0] for a, f in models.items()}
    cases = {c: (t, k, "Florence Pugh" if c == "pugh" else None) for c, (t, k) in R.HELD.items() if "rlsa" in t}
    for name, who in R.NAMED_ORDER.items():
        cases[name] = (R.V5_MULTI[name], 2, who)
        cases[name + "_woman"] = (R.V5_MULTI[name].replace("rlsa", "rlsa woman", 1), 2, who)
    for name, (text, who) in HELD_FIRST.items():  # held-out celebrity-first prompts (never in any capture)
        cases[name] = (text, 2, who)
    f_json = OUT / f"prebind_eval{suffix}.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    for arm, model in models.items():
        for case, (text, k, who) in cases.items():
            size = (1216, 832) if k > 1 else (832, 1216)
            sp, _ = span(pipe, text)
            delta = deltas.get(arm)
            if delta is not None and len(sp) != delta.shape[0]:
                print("skip (rlsa tokenises differently)", case, flush=True)
                continue
            for s in seeds or R.HELD_SEEDS:
                if (arm, case, s) in done:
                    continue
                _, _, lat, _ = pipe.sample(model, text, s, E=offset_fn(delta, sp) if delta is not None else None, size=size)
                im = G.to_img(G.vdec(pipe, lat))
                im.save(OUT / f"prebind_{arm}_{case}_{s}.jpg", quality=90)  # local only
                fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
                row = {"arm": arm, "case": case, "seed": s, "her": [round(float(g.normed_embedding @ her), 3) for g in fs]}
                if who:
                    row["named"] = [round(float(g.normed_embedding @ concepts[who]), 3) for g in fs]
                out.append(row)
                G.common.write(f_json, out)
                print(arm, case, s, row["her"], row.get("named", ""), flush=True)
    summarize(out, suffix)


def summarize(out=None, suffix=""):
    out = out or json.loads((OUT / f"prebind_eval{suffix}.json").read_text())
    res = {}
    for arm in sorted({r["arm"] for r in out}):
        for case in sorted({r["case"] for r in out}):
            rs = [r for r in out if r["arm"] == arm and r["case"] == case and r["her"]]
            if not rs:
                continue
            hers, others, named, her_on_named = [], [], [], []
            for r in rs:
                h = list(r["her"])
                if "named" in r and len(h) == 2:
                    j = int(np.argmax(r["named"]))
                    named.append(r["named"][j])
                    her_on_named.append(h[j])
                    hers.append(h[1 - j])
                else:
                    h = sorted(h, reverse=True)
                    hers.append(h[0])
                    others += h[1:]
            res[f"{arm}/{case}"] = {"her": round(float(np.mean(hers)), 3), **({"others": round(float(np.mean(others)), 3), "ge035": f"{sum(v >= .35 for v in others)}/{len(others)}"} if others else {}),
                                     **({"named": round(float(np.mean(named)), 3), "her_on_named": round(float(np.mean(her_on_named)), 3)} if named else {}), "n": len(rs)}
            print("SUMMARY", arm, case, res[f"{arm}/{case}"], flush=True)
    (OUT / f"prebind_summary{suffix}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval", "summary"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--blocks", default="7-11")
    ap.add_argument("--tag", default="prebind")
    ap.add_argument("--tags", default="prebind")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    lo, hi = map(int, a.blocks.split("-"))
    {"fit": lambda: fit(a.iters, blocks=range(lo, hi + 1), tag=a.tag, model_file=a.model), "eval": lambda: evaluate(tuple(a.tags.split(",")), a.model, [int(x) for x in a.seeds.split(",")] if a.seeds else None, a.suffix), "summary": summarize}[a.cmd]()
