#!/usr/bin/env python3
"""Key targets for the Krea 2 embedding pre-bind (renders stay local; Subject A, adapter not distributed). DEV-set experiments.

The G6 pre-bind (studies/2026-10-26-krea-gate/prebind.py, tag pb_b0_27_200) fits an offset on rlsa's own txtmlp.3 rows so rlsa's
post-qknorm key matches the AVERAGED last-token key of 12 celebrity names (keyswap/targets_tok.pt) in every block and step. A longer
fit that matched that mean more closely bound worse (pb_b7_15_600). Targets tried here (FITS):
  knnK   mean of the K names' keys nearest to rlsa's current key (2026-11-01 run 1; knn1_nopugh = the same without Florence Pugh)
  mean   the original averaged key (size-matched control, |delta| capped)
  span   rlsa's own key projected onto span{12 name keys} (per block, step, kv group): be inside the names' region, not at a point
  hull   the same projected onto the names' convex hull (simplex weights), after rescaling rlsa's key to the names' norm

  python knn_prebind.py names                          per-name last-token keys (Base, chair sentence, seed 182401) -> name_keys.pt
  python knn_prebind.py fit --tags span_200,hull_200   one process, shared trajectories; each tag's settings come from FITS
  python knn_prebind.py eval --tags a,b [--seeds2]     nolock export + offset × dev cases of prebind.evaluate × seeds 181909-12
                                                       (--seeds2: fresh seeds 181913-16) -> eval{_seeds2}.json, prebind_summary{_seeds2}{--out-suffix}.json
  python knn_prebind.py guard --tags a,b [--seeds2]    identity-import guard: her faces vs the celebrities' reference faces -> guard{_seeds2}{--out-suffix}.json
"""
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-26-krea-gate"))
sys.path.insert(0, str(HERE.parent / "2026-10-20-grad-components"))
import prebind as P  # noqa: E402
import keyswap as K  # noqa: E402

G, R = P.G, P.R
OUT = G.DS / "runs/2026-11-01-knn-prebind"
OLD = P.OUT  # runs/2026-10-26-krea-gate: the baseline fit pb_b0_27_200.pt, its eval rows and renders
BASELINE = "pb_b0_27_200"
SEEDS2 = (181913, 181914, 181915, 181916)
FITS = {"knn1_200": {"mode": "knn1"},
        "knn3_200": {"mode": "knn3"},
        "mean_cap487_400": {"mode": "mean", "iters": 400, "cap": 487},
        "knn1_nopugh_200": {"mode": "knn1", "exclude": ("Florence Pugh",), "seed": 0},
        "span_200": {"mode": "span", "seed": 0},
        "hull_200": {"mode": "hull", "seed": 0}}


def names():
    """Per-name keys at the token targets_tok.pt chose (its pass 1, kept per name instead of averaged)."""
    import torch
    pipe = G.Pipe([])
    saved = torch.load(G.RUN / "keyswap/targets_tok.pt", weights_only=False)
    cap = {}
    for b, blk in enumerate(pipe.dm.blocks):
        blk.attn.qknorm.knorm.register_forward_hook(lambda mod, inp, out, b=b: cap.__setitem__((b, pipe.step), out[0, :, cap["sp"]].float().cpu()) if "sp" in cap else None)
    keys = {}
    for who in saved["names"]:
        text = K.PROMPTS["chair"][0].replace("rlsa", who)
        n = lambda t: pipe.encode(t)[0][0][0].shape[1]
        i, T = text.index(who), n(text)
        cap.clear()
        cap["sp"] = list(range(T - n(text[i:]), T - n(text[i + len(who):])))
        pipe.sample(pipe.base, text, 182401, size=(1216, 832))
        j = saved["best_token"][who]
        keys[who] = {b: torch.stack([cap[(b, s)][:, j] for s in range(G.STEPS)], 1) for b in range(len(pipe.dm.blocks))}  # kvH x step x D
        print(who, "captured", flush=True)
    NK = {b: torch.stack([keys[w][b] for w in saved["names"]]) for b in range(len(pipe.dm.blocks))}  # names x kvH x step x D
    err = max(float((NK[b].mean(0) - saved["name"][b]).abs().max()) for b in NK)
    print("max |mean of per-name keys - targets_tok| =", err, flush=True)
    torch.save({"keys": NK, "names": saved["names"], "check_max_abs_err": err}, OUT / "name_keys.pt")


def simplex(v):
    """Euclidean projection of the last dim onto the probability simplex (sort-based)."""
    import torch
    u = v.sort(-1, descending=True).values
    css = u.cumsum(-1) - 1
    ind = torch.arange(1, v.shape[-1] + 1, device=v.device, dtype=v.dtype)
    rho = (u - css / ind > 0).sum(-1, keepdim=True)
    return (v - css.gather(-1, rho - 1) / rho).clamp_min(0)


class Targets:
    """Per-step name geometry for all blocks at once. N: blocks x kvH x names x D (one step). target(mode, k) -> (t, info) with
    k: blocks x kvH x span x D (rlsa's current keys) and t broadcastable to k; info = names picked (knn) or name weights (span, hull)."""

    def __init__(self, NK, keep, step, rtol=3e-3):
        import torch
        self.torch = torch
        self.N = torch.stack([NK[b][keep][:, :, step] for b in sorted(NK)]).transpose(1, 2)  # B x kvH x n x D
        self.G = self.N @ self.N.transpose(-1, -2)  # B x kvH x n x n
        # Span basis from the SVD, dropping directions below rtol of the top singular value: in 5 kv groups of block 0 the 12 name
        # keys are one shared vector (norm 556, singular values 7e-5 of the top); everywhere else all 12 directions are >= 9e-3.
        U, S, Vh = torch.linalg.svd(self.N, full_matrices=False)
        on = (S > rtol * S[..., :1]).to(S.dtype)
        self.Q = Vh.transpose(-1, -2) * on[..., None, :]  # B x kvH x D x n (dropped directions are zero columns)
        self.pinv = (Vh.transpose(-1, -2) * (on / S.clamp_min(1e-30))[..., None, :]) @ U.transpose(-1, -2)  # B x kvH x D x n
        self.L = 2 * torch.linalg.eigvalsh(self.G)[..., -1]  # B x kvH, Lipschitz constant of the hull objective's gradient
        self.norm = self.N.norm(dim=-1).mean(-1)  # B x kvH, mean name-key norm

    def hull_weights(self, k, iters=150):
        """min_w |w N - k|^2 over the simplex (FISTA), per block, kv group and rlsa token; k is rescaled to the names' norm first."""
        torch = self.torch
        k = k * (self.norm[..., None, None] / k.norm(dim=-1, keepdim=True))
        b = k @ self.N.transpose(-1, -2)  # B x kvH x S x n
        w = y = torch.full_like(b, 1 / b.shape[-1])
        t, L = 1.0, self.L[..., None, None]
        for _ in range(iters):
            w_new = simplex(y - 2 * (y @ self.G - b) / L)
            t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
            y = w_new + (t - 1) / t_new * (w_new - w)
            w, t = w_new, t_new
        return w

    def target(self, mode, k):
        torch = self.torch
        with torch.no_grad():
            if mode == "mean":
                return self.N.mean(2).unsqueeze(2), None
            if mode.startswith("knn"):
                # kNN target instead of the 12-name centroid: the centroid is not where any real name's key sits, so pull rlsa
                # toward the mean of the k names nearest to where it already is.
                sim = torch.nn.functional.cosine_similarity(k.mean(2, keepdim=True), self.N, dim=-1).mean(1)  # B x n
                idx = sim.topk(int(mode[3:]), dim=-1).indices  # B x k
                t = torch.stack([self.N[i][:, ix].mean(1) for i, ix in enumerate(idx)]).unsqueeze(2)  # B x kvH x 1 x D
                return t, idx.tolist()
            if mode == "span":
                w = k @ self.pinv  # B x kvH x S x n, minimum-norm name coefficients of the projection
                return (k @ self.Q) @ self.Q.transpose(-1, -2), w.mean((1, 2)).tolist()
            if mode == "hull":
                w = self.hull_weights(k)
                return w @ self.N, w.mean((1, 2)).tolist()
        raise ValueError(mode)


def fit(tags, lr=0.05):
    """prebind.fit on all 28 blocks, one Adam fit per tag (settings in FITS), sharing one model load and the FIT trajectories.
    knn targets are re-picked every iteration from rlsa's CURRENT key (it moves far during the fit, 1 - cos 0.38 -> 0.15, so a pick
    from the untrained key would be arbitrary). span / hull: rlsa's key projected onto the names' span / convex hull, also every
    iteration, as a fixed (no-grad) target; for span this gives the same gradient as minimising 1 - |Pk| / |k| directly."""
    import torch
    import torch.nn.functional as F
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils  # on sys.path once Pipe has added the pinned checkout
    model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(P.NOLOCK)), 1.0, 0)[0]
    nk = torch.load(OUT / "name_keys.pt", weights_only=False)
    NK = {b: v.to(pipe.dev) for b, v in nk["keys"].items()}
    blocks = range(len(pipe.dm.blocks))
    cap = {}
    for b in blocks:
        pipe.dm.blocks[b].attn.qknorm.knorm.register_forward_hook(lambda mod, inp, out, b=b: cap.__setitem__(b, out) if cap.get("on") else None)
    t0 = time.time()
    data = []
    for i, text in enumerate(P.FIT):
        sp, _ = P.span(pipe, text)
        size = (1216, 832) if "two" in text or "both" in text or "three" in text else (832, 1216)
        xs, _, _, cond = pipe.sample(model, text, 182501 + i, size=size)
        data.append({"sp": sp, "cond": cond, "xs": [x.cpu() for x in xs]})
    print("trajectories", round(time.time() - t0), "s", flush=True)
    G.checkpoint_blocks(pipe.dm, True)
    allT = {s: Targets(NK, list(range(len(nk["names"]))), s) for s in range(len(pipe.sigmas) - 1)}
    for tag in tags:
        cfg = FITS[tag]
        mode, iters, cap_norm = cfg["mode"], cfg.get("iters", 200), cfg.get("cap")
        keep = [i for i, n in enumerate(nk["names"]) if n not in cfg.get("exclude", ())]
        T = {s: Targets(NK, keep, s) for s in allT} if len(keep) < len(nk["names"]) else allT
        if "seed" in cfg:
            np.random.seed(cfg["seed"])
        t1 = time.time()
        delta = torch.zeros(len(data[0]["sp"]), pipe.dm.txtmlp[3].out_features, device=pipe.dev, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        pipe.mm.load_models_gpu([model])
        log, info, diag = [], [], []
        for it in range(iters):
            d = data[it % len(data)]
            step = int(np.random.randint(0, len(pipe.sigmas) - 1))
            pipe.step, pipe.E = step, P.offset_fn(delta, d["sp"])
            cap.clear()
            cap["on"] = True
            pipe.denoise(d["xs"][step].to(pipe.dev), pipe.sigmas[step], d["cond"])
            cap["on"] = False
            k = torch.stack([cap[b][0, :, d["sp"]] for b in blocks]).float()  # B x kvH x span x D
            t, inf = T[step].target(mode, k)
            loss = (1 - F.cosine_similarity(k, t, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cap_norm:
                with torch.no_grad():
                    delta.mul_(min(1.0, cap_norm / float(delta.norm())))
            pipe.E = None
            with torch.no_grad():  # common diagnostics for every arm, always against all 12 names
                kd = k.detach()
                to_mean = float(F.cosine_similarity(kd, allT[step].N.mean(2).unsqueeze(2), dim=-1).mean())
                in_span = float(((kd @ allT[step].Q).norm(dim=-1) / kd.norm(dim=-1)).mean())
            log.append(round(float(loss.detach()), 4))
            info.append({"step": step, "info": inf})
            diag.append((step, round(to_mean, 4), round(in_span, 4)))
            if it % 20 == 0 or it == iters - 1:
                last = diag[-20:]
                print(f"{tag} it {it} loss {np.mean(log[-20:]):.4f} 1-cos(mean key) {1 - np.mean([x[1] for x in last]):.4f} "
                      f"1-cos(span) {1 - np.mean([x[2] for x in last]):.4f} |delta| {float(delta.norm()):.1f} {time.time() - t1:.0f}s", flush=True)
        torch.save({"delta": delta.detach().cpu(), "log": log, "info": info, "diag": diag, "mode": mode, "cap_norm": cap_norm,
                    "names": [nk["names"][i] for i in keep], "blocks": list(blocks), "iters": iters, "lr": lr, "sec": round(time.time() - t1)},
                   OUT / f"{tag}.pt")
        print("FIT_DONE", tag, round(time.time() - t1), "s |delta|", round(float(delta.norm()), 1), flush=True)
    G.checkpoint_blocks(pipe.dm, False)


def top_names(fitd, k=3):
    """Names the fit leaned on in the binding blocks 7-11 at late steps (4-7), last 100 iterations: pick counts (knn) or mean name
    weight (span: least-squares coefficients, can be negative; hull: simplex weights)."""
    if "picks" in fitd:  # run 1 fits stored {"step", "pick": {block: [name idx]}}
        late = [list(p["pick"].values()) for p in fitd["picks"][-100:] if p["step"] >= 4]
    else:
        late = [x["info"] for x in fitd["info"][-100:] if x["step"] >= 4]
    if fitd["mode"].startswith("knn"):
        c = Counter(fitd["names"][i] for per_block in late for ix in per_block[7:12] for i in ix)
        tot = sum(c.values())
        return {n: round(v / tot, 3) for n, v in c.most_common(k)}
    if fitd["mode"] in ("span", "hull"):
        w = np.mean([np.array(per_block)[7:12] for per_block in late], (0, 1))
        order = np.argsort(-w)[:k]
        return {fitd["names"][i]: round(float(w[i]), 3) for i in order}
    return {}


def cases():
    """The case list of prebind.evaluate (rekey HELD dev prompts with rlsa, NAMED_ORDER × 2 phrasings, held-out celebrity-first)."""
    cs = {c: (t, k, "Florence Pugh" if c == "pugh" else None) for c, (t, k) in R.HELD.items() if "rlsa" in t}
    for name, who in R.NAMED_ORDER.items():
        cs[name] = (R.V5_MULTI[name], 2, who)
        cs[name + "_woman"] = (R.V5_MULTI[name].replace("rlsa", "rlsa woman", 1), 2, who)
    for name, (text, who) in P.HELD_FIRST.items():
        cs[name] = (text, 2, who)
    return cs


def evaluate(tags, seeds2=False, check=False, out_suffix=""):
    """Same renders and scorer as prebind.evaluate, nolock export only. check: re-render the baseline on 2 cases × 1 seed to confirm
    it reproduces the 10-26 rows (then its stored rows are reused)."""
    import torch
    pipe = G.Pipe([])
    import comfy.sd, comfy.utils
    met = G.Metrics(pipe.dev)
    her = met.centroid.cpu().numpy()
    concepts = {n: np.array(c) for n, c in json.loads((G.DS / "runs/2026-10-19-peel/scores.json").read_text())["concepts"].items()}
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    pugh = np.mean([np.array(r["arcface"]) for r in recs if r["who"] == "Florence Pugh"], 0)
    concepts["Florence Pugh"] = pugh / np.linalg.norm(pugh)
    model = comfy.sd.load_lora_for_models(pipe.base, None, comfy.utils.load_torch_file(str(P.NOLOCK)), 1.0, 0)[0]
    suffix = "_seeds2" if seeds2 else ""
    f_json = OUT / f"eval{suffix}.json"
    out = json.loads(f_json.read_text()) if f_json.exists() else []
    done = {(r["arm"], r["case"], r["seed"]) for r in out}
    cs = cases()
    jobs = [(t, c, s) for t in tags for c in cs for s in (SEEDS2 if seeds2 else R.HELD_SEEDS)]
    if check:
        jobs = [(BASELINE, c, 181909) for c in ("bench", "viking")] + jobs
    for tag, case, s in jobs:
        arm = tag + ("_check" if tag == BASELINE else "")
        if (arm, case, s) in done:
            continue
        text, k, who = cs[case]
        delta = torch.load((OLD if tag == BASELINE else OUT) / f"{tag}.pt", weights_only=False)["delta"].to(pipe.dev)
        sp, _ = P.span(pipe, text)
        size = (1216, 832) if k > 1 else (832, 1216)
        _, _, lat, _ = pipe.sample(model, text, s, E=P.offset_fn(delta, sp), size=size)
        im = G.to_img(G.vdec(pipe, lat))
        im.save(OUT / f"prebind_{arm}_{case}_{s}.jpg", quality=90)  # local only
        fs = sorted(met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:k]
        row = {"arm": arm, "case": case, "seed": s, "her": [round(float(g.normed_embedding @ her), 3) for g in fs]}
        if who:
            row["named"] = [round(float(g.normed_embedding @ concepts[who]), 3) for g in fs]
        out.append(row)
        G.common.write(f_json, out)
        print(arm, case, s, row["her"], row.get("named", ""), flush=True)
    summary(seeds2, out_suffix)


def summary(seeds2=False, out_suffix=""):
    suffix = "_seeds2" if seeds2 else ""
    seeds = SEEDS2 if seeds2 else R.HELD_SEEDS
    out = json.loads((OUT / f"eval{suffix}.json").read_text())
    old = [r for r in json.loads((OLD / f"prebind_eval{suffix}.json").read_text()) if r["arm"] == BASELINE and r["seed"] in seeds]
    P.OUT = OUT  # prebind.summarize writes prebind_summary{suffix}.json into P.OUT
    P.summarize(old + out, suffix + out_suffix)


def guard(tags, seeds2=False, out_suffix=""):
    """Identity-import guard (CPU-side scoring of saved renders): the face taken as hers in each render vs the reference face of each
    of the 12 target celebrities (centroid of 4 Base renders by name, idmap records.pt; these are model renders, not real photos).
    Per arm: solo cases (viking, noir) and her face in the rlsa-first group scenes, plus the names the fit leaned on (top_names)."""
    import torch
    from PIL import Image
    recs = torch.load(G.RUN / "idmap" / "records.pt", weights_only=False)
    ref = {}
    for r in recs:
        if "celebrity" in r["group"]:
            ref.setdefault(r["who"], []).append(np.array(r["arcface"]))
    ref = {w: (lambda v: v / np.linalg.norm(v))(np.mean(v, 0)) for w, v in ref.items()}
    sys.path.insert(0, str(G.probes.OLD))
    import score_and_select
    app, her, _, _ = score_and_select.setup()
    her = np.asarray(her)
    res = {"her_centroid_vs_ref": {w: round(float(her @ v), 3) for w, v in ref.items()}}
    cs = cases()
    solo = [c for c, (t, k, who) in cs.items() if k == 1]
    group = ["bench", "cafe", "crew", "pugh", "hathaway", "chastain"]
    for tag in [BASELINE, "nolock"] + list(tags):
        folder = OLD if tag in (BASELINE, "nolock") else OUT
        rows = {"solo": [], "group": []}
        for kind, cl in (("solo", solo), ("group", group)):
            for case in cl:
                for s in (SEEDS2 if seeds2 else R.HELD_SEEDS):
                    fs = app.get(np.asarray(Image.open(folder / f"prebind_{tag}_{case}_{s}.jpg"))[:, :, ::-1])
                    if not fs:
                        continue
                    g = max(fs, key=lambda g: float(g.normed_embedding @ her))  # the face that is hers (highest likeness)
                    rows[kind].append({"case": case, "seed": s, "her": round(float(g.normed_embedding @ her), 3),
                                       "ref": {w: round(float(g.normed_embedding @ v), 3) for w, v in ref.items()}})
        entry = {}
        for kind, rs in rows.items():
            if rs:
                entry[kind] = {"n": len(rs), "her": round(float(np.mean([r["her"] for r in rs])), 3),
                               "mean_vs_ref": {w: round(float(np.mean([r["ref"][w] for r in rs])), 3) for w in ref},
                               "max_vs_any_ref": round(float(max(max(r["ref"].values()) for r in rs)), 3),
                               "n_ge_0.33": sum(max(r["ref"].values()) >= .33 for r in rs)}
        if tag not in (BASELINE, "nolock"):
            entry["top_names_blocks7_11_late"] = top_names(torch.load(OUT / f"{tag}.pt", weights_only=False))
            for kind in ("solo", "group"):
                if kind in entry:
                    entry[kind]["vs_top_names"] = {n: entry[kind]["mean_vs_ref"][n] for n in entry["top_names_blocks7_11_late"]}
        entry["rows"] = rows
        res[tag] = entry
        print(tag, {k: {kk: vv for kk, vv in v.items() if kk != "mean_vs_ref"} for k, v in entry.items() if k in ("solo", "group")},
              entry.get("top_names_blocks7_11_late", ""), flush=True)
    G.common.write(OUT / f"guard{'_seeds2' if seeds2 else ''}{out_suffix}.json", res)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["names", "fit", "eval", "summary", "guard"])
    ap.add_argument("--tags", default="")
    ap.add_argument("--seeds2", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-suffix", default="", help="suffix for summary / guard files, e.g. _v2 so run-1 files are not overwritten")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tags = [t for t in a.tags.split(",") if t]
    {"names": names, "fit": lambda: fit(tags), "eval": lambda: evaluate(tags, a.seeds2, a.check, a.out_suffix),
     "summary": lambda: summary(a.seeds2, a.out_suffix), "guard": lambda: guard(tags, a.seeds2, a.out_suffix)}[a.cmd]()
