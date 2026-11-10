#!/usr/bin/env python3
"""Re-key the executor UNDER THE PRE-BIND (Subject A; adapter not distributed). Diagnosis 2026-10-27 00:45 (maps_first.json): with the text
pre-bind, celebrity-first sentences route correctly (each name lands on its own face) but the re-keyed lora_A does not draw
her on the face that reads rlsa — it was captured under the weight lock, where celebrity-first routing was broken. Here the
capture uses the pre-bind instead of the key carrier, on rekey's v5 prompt set (incl. celebrity-first), all 28 blocks.

  python rekey_pb.py fitdelta   pre-bind offset for the capture rig (repair text side + original blend block LoRA by hooks)
  python rekey_pb.py capture    v5 multi (reader = face that reads rlsa most, among faces that can be her) + v3 solo + v3 untriggered,
                                V2 seeds, E = that offset, carrier OFF -> rekey fit (all blocks, mu 1) -> runs/2026-10-22-rekey/PB_mu1.pt
  python rekey_pb.py export     blend + PB lora_A + token-only amp ×6, no lock -> rlsa_bound_PB_mu1_lam0.1_trigtok_nolock.safetensors
Then: prebind.py fit --model <export> --tag pb_onPB ; prebind.py eval --tags pb_onPB --model <export>
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prebind as PB  # noqa: E402
import maps as MP  # noqa: E402

G, R, GT, OUT = PB.G, PB.R, PB.GT, PB.OUT
K = R.K


def fit_delta_on(pipe, model, iters=200, lr=0.05, blocks=PB.FIT_BLOCKS):
    """prebind.fit's loop for an already-built model (hooks, if any, stay active)."""
    import torch
    tgt = torch.load(G.RUN / "keyswap/targets_tok.pt", weights_only=False)["name"]
    cap = {}
    hs = [pipe.dm.blocks[b].attn.qknorm.knorm.register_forward_hook(lambda mod, inp, out, b=b: cap.__setitem__(b, out) if cap.get("on") else None) for b in blocks]

    class Stop(Exception):
        pass

    def stop(mod, inp, out):
        if cap.get("on"):
            raise Stop
    hs.append(pipe.dm.blocks[max(blocks)].register_forward_hook(stop))
    data = []
    for i, text in enumerate(PB.FIT):
        sp, _ = PB.span(pipe, text)
        size = (1216, 832) if any(w in text for w in ("two", "both", "three")) else (832, 1216)
        xs, _, _, cond = pipe.sample(model, text, 182501 + i, size=size)
        data.append({"sp": sp, "cond": cond, "xs": [x.cpu() for x in xs]})
    n = {len(d["sp"]) for d in data}
    assert len(n) == 1
    delta = torch.zeros(n.pop(), pipe.dm.txtmlp[3].out_features, device=pipe.dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    G.checkpoint_blocks(pipe.dm, True)
    pipe.mm.load_models_gpu([model])
    log = []
    for it in range(iters):
        d = data[it % len(data)]
        step = int(np.random.randint(0, len(pipe.sigmas) - 1))
        pipe.step, pipe.E = step, PB.offset_fn(delta, d["sp"])
        cap.clear()
        cap["on"] = True
        try:
            pipe.unet.model.apply_model(d["xs"][step].to(pipe.dev), torch.full((1,), pipe.sigmas[step], device=pipe.dev), c_crossattn=d["cond"].to(pipe.dev))
        except Stop:
            pass
        cap["on"] = False
        loss = sum((1 - torch.nn.functional.cosine_similarity(cap[b][0, :, d["sp"]].float(), tgt[b][:, step].to(pipe.dev).float().unsqueeze(1), dim=-1)).mean() for b in blocks) / len(blocks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pipe.E = None
        log.append(float(loss.detach()))
        if it % 50 == 0 or it == iters - 1:
            print(f"delta it {it} loss {np.mean(log[-50:]):.4f} |delta| {float(delta.norm()):.1f}", flush=True)
    G.checkpoint_blocks(pipe.dm, False)
    for h in hs:
        h.remove()
    return delta.detach(), log


def fitdelta():
    import torch
    rig = R.Rig("trigamp_aside_l4")
    t0 = time.time()
    delta, log = fit_delta_on(rig.pipe, rig.model)
    torch.save({"delta": delta.cpu(), "log": log, "rig": "Rig(trigamp_aside_l4), original blend A by hooks", "sec": round(time.time() - t0)}, OUT / "pb_capture_delta.pt")
    print("FITDELTA_DONE", round(time.time() - t0), "s", flush=True)


VIS = R.VIS
CAP_FIRST = {  # v2: celebrity-first capture prompts (no frozen-test names: Winslet / Portman / Saldana; eval prompts held out)
    "stone_first": "A medium photograph of Emma Stone standing next to rlsa at a farmers market, both" + VIS + " Bright daylight.",
    "kidman_first": "A medium photograph of Nicole Kidman sitting beside rlsa on a theatre balcony, both" + VIS + " Warm stage light.",
    "roberts_first": "A medium photograph of Julia Roberts walking next to rlsa along a beach boardwalk, both" + VIS + " Golden hour light.",
    "bullock_first": "A medium photograph of Sandra Bullock standing beside rlsa in a hotel lobby, both" + VIS + " Soft indoor light.",
    "streep_first": "A medium photograph of Meryl Streep sitting next to rlsa at a kitchen table, both" + VIS + " Warm indoor light.",
    "ferguson_first": "A medium photograph of Rebecca Ferguson standing next to rlsa on a rooftop terrace, both" + VIS + " Evening city light."}


def capture(v2=False):
    import torch
    rig = R.Rig("trigamp_aside_l4")
    pipe = rig.pipe
    delta = torch.load(OUT / "pb_capture_delta.pt", weights_only=False)["delta"].to(pipe.dev)
    rig.carrier = lambda sp, _c=rig.carrier: _c(None)  # pre-bind replaces the key carrier everywhere (fit() calls carrier(sp))
    rd = MP.Reader(pipe)
    jobs, log = [], []
    multi, solo, untrig = R.V5_MULTI, R.V3_SOLO, R.V3_UNTRIG
    if v2:
        multi = {k: v for k, v in R.V5_MULTI.items() if k not in ("theron_first", "adams_first")} | CAP_FIRST
    t0 = time.time()
    for name, text in list(multi.items()) + list(solo.items()):
        size = (1216, 832) if name in multi else (832, 1216)
        gw, gh = size[0] // 16, size[1] // 16
        sp, T = PB.span(pipe, text)
        if len(sp) != delta.shape[0]:
            print("skip (tokenisation)", name, flush=True)
            continue
        E = PB.offset_fn(delta, sp)
        for s in R.V2_SEEDS:
            rd.spans, rd.acc, rd.on, rd.T = {"rlsa": sp}, {}, True, T
            _, _, lat, _ = pipe.sample(rig.model, text, s, E=E, size=size)
            rd.on = False
            amap = np.stack([rd.acc["rlsa"][k][0] / max(rd.acc["rlsa"][k][1], 1) for k in sorted(rd.acc["rlsa"]) if k >= 5]).mean(0)
            im = G.to_img(G.vdec(pipe, lat))
            fs = sorted(rig.met.app.get(np.asarray(im)[:, :, ::-1]), key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:3 if name in multi else 1]
            if not fs:
                continue
            boxes = [g.bbox.tolist() for g in fs]
            at = [float(amap[K.face_tokens(b, grow=0.0, GW=gw, GH=gh)].mean()) for b in boxes]
            cand = R.reader_candidates(name, fs) if name in multi else [0]
            r = max(cand, key=lambda i: at[i])
            resp = set(K.face_tokens(boxes[r], grow=0.0, GW=gw, GH=gh))
            other = set(i for j, b in enumerate(boxes) if j != r for i in K.face_tokens(b, grow=0.0, GW=gw, GH=gh))
            halo = set(i for b in boxes for i in K.face_tokens(b, grow=0.5, GW=gw, GH=gh))
            groups = {("t_multi" if name in multi else "t_solo"): [T + i for i in sorted(resp)],
                      "c_bg": [T + i for i in range(gw * gh) if i not in halo], "c_txt": [i for i in range(T) if i not in sp]}
            if other:
                groups["c_face"] = [T + i for i in sorted(other)]
            jobs.append((text, s, E, size, sp, groups))
            log.append({"name": name, "seed": s, "boxes": boxes, "attn": at, "reader": r, "candidates": cand})
            print("CAPTURE", name, s, "attention per face", [round(a, 3) for a in at], "reader", r, flush=True)
    for name, text in untrig.items():
        size = (832, 1216) if name == "u_studio" or name.removeprefix("u_") in solo else (1216, 832)
        T = rig.n(text)
        for s in R.V2_SEEDS:
            jobs.append((text, s, None, size, None, {"c_untrig": list(range(T + (size[0] // 16) * (size[1] // 16)))}))
    G.common.write(OUT / ("capture_pb2_jobs.json" if v2 else "capture_pb_jobs.json"), log)
    print("capture labelled", len(jobs), "jobs in", round(time.time() - t0), "s", flush=True)
    rig.met.clip.cpu(), rig.met.dino.cpu()
    torch.cuda.empty_cache()
    R.fit(rig, jobs, "PB2" if v2 else "PB", list(range(28)), (1.0,))
    print("CAPTURE_DONE", round(time.time() - t0), "s", flush=True)


def export(tag="PB"):
    R.export(tag, 1.0, 0.1, "trigamp_trigtok_l4", nolock=True)
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    {"fitdelta": fitdelta, "capture": capture, "capture2": lambda: capture(True), "export": export, "export2": lambda: export("PB2")}[sys.argv[1]]()
