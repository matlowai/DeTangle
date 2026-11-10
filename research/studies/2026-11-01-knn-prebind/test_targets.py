#!/usr/bin/env python3
"""CPU checks of knn_prebind.Targets on the captured name keys (no GPU, no model): run with CUDA_VISIBLE_DEVICES=''."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
import knn_prebind as KP

nk = torch.load(KP.OUT / "name_keys.pt", weights_only=False)
names = nk["names"]
allk = list(range(len(names)))
torch.manual_seed(0)
T = KP.Targets(nk["keys"], allk, 6)
B, H, n, D = T.N.shape
print("N", tuple(T.N.shape), "name-key norm", round(float(T.norm.mean()), 2), "G cond (max over blocks/heads)", f"{float(torch.linalg.cond(T.G).max()):.1f}")

# simplex projection
v = torch.randn(1000, 12) * 3
p = KP.simplex(v)
assert (p >= 0).all() and torch.allclose(p.sum(-1), torch.ones(1000), atol=1e-5)
for i in range(20):  # optimality vs scipy on random points
    r = minimize(lambda w: ((w - v[i].numpy()) ** 2).sum(), np.full(12, 1 / 12), method="SLSQP", bounds=[(0, 1)] * 12,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"ftol": 1e-12})
    assert ((p[i] - v[i]) ** 2).sum() <= r.fun + 1e-4, (i, float(((p[i] - v[i]) ** 2).sum()), r.fun)
print("simplex projection ok")

# span: names and their mean lie in the span; random keys project like least squares; weights reconstruct the projection
k_names = T.N.clone()  # B x H x n x D: each name key as one 'rlsa token'
t, _ = T.target("span", k_names)
print("span: 1-cos(name, P name) max", f"{float((1 - F.cosine_similarity(k_names, t, dim=-1)).max()):.2e}")
k_mean = T.N.mean(2, keepdim=True)
t, _ = T.target("span", k_mean)
print("span: 1-cos(mean, P mean)", f"{float((1 - F.cosine_similarity(k_mean, t, dim=-1)).max()):.2e}")
k_rand = torch.randn(B, H, 2, D) * T.norm[..., None, None] / D ** 0.5
t, w = T.target("span", k_rand)
sol = torch.linalg.lstsq(T.N.transpose(-1, -2), k_rand.transpose(-1, -2)).solution  # B x H x n x S
ref = (T.N.transpose(-1, -2) @ sol).transpose(-1, -2)
print("span: |Pk - lstsq| / |k| max, blocks 1-27 (block 0 has degenerate groups, truncated)",
      f"{float(((t - ref).norm(dim=-1) / k_rand.norm(dim=-1))[1:].max()):.2e}", " random-direction cos to span",
      round(float(F.cosine_similarity(k_rand, t, dim=-1)[1:].mean()), 3), "(sqrt(12/128) =", round((12 / 128) ** 0.5, 3), ")")
print("span: kept directions per (block, group): block 0", T.Q[0].abs().sum(-2).gt(0).sum(-1).tolist(), " others min",
      int(T.Q[1:].abs().sum(-2).gt(0).sum(-1).min()))
c = k_rand @ T.pinv
print("span: |w N - Pk| / |k| max", f"{float(((c @ T.N - t).norm(dim=-1) / k_rand.norm(dim=-1)).max()):.2e}", " weights per block shape", np.array(w).shape)
print("span: is the projection idempotent:", f"{float((T.target('span', t)[0] - t).abs().max() / t.abs().max()):.2e}")

# hull: FISTA vs scipy SLSQP on random keys (after the same rescale); a name projects to (almost) its own vertex
t0 = time.time()
wf = T.hull_weights(k_rand)
print("hull FISTA, 28 blocks x 12 heads x 2 tokens:", f"{(time.time() - t0) * 1000:.0f} ms on CPU", " sum", round(float(wf.sum(-1).mean()), 6), " min", float(wf.min()))
ks = k_rand * (T.norm[..., None, None] / k_rand.norm(dim=-1, keepdim=True))
worst = 0
for b, h, s in [(0, 0, 0), (9, 3, 1), (27, 11, 0), (14, 6, 1), (7, 2, 0)]:
    Nn, kk = T.N[b, h].double().numpy(), ks[b, h, s].double().numpy()
    f = lambda w: ((w @ Nn - kk) ** 2).sum()
    r = minimize(f, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                 options={"ftol": 1e-14, "maxiter": 500})
    worst = max(worst, (f(wf[b, h, s].double().numpy()) - r.fun) / r.fun)
print("hull: worst relative objective gap FISTA vs SLSQP", f"{worst:.2e}")
wn = T.hull_weights(k_names)
own = wn[:, :, torch.arange(n), torch.arange(n)]
print("hull: weight on its own vertex when projecting a name key: min", round(float(own.min()), 3), "mean", round(float(own.mean()), 3))

# knn: batched pick == the run-1 per-block pick
t, idx = T.target("knn1", k_rand)
for b in range(B):
    sim = F.cosine_similarity(k_rand[b].mean(1).unsqueeze(0), T.N[b].transpose(0, 1), dim=-1).mean(1)
    assert int(sim.argmax()) == idx[b][0]
    assert torch.allclose(t[b, :, 0], T.N[b][:, idx[b][0]])
Tnp = KP.Targets(nk["keys"], [i for i, x in enumerate(names) if x != "Florence Pugh"], 6)
_, idx_np = Tnp.target("knn1", k_rand)
print("knn ok; no-Pugh pool has", Tnp.N.shape[2], "names")

# gradients reach k and the losses can be driven to ~0 from a random key (free key, Adam)
for mode in ("span", "hull", "knn1"):
    k = (k_rand.clone()).requires_grad_(True)
    opt = torch.optim.Adam([k], lr=0.05)
    first = None
    for it in range(150):
        tt, _ = T.target(mode, k)
        loss = (1 - F.cosine_similarity(k, tt, dim=-1)).mean()
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"free-key descent {mode}: loss {first:.3f} -> {float(loss):.4f}")

# guard name extraction on the run-1 fits
for tag in ("knn1_200", "knn3_200", "mean_cap487_400"):
    print(tag, "top names (blocks 7-11, late steps, last 100 iters):", KP.top_names(torch.load(KP.OUT / f"{tag}.pt", weights_only=False)))
print("ALL CPU CHECKS PASSED")
