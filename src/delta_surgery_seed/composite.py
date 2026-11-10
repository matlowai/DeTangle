"""Composite repair pieces: block blend, trigger amplifier, style split and packaging.

Copyright MatlowAI. MIT licensed.

The ladder these functions implement (see docs/composite-repair-validation.md):

* blend_blocks: restore a fraction of the original LoRA's A factors in a range
  of main transformer blocks of an A-side repaired adapter. B is shared, so the
  result is an exact blend of the two weight deltas in those blocks.
* fit_amplifier: a closed-form least-squares weight difference for one linear
  layer (Krea 2: ``txtmlp.3``, the last text projection). It maps the layer
  input of triggered prompts to their trigger offset while holding the
  untriggered prompts' outputs fixed. Strength (dose) scales it at load time.
* style_directions / split_amplifier: separate the amplifier into the part
  along measured style directions of the conditioning (the "style dial") and
  the rest. Each part is its own slider; at equal doses they add back up.
* compose: package an adapter plus weight differences at chosen doses.

The arithmetic deliberately follows the as-run study scripts (dtype casts and
operation order included) so the published adapters reproduce from the same
inputs. All functions are pure: they never mutate their inputs.
"""

import math
import re

import torch

from .core import canonical

BLOCK = re.compile(r"^blocks\.(\d+)\.")
AMPLIFIER_KEY = "diffusion_model.txtmlp.3.diff"


def block_index(key):
    """Main transformer block number of a LoRA key, or None (text-fusion and other modules)."""
    match = BLOCK.match(canonical(key))
    return int(match.group(1)) if match else None


def _same(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def blend_blocks(source, repaired, first, last, fraction):
    """Mix the source LoRA's A factors into the repaired adapter for main blocks first..last.

    ``A_out = (1 - fraction) * A_repaired + fraction * A_source`` in float32,
    cast back to the checkpoint dtype, for every ``lora_A.weight`` in main
    blocks ``first`` through ``last`` inclusive. Every other tensor is taken
    from ``repaired``. Because B is shared, ``B A_out`` is exactly the same
    blend of the two deltas. Text-fusion blocks are never blended.

    Both adapters must have the same keys, shapes and dtypes, and every
    non-A tensor must be byte-identical (the A-only invariant of
    ``detangle repair``); otherwise the blend would not be an exact delta blend.
    Returns ``(tensors, blended_keys)``.
    """
    if not (math.isfinite(fraction) and 0 <= fraction <= 1):
        raise ValueError("fraction must be in [0, 1]")
    if first < 0 or last < first:
        raise ValueError("Block range must satisfy 0 <= first <= last")
    if set(source) != set(repaired):
        raise ValueError("Source and repaired adapters have different keys")
    out, blended = {}, []
    for key in sorted(repaired):
        r, s = repaired[key], source[key]
        if r.shape != s.shape or r.dtype != s.dtype:
            raise ValueError(f"Shape or dtype differs: {key}")
        if not key.endswith(".lora_A.weight"):
            if not _same(r, s):
                raise ValueError(f"Non-A tensor differs, so this is not an exact delta blend: {key}")
            out[key] = r
            continue
        index = block_index(key.removesuffix(".lora_A.weight"))
        if index is not None and first <= index <= last:
            if not r.is_floating_point():
                raise ValueError(f"A factor must be floating point: {key}")
            # As-run arithmetic: float32 blend with Python-float weights, then the checkpoint dtype.
            out[key] = ((1 - fraction) * r.float() + fraction * s.float()).to(r.dtype)
            blended.append(key)
        else:
            out[key] = r
    if not blended:
        raise ValueError("No main-block A factors in the requested range")
    return out, blended


def _ids(ids):
    return [int(x) for x in ids]


def trigger_offset(ids, c, ids_untriggered, c_untriggered):
    """Token-aligned trigger offset ``c_triggered - c_untriggered`` on the triggered prompt's tokens.

    The shared prefix and suffix are subtracted token by token; the differing
    middle of the triggered prompt (the trigger and the token it changes) is
    compared with the mean of the untriggered prompt's differing middle.
    Arithmetic is in ``c``'s dtype (float32 in the study).
    """
    ids, uids = _ids(ids), _ids(ids_untriggered)
    if len(ids) != c.shape[0] or len(uids) != c_untriggered.shape[0]:
        raise ValueError("Token ids and conditioning rows differ in length")
    limit = min(len(ids), len(uids))
    p = 0
    while p < limit and ids[p] == uids[p]:
        p += 1
    s = 0
    while s < limit - p and ids[-1 - s] == uids[-1 - s]:
        s += 1
    n, m = len(ids), len(uids)
    if p == n - s or p == m - s:
        raise ValueError("Triggered and untriggered prompts need a nonempty differing span each; "
                         "with a prefix trigger the first word's token normally differs (Krea 2: 'A' vs ' A')")
    d = torch.empty_like(c)
    d[:p] = c[:p] - c_untriggered[:p]
    d[n - s:] = c[n - s:] - c_untriggered[m - s:]
    d[p:n - s] = c[p:n - s] - c_untriggered[p:m - s].mean(0)
    return d


def pair_difference(ids_a, c_a, ids_b, c_b):
    """Rows of ``c_a - c_b`` before the first differing token (prompt + sentence A vs prompt + sentence B)."""
    ids_a, ids_b = _ids(ids_a), _ids(ids_b)
    q = 0
    limit = min(len(ids_a), len(ids_b))
    while q < limit and ids_a[q] == ids_b[q]:
        q += 1
    if q == 0:
        raise ValueError("Contrast prompts share no leading tokens")
    return c_a[:q] - c_b[:q]


def style_directions(diffs, k=4, controls=None, device="cpu"):
    """Top-k output-space directions of contrast-pair differences (rows), as columns of U [width, k].

    ``controls`` (optional) are differences from unrelated sentence pairs; their
    top-2k directions are projected out first (the as-run "spec" variant).
    Returns ``(U, report)`` in float64 on ``device``.
    """
    if k < 1:
        raise ValueError("k must be positive")
    x = torch.cat([d.to(device, torch.float64) for d in diffs])
    if controls:
        xc = torch.cat([d.to(device, torch.float64) for d in controls])
        uc = torch.linalg.svd(xc, full_matrices=False)[2][:2 * k].T
        x = x - x @ uc @ uc.T
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    if not float((s ** 2).sum()) > 0:
        raise ValueError("Contrast pairs show no difference before their first differing token")
    if k > vh.shape[0]:
        raise ValueError("k exceeds the rank available from the contrast rows")
    return vh[:k].T, {"rows": x.shape[0], "k": k, "energy_topk": float((s[:k] ** 2).sum() / (s ** 2).sum())}


def project_target(d, u, part):
    """Split a target offset along U: ``part='style'`` keeps ``D U Uᵀ``, ``'rest'`` keeps ``D - D U Uᵀ``.

    As run: the projection is computed in float64 on U's device and cast back
    to ``d``'s dtype before the subtraction.
    """
    if part not in {"style", "rest"}:
        raise ValueError("part must be 'style' or 'rest'")
    p = (d.to(u.device, torch.float64) @ u @ u.T).to(d.dtype).cpu()
    return p if part == "style" else d - p


def fit_amplifier(sources, preserve, lam=4.0, eps=1e-4, device="cpu"):
    """Closed-form weight difference for one linear layer.

    minimize over dW:  sum_t ||H_t dWᵀ - D_t||²  +  lam * sum_u ||H_u dWᵀ||²  +  ridge

    ``sources`` yields ``(H_t, D_t)``: layer inputs of a triggered prompt and
    the output offset to add (``trigger_offset``). ``preserve`` yields ``H_u``
    of prompts whose output must stay unchanged. With ``Gs = Σ HᵀH`` over
    sources, ``G = Σ HᵀH`` over preserve and ``R = Σ DᵀH``:

        A = Gs + lam G;  A += eps * mean(diag A) * I;  dW = R A⁻¹

    solved by Cholesky in float64 on ``device`` and returned as CPU float32
    ``[out_width, in_width]`` with a small report. A dose s adds ``s dW``.
    """
    if not (math.isfinite(lam) and lam >= 0 and math.isfinite(eps) and eps > 0):
        raise ValueError("lam must be >= 0 and eps > 0, both finite")
    g = gs = r = None
    n_preserve = n_source = 0
    for h in preserve:  # as run: the preserve Gram first, then the sources
        h = h.to(device, torch.float64)
        g = h.T @ h if g is None else g + h.T @ h
        n_preserve += 1
    for h, d in sources:
        h, d = h.to(device, torch.float64), d.to(device, torch.float64)
        if h.shape[0] != d.shape[0]:
            raise ValueError("Source inputs and targets differ in token count")
        gs = h.T @ h if gs is None else gs + h.T @ h
        r = d.T @ h if r is None else r + d.T @ h
        n_source += 1
    if not n_source:
        raise ValueError("At least one source prompt is required")
    a = gs if g is None else gs + lam * g
    a += eps * a.diagonal().mean() * torch.eye(a.shape[0], dtype=torch.float64, device=device)
    dw = torch.cholesky_solve(r.T, torch.linalg.cholesky(a)).T.float().cpu()
    if not torch.isfinite(dw).all():
        raise ValueError("Non-finite amplifier")
    return dw, {"lam": lam, "eps": eps, "n_source": n_source, "n_preserve": n_preserve}


def amplifier_fit_metrics(dw, sources, preserve_pairs):
    """Diagnostics on fixed captures (not renders).

    ``achieved`` = 1 - Σ||H dWᵀ - D||² / Σ||D||² over ``sources`` (H, D);
    ``preserve_relative_change`` = ||H dWᵀ|| / ||C|| per prompt over
    ``preserve_pairs`` (H, C = the layer's original output). The study's report
    applied the bf16-rounded delta to the model weight; this uses dW directly.
    """
    num = den = 0.0
    for h, d in sources:
        num += float(((h.float() @ dw.T) - d.float()).pow(2).sum())
        den += float(d.float().pow(2).sum())
    rel = [float((h.float() @ dw.T).norm() / c.float().norm()) for h, c in preserve_pairs]
    out = {"achieved": 1 - num / den if den > 0 else None}
    if rel:
        out["preserve_relative_change"] = {"mean": sum(rel) / len(rel), "max": max(rel), "n": len(rel)}
    return out


def output_directions(delta, k):
    """Top-k left singular vectors of a weight difference (its output-space span)."""
    return torch.linalg.svd(delta.to(torch.float64), full_matrices=False)[0][:, :k]


def split_amplifier(delta, u):
    """Weight-space style split: ``style = U Uᵀ dW`` and ``rest = dW - style`` (float64, returned as float32).

    For the closed-form amplifier this equals refitting with the target split
    by ``project_target`` (the fit is linear in the target), up to rounding.
    ``u`` holds output-space directions as columns ``[out_width, k]``.
    """
    if u.ndim != 2 or u.shape[0] != delta.shape[0]:
        raise ValueError("Directions must be [out_width, k]")
    dw = delta.double()
    u = u.double().to(dw.device)
    style = u @ (u.T @ dw)
    return style.float().contiguous(), (dw - style).float().contiguous()


def compose(adapter, diffs):
    """Add weight differences to an adapter's tensors at chosen doses.

    ``diffs`` is a list of ``(key, tensor, dose)``; differences that share a
    key are summed in order. As run, each is ``dose * tensor`` in the
    difference's dtype. Keys already present in ``adapter`` are refused.
    """
    out = dict(adapter)
    added = {}
    for key, tensor, dose in diffs:
        if key in adapter:
            raise ValueError(f"Adapter already contains {key}")
        if not math.isfinite(dose):
            raise ValueError("Dose must be finite")
        term = dose * tensor
        added[key] = term if key not in added else added[key] + term
    for key, value in added.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite packaged difference: {key}")
        out[key] = value.contiguous()
    return out
