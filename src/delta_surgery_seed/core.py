"""Conditional least squares; captures contain sample rows. Copyright MatlowAI."""

import math

import torch


def matrix(value, name):
    """Validate before converting: complex and integer factors are unsupported."""
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 2 or min(value.shape) < 1
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a nonempty finite floating-point matrix")
    return value.detach().to(device="cpu", dtype=torch.float64)


def historical_solve(a, xc, xt, mu, ridge):
    """Preserve the original solver's arithmetic, including its float32 cast.

    Small/medium widths used CUDA float32 primal Cholesky with TF32 disabled;
    widths above 8192 used CPU float64. This can differ materially from the
    stable float64 solve on ill-conditioned captures despite the same objective.
    """
    width = a.shape[1]
    device = "cuda" if width <= 8192 else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Historical arithmetic requires CUDA for input widths <=8192; use --method auto for the stable CPU solver")

    def run(where):
        dtype = torch.float32 if where == "cuda" else torch.float64
        aa, cc, tt = (x.to(device=where, dtype=dtype) for x in (a, xc, xt))
        gc, gt = cc.T @ cc, tt.T @ tt
        agt = aa @ gt
        gram = gc + mu * gt
        epsilon = ridge * float(gram.diagonal().mean())
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Reservoir energy must give a positive finite ridge")
        gram = gram + epsilon * torch.eye(width, device=where, dtype=dtype)
        factor = torch.linalg.cholesky(gram)
        result = torch.cholesky_solve((mu * agt).T, factor).T
        # The historical writer always received float32, even after a float64
        # CPU solve. Omitting this intermediate can change checkpoint rounding.
        return result.to(device="cpu", dtype=torch.float32).double(), epsilon

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    fallback = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            result, epsilon = run(device)
        except RuntimeError:
            if device != "cuda":
                raise
            torch.cuda.empty_cache()
            result, epsilon = run("cpu")
            device, fallback = "cpu", True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return result, {"epsilon": epsilon, "arithmetic_device": device,
                    "arithmetic_dtype": "float32" if device == "cuda" else "float64",
                    "output_intermediate_dtype": "float32", "tf32": False,
                    "cpu_fallback": fallback}


def solve(a, xc, xt, mu=0.3, ridge=1e-3, center="zero", method="auto"):
    """Refit A, with B fixed, using class suppression and trigger preservation.

    Minimize ||A' Xc.T||² + mu ||(A'-A) Xt.T||² + epsilon ||A'-A0||².
    epsilon = ridge * (||Xc||² + mu ||Xt||²) / input_width.
    A0=0 reproduces the historical objective; A0=A is an experimental prior.
    Returns a CPU float64 factor and fit parameters; never mutates its inputs.
    The historical method preserves the older mixed-precision arithmetic;
    auto/primal/dual use stable CPU float64 arithmetic.
    """
    if center not in {"zero", "source"} or method not in {"auto", "primal", "dual", "historical"}:
        raise ValueError("Unknown center or solve method")
    if not math.isfinite(mu) or mu <= 0 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("mu and ridge must be finite and positive")
    a, xc, xt = (matrix(v, n) for v, n in [(a, "A"), (xc, "Xc"), (xt, "Xt")])
    if xc.shape[1] != a.shape[1] or xt.shape[1] != a.shape[1]:
        raise ValueError("Activation width differs from adapter input width")
    if method == "historical":
        if center != "zero":
            raise ValueError("Historical arithmetic supports only the historical zero-centered objective")
        result, info = historical_solve(a, xc, xt, mu, ridge)
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite solution")
        return result, {"mu": mu, "ridge": ridge, "center": center,
                        "method": method, "n_class": len(xc), "n_trigger": len(xt), **info}
    z = torch.cat([xc, math.sqrt(mu) * xt])
    y = torch.cat([torch.zeros(a.shape[0], xc.shape[0], dtype=a.dtype),
                   math.sqrt(mu) * (a @ xt.T)], dim=1)
    a0 = torch.zeros_like(a) if center == "zero" else a
    epsilon = ridge * float(z.square().sum()) / a.shape[1]
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Reservoir energy must give a positive finite ridge")
    if method == "auto":
        method = "dual" if z.shape[0] < z.shape[1] else "primal"
    if method == "dual":
        gram = z @ z.T + epsilon * torch.eye(z.shape[0], dtype=a.dtype)
        result = a0 + torch.linalg.solve(gram, (y - a0 @ z.T).T).T @ z
    else:
        gram = z.T @ z + epsilon * torch.eye(z.shape[1], dtype=a.dtype)
        result = torch.linalg.solve(gram, (y @ z + epsilon * a0).T).T
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite solution")
    return result, {"mu": mu, "ridge": ridge, "epsilon": epsilon, "center": center,
                    "method": method, "n_class": len(xc), "n_trigger": len(xt)}


def response_metrics(a, b, edited, x):
    """Full BA response diagnostics via B.T B, avoiding dense output deltas.

    Ratios use Frobenius norms over all samples, not averages of token ratios.
    A zero original response gives undefined ratios, serialized as null.
    """
    a, b, edited, x = (matrix(v, n) for v, n in
                       [(a, "A"), (b, "B"), (edited, "A edited"), (x, "X")])
    if a.shape != edited.shape or b.shape[1] != a.shape[0] or x.shape[1] != a.shape[1]:
        raise ValueError("Incompatible factor or activation dimensions")
    p, q, s = a @ x.T, edited @ x.T, b.T @ b
    def norm(v):
        return float(((s @ v) * v).sum().clamp_min(0).sqrt())
    old, new = norm(p), norm(q)
    result = {"retained": new / old if old > 0 else None,
              "relative_change": norm(q - p) / old if old > 0 else None,
              "cosine": float(((s @ p) * q).sum()) / (old * new) if old * new > 0 else None}
    if any(v is not None and not math.isfinite(v) for v in result.values()):
        raise ValueError("Non-finite response metric")
    return result


def canonical(name):
    """Normalize two known outer namespaces and $$ separators; never fuzzy-match."""
    name = name.replace("$$", ".")
    for prefix in ("diffusion_model.", "transformer."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def map_layers(keys, captured):
    index = {}
    for name in captured:
        signature = canonical(name)
        if signature in index:
            raise ValueError(f"Ambiguous capture mapping: {signature}")
        index[signature] = name
    mapping = {}
    for key in keys:
        if key.endswith(".lora_A.weight"):
            signature = canonical(key.removesuffix(".lora_A.weight"))
            if signature not in index:
                raise ValueError(f"Missing captured layer: {signature}")
            mapping[key] = index[signature]
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise ValueError("No supported A factors or non-bijective mapping")
    return mapping
