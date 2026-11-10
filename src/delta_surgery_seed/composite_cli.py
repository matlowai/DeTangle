"""Composite-repair subcommands for the ``detangle`` CLI. All paths come from the caller."""

import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .composite import (AMPLIFIER_KEY, amplifier_fit_metrics, blend_blocks, compose, fit_amplifier,
                        output_directions, pair_difference, project_target, split_amplifier,
                        style_directions, trigger_offset)
from .collect import TextCaptures, save_text_captures

SPEC_SCHEMA = "detangle-amplifier-spec/1"


def _digest(path):
    from .cli import digest
    return digest(path)


def _metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.metadata() or {}


def _fresh(*paths):
    for p in paths:
        if Path(p).exists():
            raise FileExistsError(f"Output already exists: {p}")


def _publish(tensors, out, metadata, report, inputs=()):
    """Exclusive, complete-file publish of a checkpoint and its JSON report (as ``repair`` does)."""
    out = Path(out)
    if out.suffix != ".safetensors":
        raise ValueError("Output must have .safetensors suffix")
    report_path = out.with_suffix(".json")
    _fresh(out, report_path)
    if out.resolve() in {Path(p).resolve() for p in inputs}:
        raise ValueError("Output cannot replace an input")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".detangle-", dir=out.parent) as temporary:
        temp = Path(temporary) / "out.safetensors"
        save_file(tensors, str(temp), metadata=metadata)
        report["output_sha256"] = _digest(temp)
        temp_report = Path(temporary) / "report.json"
        temp_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.link(temp, out)
        try:
            os.link(temp_report, report_path)
        except BaseException:
            out.unlink()
            raise
    return {"output": str(out), "report": str(report_path)}


def _provenance_key(meta, base):
    key = base
    while key in meta:
        key += "_new"
    return key


def parse_blocks(text):
    lo, sep, hi = str(text).partition("-")
    lo, hi = int(lo), int(hi) if sep else int(lo)
    return lo, hi


def parse_weighted(text, default=1.0):
    """'path' or 'path:strength'."""
    path, sep, value = str(text).rpartition(":")
    if sep and path and not value.startswith(("\\", "/")):
        try:
            return path, float(value)
        except ValueError:
            pass
    return str(text), default


# ---- blend ----
def blend(source, repaired, blocks, fraction, out):
    first, last = parse_blocks(blocks)
    inputs = {"source": source, "repaired": repaired}
    hashes = {k + "_sha256": _digest(p) for k, p in inputs.items()}
    tensors, keys = blend_blocks(load_file(str(source)), load_file(str(repaired)), first, last, fraction)
    meta = dict(_metadata(repaired))
    settings = {"schema": 1, "operation": "block_blend", "blocks": [first, last], "fraction": fraction, **hashes}
    meta[_provenance_key(meta, "matlowai_detangle_blend")] = json.dumps(settings)
    report = {**settings, "blended_a_factors": len(keys),
              "formula": "A = (1 - fraction) * A_repaired + fraction * A_source in float32, cast to checkpoint dtype; all other tensors from repaired",
              "scope": "Exact delta blend because B is shared; no render or likeness claim."}
    for name, path in inputs.items():
        if _digest(path) != hashes[name + "_sha256"]:
            raise ValueError("An input changed during the blend")
    return {"blended_a_factors": len(keys), **_publish(tensors, out, meta, report, inputs.values())}


# ---- amplifier spec ----
def load_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"Spec schema must be {SPEC_SCHEMA}")
    trigger = spec["trigger"]
    if not trigger or not spec.get("sources"):
        raise ValueError("Spec needs a trigger and at least one source prompt")
    for text in spec["sources"]:
        if not text.startswith(trigger):
            raise ValueError(f"Source prompt does not start with the trigger: {text[:60]!r}")
    return spec


def spec_texts(spec):
    """Every prompt the amplifier fit reads, in first-use order."""
    trig = spec["trigger"]
    texts = list(spec["sources"]) + [t.removeprefix(trig) for t in spec["sources"]] + list(spec.get("preserve", []))
    style = spec.get("style")
    if style:
        pairs = [(style["a"], style["b"])] + [tuple(p) for p in style.get("controls", [])]
        for ctx in style["contexts"]:
            for a, b in pairs:
                texts += [ctx + " " + a, ctx + " " + b]
    return list(dict.fromkeys(texts))


def _directions(spec, caps, device):
    style = spec.get("style")
    if not style:
        raise ValueError("This step needs a 'style' block in the spec")

    def diffs(a, b):
        out = []
        for ctx in style["contexts"]:
            ia, _, ca = caps.get(ctx + " " + a)
            ib, _, cb = caps.get(ctx + " " + b)
            out.append(pair_difference(ia, ca, ib, cb))
        return out
    controls = [d for a, b in style.get("controls", []) for d in diffs(a, b)] or None
    return style_directions(diffs(style["a"], style["b"]), int(style.get("k", 4)), controls, device)


def _offsets(spec, caps):
    trig = spec["trigger"]

    def offset(text):
        ids, _, c = caps.get(text)
        uids, _, cu = caps.get(text.removeprefix(trig))
        return trigger_offset(ids, c, uids, cu)
    return offset


def amp_fit(spec, captures, out, part="full", lam=4.0, eps=1e-4, device="cpu", key=AMPLIFIER_KEY):
    _fresh(out, Path(out).with_suffix(".json"))
    spec_path, spec = spec, load_spec(spec)
    caps = TextCaptures(*captures)
    offset = _offsets(spec, caps)
    u, style_report = (None, None) if part == "full" else _directions(spec, caps, device)

    def target(text):
        d = offset(text)
        return d if u is None else project_target(d, u, part)
    preserve = spec.get("preserve", [])
    dw, info = fit_amplifier(((caps.get(t)[1], target(t)) for t in spec["sources"]),
                             (caps.get(t)[1] for t in preserve), lam, eps, device)
    metrics = amplifier_fit_metrics(dw, ((caps.get(t)[1], target(t)) for t in spec["sources"]),
                                    (caps.get(t)[1:] for t in preserve))
    rel = tot = inside = 0.0
    for t in spec["sources"]:  # streamed: captures can be several GB
        d = offset(t)
        rel += float(d.norm() / caps.get(t)[2].norm())
        if u is not None:
            tot += float(d.double().pow(2).sum())
            inside += float((d.to(u.device, torch.float64) @ u).pow(2).sum())
    report = {"schema": 1, "operation": "trigger_amplifier", "part": part, "key": key, **info,
              "spec_sha256": _digest(spec_path), "captures_sha256": [_digest(p) for p in captures],
              "fit_achieved": metrics["achieved"], "preserve_relative_change": metrics.get("preserve_relative_change"),
              "trigger_offset_relative_norm": rel / len(spec["sources"]),
              "scope": "Fit diagnostics on captured conditioning, not renders. Apply as a weight difference: W + dose * dW."}
    if u is not None:
        report["style"] = style_report | {"trigger_offset_energy_in_directions": inside / tot}
    meta = {"matlowai_detangle_amplifier": json.dumps({k: report[k] for k in ("schema", "part", "lam", "eps", "n_source", "n_preserve", "spec_sha256")})}
    return {"part": part, **_publish({key: dw.contiguous()}, out, meta, report, [*captures, spec_path])}


def style_dirs(spec, captures, out, device="cpu"):
    _fresh(out, Path(out).with_suffix(".json"))
    caps = TextCaptures(*captures)
    u, report = _directions(load_spec(spec), caps, device)
    report |= {"schema": 1, "operation": "style_directions", "spec_sha256": _digest(spec),
               "layout": "directions: [out_width, k] float64 columns in the layer's output space"}
    return {"k": u.shape[1], **_publish({"directions": u.cpu().contiguous()}, out,
                                        {"matlowai_detangle_directions": json.dumps({"k": u.shape[1]})}, report, [*captures, spec])}


def _single_diff(path, key=None):
    tensors = load_file(str(path))
    if key is None:
        if len(tensors) != 1:
            raise ValueError(f"{path} holds {len(tensors)} tensors; pass --key")
        key = next(iter(tensors))
    return key, tensors[key]


def amp_split(amplifier, out_style, out_rest, directions=None, directions_from=None, k=4, key=None):
    if (directions is None) == (directions_from is None):
        raise ValueError("Pass exactly one of --directions or --directions-from")
    _fresh(out_style, out_rest, Path(out_style).with_suffix(".json"), Path(out_rest).with_suffix(".json"))
    key, dw = _single_diff(amplifier, key)
    if directions is not None:
        u = load_file(str(directions))["directions"]
        source = {"directions_sha256": _digest(directions)}
    else:
        u = output_directions(_single_diff(directions_from)[1], k)
        source = {"directions_from_sha256": _digest(directions_from), "k": k}
    style, rest = split_amplifier(dw, u)
    base = {"schema": 1, "operation": "amplifier_split", "amplifier_sha256": _digest(amplifier), "key": key, **source,
            "style_energy_share": float(style.double().pow(2).sum() / dw.double().pow(2).sum()),
            "formula": "style = U Uᵀ dW, rest = dW - style (float64, saved float32)"}
    results = {}
    for part, tensor, path in (("style", style, out_style), ("rest", rest, out_rest)):
        meta = {"matlowai_detangle_amplifier": json.dumps({"part": part, "amplifier_sha256": base["amplifier_sha256"]})}
        results[part] = _publish({key: tensor}, path, meta, dict(base, part=part), [amplifier])
    return results


def compose_cmd(adapter, diff, out, key=None):
    tensors = load_file(str(adapter))
    diffs, rows = [], []
    for text in diff:
        path, dose = parse_weighted(text)
        k, t = _single_diff(path, key)
        diffs.append((k, t, dose))
        rows.append({"key": k, "dose": dose, "sha256": _digest(path)})
    packaged = compose(tensors, diffs)
    meta = dict(_metadata(adapter))
    settings = {"schema": 1, "operation": "compose", "adapter_sha256": _digest(adapter), "differences": rows}
    meta[_provenance_key(meta, "matlowai_detangle_compose")] = json.dumps(settings)
    report = settings | {"formula": "out[key] = sum(dose * difference); adapter tensors unchanged",
                         "scope": "A single file a ComfyUI LoRA loader applies at strength 1. Step-scheduled components cannot be packaged."}
    return _publish(packaged, out, meta, report, [adapter, *[parse_weighted(t)[0] for t in diff]])


# ---- Krea 2 (ComfyUI) commands ----
def _krea2(args, vae=False):
    from .krea2 import Krea2
    loras = [parse_weighted(t) for t in (args.get("lora") or [])]
    return Krea2(args["comfyui"], args["diffusion_model"], args["text_encoder"],
                 args.get("vae") if vae else None, loras, args.get("study_pipeline", False)), loras


def collect_text(**args):
    out = Path(args["out"])
    _fresh(out)
    spec = load_spec(args["spec"])
    texts = spec_texts(spec)
    model, loras = _krea2(args)
    records = {}
    for i, text in enumerate(texts, 1):
        records[text] = model.text_conditioning(text)
        if i % 50 == 0:
            print(f"captured {i}/{len(texts)}", flush=True)
    prov = {"spec_sha256": _digest(args["spec"]), "adapters": [{"sha256": _digest(p), "strength": s} for p, s in loras],
            "diffusion_model": Path(args["diffusion_model"]).name, "text_encoder": Path(args["text_encoder"]).name}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_text_captures(out, records, extra=prov)
    return {"prompts": len(records), "output": str(out)}


def collect_layers(**args):
    out = Path(args["out_dir"])
    if out.exists():
        raise FileExistsError("Capture output directory already exists")
    pairs_path = Path(args["pairs"])
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Expected a nonempty pair list")
    for row in pairs:
        row["image"] = str((pairs_path.parent / row["image"]).resolve())
    if len(args.get("lora") or []) != 1:
        raise ValueError("Pass exactly one --lora: the adapter being repaired, active while capturing")
    model, loras = _krea2(args, vae=True)
    with safe_open(loras[0][0], framework="pt", device="cpu") as f:
        keys = list(f.keys())
    result = model.collect_layer_inputs(keys, pairs, noise_seed=args["noise_seed"], capacity=args["capacity"],
                                        reservoir_seed=args["reservoir_seed"], resolution=args["resolution"])
    out.mkdir(parents=True)
    for role, (tensors, _) in result.items():
        save_file(tensors, str(out / f"{role}-fit.safetensors"))
    manifest = {"schema": 1, "runtime": "ComfyUI Krea 2 (delta_surgery_seed.krea2)", "pairs_sha256": _digest(pairs_path),
                "pairs_consumed": len(pairs), "adapter_sha256": _digest(loras[0][0]), "adapter_strength": loras[0][1],
                "noise_seed": args["noise_seed"], "reservoir_seed": args["reservoir_seed"], "capacity": args["capacity"],
                "resolution": args["resolution"], "flow_fractions": [.1, .25, .4, .55, .7, .85],
                "layers": {role: counts for role, (_, counts) in result.items()},
                "images": [{"sha256": _digest(r["image"])} for r in pairs]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"layers": len(result["class"][0]), "output": str(out)}


def render(**args):
    from .schedule import Component, ScheduledDelta, parse_steps
    out = Path(args["out"])
    _fresh(out)
    model, _ = _krea2(args, vae=True)
    schedule = None
    projection = None
    if args.get("project_onto"):
        u = load_file(str(args["project_onto"]))["directions"].to(model.device)
        projection = u @ u.T
    if args.get("schedule"):
        comps = []
        for text in args["schedule"]:
            rest, _, steps = text.rpartition(":")
            path, dose = parse_weighted(rest)
            comps.append(Component(_single_diff(path)[1], dose, parse_steps(steps, args["steps"]), projection))
        schedule = ScheduledDelta(model.dm.txtmlp[3], components=comps)
    image = model.sample(args["prompt"], args["seed"], args["width"], args["height"], args["steps"], schedule=schedule)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    applied = sorted({s for s, on in schedule.calls if on}) if schedule else []
    return {"output": str(out), "scheduled_steps_applied": applied}


def add_parsers(sub):
    p = sub.add_parser("blend", help="Restore a fraction of the source A factors in a block range of a repaired adapter")
    for name in ["source", "repaired", "out"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--blocks", required=True, help="main transformer blocks, inclusive, e.g. 12-15")
    p.add_argument("--fraction", type=float, required=True, help="share of the source A factor, e.g. 0.35")

    p = sub.add_parser("amp-fit", help="Fit the closed-form trigger amplifier from text-conditioning captures")
    p.add_argument("--spec", required=True)
    p.add_argument("--captures", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--part", choices=["full", "style", "rest"], default="full",
                   help="full amplifier, or only its component along / orthogonal to the spec's style directions")
    p.add_argument("--lam", type=float, default=4.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--key", default=AMPLIFIER_KEY)

    p = sub.add_parser("style-directions", help="Measure style directions from the spec's contrast pairs")
    p.add_argument("--spec", required=True)
    p.add_argument("--captures", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("amp-split", help="Split an amplifier into a style dial and the rest (weight space)")
    p.add_argument("--amplifier", required=True)
    p.add_argument("--directions", help="output of style-directions")
    p.add_argument("--directions-from", help="derive directions from an existing style difference (top-k output span)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--key")
    p.add_argument("--out-style", required=True)
    p.add_argument("--out-rest", required=True)

    p = sub.add_parser("compose", help="Package an adapter with weight differences at fixed doses")
    p.add_argument("--adapter", required=True)
    p.add_argument("--diff", required=True, action="append", help="PATH[:DOSE], repeatable")
    p.add_argument("--key")
    p.add_argument("--out", required=True)

    def model_args(p, vae):
        p.add_argument("--comfyui", default=os.environ.get("DETANGLE_COMFYUI"), help="ComfyUI checkout (or DETANGLE_COMFYUI)")
        p.add_argument("--diffusion-model", required=True)
        p.add_argument("--text-encoder", required=True)
        if vae:
            p.add_argument("--vae", required=True)
        p.add_argument("--lora", action="append", help="PATH[:STRENGTH], repeatable; LoRA factors and .diff patches")

    p = sub.add_parser("collect-text", help="Krea 2: capture txtmlp.3 inputs/outputs for every prompt in an amplifier spec")
    model_args(p, False)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("collect-layers", help="Krea 2: capture adapted-layer inputs for `detangle repair`")
    model_args(p, True)
    p.add_argument("--pairs", required=True, help="JSON list of {image, class_prompt, trigger_prompt}")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--capacity", type=int, default=1024)
    p.add_argument("--reservoir-seed", type=int, default=1001)
    p.add_argument("--noise-seed", type=int, default=42)

    p = sub.add_parser("render", help="Krea 2: render one image, optionally with step-scheduled txtmlp.3 components")
    model_args(p, True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1216)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--schedule", action="append", help="PATH:DOSE:STEPS, e.g. style.safetensors:6:3-7 (0-based, inclusive)")
    p.add_argument("--project-onto", help="directions file: apply each scheduled difference as (x dWᵀ) U Uᵀ (the study's form)")
    p.add_argument("--study-pipeline", action="store_true", help="use the scheduling study's in-process attention/RoPE settings")
    p.add_argument("--out", required=True)


COMMANDS = {"blend": blend, "amp-fit": amp_fit, "style-directions": style_dirs, "amp-split": amp_split,
            "compose": compose_cmd, "collect-text": collect_text, "collect-layers": collect_layers, "render": render}
