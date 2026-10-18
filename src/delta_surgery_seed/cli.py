"""Portable file interface. All paths come from the caller; no model downloads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .core import map_layers, matrix, response_metrics, solve


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.metadata() or {}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def up_key(key):
    return key.removesuffix("lora_A.weight") + "lora_B.weight"


def same_tensor(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def check_factors(tensors, mapping):
    for key in mapping:
        a = matrix(tensors[key], key)
        if up_key(key) not in tensors:
            raise ValueError(f"Missing B factor: {up_key(key)}")
        b = matrix(tensors[up_key(key)], up_key(key))
        if b.shape[1] != a.shape[0]:
            raise ValueError(f"Incompatible B factor: {up_key(key)}")


def repair(checkpoint, class_capture, trigger_capture, out, mu=.3, ridge=.001,
           center="zero", method="auto"):
    out = Path(out)
    if out.suffix != ".safetensors":
        raise ValueError("Output must have .safetensors suffix")
    report_path = out.with_suffix(".json")
    if out.exists() or report_path.exists():
        raise FileExistsError("Output or report already exists")
    inputs = {"source": checkpoint, "class_capture": class_capture, "trigger_capture": trigger_capture}
    if out.resolve() in {Path(p).resolve() for p in inputs.values()}:
        raise ValueError("Output cannot replace an input")
    hashes = {name + "_sha256": digest(path) for name, path in inputs.items()}
    tensors = load_file(checkpoint, device="cpu")
    original_metadata = metadata(checkpoint)
    rows = {}
    with safe_open(class_capture, framework="pt", device="cpu") as fc, \
            safe_open(trigger_capture, framework="pt", device="cpu") as ft:
        if set(fc.keys()) != set(ft.keys()):
            raise ValueError("Class and trigger captures have different layer sets")
        mapping = map_layers(tensors, fc.keys())
        check_factors(tensors, mapping)
        for key, name in sorted(mapping.items()):
            a, b = tensors[key], tensors[up_key(key)]
            xc, xt = fc.get_tensor(name), ft.get_tensor(name)
            edited, info = solve(a, xc, xt, mu, ridge, center, method)
            edited = edited.to(a.dtype).contiguous()
            if not torch.isfinite(edited).all():
                raise ValueError(f"Repaired factor overflows checkpoint dtype: {key}")
            info["class_fit"] = response_metrics(a, b, edited, xc)
            info["trigger_fit"] = response_metrics(a, b, edited, xt)
            rows[key] = info
            tensors[key] = edited
    for name, path in inputs.items():
        if digest(path) != hashes[name + "_sha256"]:
            raise ValueError("An input changed during repair")
    added_metadata_key = "matlowai_delta_surgery"
    while added_metadata_key in original_metadata:
        added_metadata_key += "_new"
    report = {"schema": 1, **hashes, "mu": mu, "ridge": ridge, "center": center, "method": method,
              "layers": rows, "added_metadata_key": added_metadata_key,
              "invariants": "Only A factors may change; all keys, shapes, dtypes, B factors, other tensors and original metadata are retained.",
              "scope": "Fit diagnostics use fitting activations; they are not held-out behavior or image scores."}
    output_metadata = dict(original_metadata)
    output_metadata[added_metadata_key] = json.dumps({k: report[k] for k in
                                                    ["schema", "source_sha256", "mu", "ridge", "center", "method"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".delta-surgery-", dir=out.parent) as temporary:
        temp_checkpoint = Path(temporary) / "checkpoint.safetensors"
        temp_report = Path(temporary) / "report.json"
        save_file(tensors, str(temp_checkpoint), metadata=output_metadata)
        report["output_sha256"] = digest(temp_checkpoint)
        write_json(temp_report, report)
        # Hard links publish complete files exclusively, without overwriting a race winner.
        os.link(temp_checkpoint, out)
        try:
            os.link(temp_report, report_path)
        except BaseException:
            out.unlink()
            raise
    return {"layers": len(rows), "output": str(out), "report": str(report_path)}


def evaluate(source, edited, edit_report, class_capture, trigger_capture, out):
    if Path(out).exists():
        raise FileExistsError("Evaluation output already exists")
    report = json.loads(Path(edit_report).read_text(encoding="utf-8"))
    if report.get("schema") != 1:
        raise ValueError("Unsupported repair report")
    if digest(source) != report["source_sha256"] or digest(edited) != report["output_sha256"]:
        raise ValueError("Repair report does not belong to these checkpoints")
    old, new = load_file(source, device="cpu"), load_file(edited, device="cpu")
    if set(old) != set(new):
        raise ValueError("Checkpoint keys differ")
    edited_keys = {k for k in old if k.endswith(".lora_A.weight")}
    if set(report["layers"]) != edited_keys:
        raise ValueError("Report layer set differs from supported A factors")
    for key in old:
        if old[key].shape != new[key].shape or old[key].dtype != new[key].dtype:
            raise ValueError(f"Tensor shape or dtype changed: {key}")
        if key not in edited_keys and not same_tensor(old[key], new[key]):
            raise ValueError(f"A-only invariant failed: {key}")
    source_meta, edited_meta = metadata(source), metadata(edited)
    if any(edited_meta.get(k) != v for k, v in source_meta.items()):
        raise ValueError("Original metadata changed")
    rows, capture_hashes = {}, {}
    fit_hashes = {report["class_capture_sha256"], report["trigger_capture_sha256"]}
    for role, path in [("class", class_capture), ("trigger", trigger_capture)]:
        capture_hashes[role + "_capture_sha256"] = digest(path)
        if capture_hashes[role + "_capture_sha256"] in fit_hashes:
            raise ValueError("Evaluation reuses a fitting file; collect held-out captures")
        with safe_open(path, framework="pt", device="cpu") as f:
            mapping = map_layers(old, f.keys())
            check_factors(old, mapping)
            for key, name in mapping.items():
                rows.setdefault(key, {})[role] = response_metrics(old[key], old[up_key(key)], new[key], f.get_tensor(name))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"schema": 1, "source_sha256": report["source_sha256"],
                     "output_sha256": report["output_sha256"], **capture_hashes, "layers": rows,
                     "holdout": "Different file hashes only. Audit prompts, seeds and capture conditions to establish independence.",
                     "scope": "B A responses on fixed captured inputs; no rerender or causal layer attribution."})
    return {"layers": len(rows), "output": str(out)}


def demo(out):
    """A separable toy with an uncovered direction; this is not an image model."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(42)
    width, rank = 12, 3
    a = torch.randn(rank, width, generator=generator)
    b = torch.randn(5, rank, generator=generator)
    name = "transformer.blocks.0.attn.q"
    key = name + ".lora_A.weight"
    save_file({key: a, up_key(key): b, "adapter.alpha": torch.tensor(float(rank))},
              str(out / "source.safetensors"), metadata={"example": "synthetic only"})
    def batch(count):
        xc = torch.zeros(count, width)
        xt = torch.zeros(count, width)
        xc[:, :4] = torch.randn(count, 4, generator=generator)
        xt[:, 4:8] = torch.randn(count, 4, generator=generator)
        return xc, xt
    for suffix, count in [("fit", 64), ("heldout", 48)]:
        xc, xt = batch(count)
        for role, value in [("class", xc), ("trigger", xt)]:
            save_file({name: value}, str(out / f"{role}-{suffix}.safetensors"))
    results = {}
    for center in ["zero", "source"]:
        repaired = out / f"repaired-{center}.safetensors"
        repair(out / "source.safetensors", out / "class-fit.safetensors", out / "trigger-fit.safetensors",
               repaired, center=center)
        evaluate(out / "source.safetensors", repaired, repaired.with_suffix(".json"),
                 out / "class-heldout.safetensors", out / "trigger-heldout.safetensors",
                 out / f"heldout-{center}.json")
        # Dimensions 8:12 never occur in either fitting role.
        x_unseen = torch.eye(width)[8:]
        results[center] = response_metrics(a, b, load_file(repaired)[key], x_unseen)
    write_json(out / "demo.json", {"kind": "synthetic linear task", "uncovered_response": results,
                                   "lesson": "Zero-centered ridge deletes uncovered directions; source-centered ridge preserves them. This does not establish which prior is better for image generation."})
    return {"output": str(out), "uncovered_response": results}


def main():
    parser = argparse.ArgumentParser(description="Conditional least-squares LoRA repair")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("repair", help="Refit all supported A factors from activation files")
    for name in ["checkpoint", "class-capture", "trigger-capture", "out"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--mu", type=float, default=.3)
    p.add_argument("--ridge", type=float, default=.001)
    p.add_argument("--center", choices=["zero", "source"], default="zero")
    p.add_argument("--method", choices=["auto", "primal", "dual", "historical"], default="auto",
                   help="auto: stable CPU float64; historical: original mixed-precision primal solve (CUDA required for widths <=8192)")
    p = sub.add_parser("evaluate", help="Check A-only invariants and held-out local responses")
    for name in ["source", "edited", "edit-report", "class-capture", "trigger-capture", "out"]:
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("demo", help="Run a synthetic CPU example without private weights")
    p.add_argument("--out", required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    result = {"repair": repair, "evaluate": evaluate, "demo": demo}[command](**arguments)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
