#!/usr/bin/env python3
"""Space-station entanglement probes (E1/E2 + concept overlap).

Same render protocol, adapters and evaluator as studies/2026-10-18-factor-and-block-repair (imported, not copied); new outputs
only under runs/2026-10-19-entanglement-probes/. Nothing in the frozen study is modified.

  python probes.py prepare   # hybrid adapters + cells.json
  python probes.py render    # owns an isolated ComfyUI on GPU0 (DETANGLE_COMFYUI, port DETANGLE_COMFY_PORT), renders all cells, stops it
  python probes.py score     # ArcFace to the 34-reference centroid, to Base's face for the same cell, to Base's named-actress render
  python probes.py report    # tables -> RESULTS.md

E1 phrase ablation: single-edit variants of space_station (V0-V7) and the sci-fi film-still sentence added to arcade/beach (V8/V9).
E2 block hybrids: "source in window W, A-side elsewhere" and the converse, for the 8 Fizgig windows, on space_station + wizard.
C  concept overlap: the subject phrase "a woman" replaced by a named actress, ethnicity descriptions, on space_station + beach.
Arms: base (no LoRA), source @1, a_side @1 / @1.25 (E1), hybrids @1 (E2); seeds 181901/181902; roles target (trigger) / generic.
"""
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-10-18-factor-and-block-repair"
sys.path.insert(0, str(OLD))
import common  # noqa: E402  (protocol, graph template, block map, sha/write helpers)

ROOT = common.ROOT
RUN = common.RUNS / "2026-10-19-entanglement-probes"
SRC_ADAPTERS = common.RUN / "adapters"
COMFY = Path(os.environ.get("DETANGLE_COMFYUI", "ComfyUI"))  # ComfyUI with Krea 2 support; the study pinned commit 87d23b817
PY = Path(os.environ.get("DETANGLE_COMFY_PYTHON", sys.executable))
PORT = int(os.environ.get("DETANGLE_COMFY_PORT", "8188"))
URL = f"http://127.0.0.1:{PORT}"
REUSE = None  # optional cell -> existing image path (set by callers that use the render catalog)
SEEDS = [181901, 181902]
SEEDS_S = [181901, 181902, 181903, 181904]
ACTRESSES = ["Sigourney Weaver", "Jodie Foster", "Natalie Portman", "Meryl Streep", "Julia Roberts"]
# P: wider concept panel (studio portraits by name): actresses with sci-fi/space roles plus non-sci-fi controls
PANEL = ["Sandra Bullock", "Rebecca Ferguson", "Jessica Chastain", "Anne Hathaway", "Noomi Rapace", "Charlize Theron", "Carrie Fisher",
         "Linda Hamilton", "Amy Adams", "Zoe Saldana", "Scarlett Johansson", "Emma Stone", "Kate Winslet", "Nicole Kidman"]
FILM = "Photorealistic live-action science-fiction film still."


def prompts():
    p = common.read_plan()["prompts"]
    ss = p["space_station_target"]["text"].removeprefix("rlsa, ")
    out = {"V0": ss,
           "V1": ss.replace(" " + FILM, ""),
           "V2": ss.replace(FILM, "Photograph."),
           "V3": ss.replace("She has a calm, curious expression.", "She has a relaxed neutral expression."),
           "V4": ss.replace("Soft cinematic key light illuminates her face.", "A soft neutral key light clearly illuminates her face."),
           "V5": ss.replace("She is wearing an opaque silver flight suit with a high closed collar and orange mission patches, with her chest and shoulders fully covered and no helmet.",
                            "She wears a dark blue buttoned shirt with a closed collar."),
           "V6": ss.replace("A circular space-station window shows a softly blurred blue planet behind her, with warm instrument lights in the background.",
                            "A neutral gray photography studio is softly blurred behind her."),
           "V7": "A close-up head-and-shoulders photograph of a woman, facing the camera with a relaxed neutral expression. Her face is large in the frame and unobstructed, with natural skin texture and both eyes sharply in focus. She is wearing an opaque silver flight suit with a high closed collar, covering her chest and shoulders, with no helmet. A circular space-station window showing a blue planet is softly blurred behind her. Soft neutral light illuminates her face.",
           "V8": p["arcade_target"]["text"].removeprefix("rlsa, ") + " " + FILM,
           "V9": p["beach_target"]["text"].removeprefix("rlsa, ") + " " + FILM}
    for k, v in out.items():
        assert k in ("V0", "V7", "V8", "V9") or v != ss, f"{k}: edit did not apply"
    subjects = {"actress": "Sigourney Weaver", "black": "a Black woman", "eastasian": "an East Asian woman",
                "southasian": "a South Asian woman", "latina": "a Latina woman"}
    for case in ("space_station", "beach"):
        base = p[case + "_target"]["text"].removeprefix("rlsa, ")
        anchor = "photograph of a woman"
        assert anchor in base
        for k, s in subjects.items():
            out[f"C_{case}_{k}"] = base.replace(anchor, "photograph of " + s, 1)
    out["wizard"] = p["wizard_workshop_target"]["text"].removeprefix("rlsa, ")
    # S: is the space-station default face specifically the model's Sigourney concept? Named actresses vs default women, Base only.
    studio = p["calibration_studio_generic"]["text"]
    for name in ACTRESSES:
        for scene, text in (("studio", studio), ("space", out["V0"])):
            assert "photograph of a woman" in text
            out[f"S_{name.split()[1].lower()}_{scene}"] = text.replace("photograph of a woman", "photograph of " + name, 1)
    for name in PANEL:
        out["P_" + name.replace(" ", "").lower()] = studio.replace("photograph of a woman", "photograph of " + name, 1)
    for k, text in (("S_default_space", out["V0"]), ("S_default_space_nofilm", out["V1"]), ("S_default_arcade_film", out["V8"]),
                    ("S_default_beach", p["beach_target"]["text"].removeprefix("rlsa, ")), ("S_default_arcade", p["arcade_target"]["text"].removeprefix("rlsa, ")),
                    ("S_default_studio", studio)):
        out[k] = text
    return out


def hybrid_adapters():
    from safetensors.torch import load_file, save_file
    src, a = load_file(SRC_ADAPTERS / "source.safetensors"), load_file(SRC_ADAPTERS / "a_side.safetensors")
    _, mapping = common.engine_for(list(src))
    windows = common.read_plan()["profile_windows"]
    names = []
    for wid, label, blocks in windows:
        tag = wid.replace("..", "-")
        for kind, inside, outside in (("srcin", src, a), ("asidein", a, src)):
            out = {}
            for k in src:
                m = k.removeprefix("diffusion_model.").rsplit(".lora_", 1)[0]
                out[k] = (inside if mapping.get(m) in blocks else outside)[k]
            name = f"hyb_{kind}_{tag}.safetensors"
            save_file(out, RUN / "adapters" / name)
            names.append(name)
    return names


def cells():
    ps = prompts()
    rows = []

    def add(group, pid, arm, adapter, strength):
        for seed in SEEDS:
            for role in ("target", "generic"):
                rows.append({"id": f"{group}_{pid}_{arm}_{role}_{seed}", "group": group, "prompt_id": pid, "arm": arm, "adapter": adapter,
                             "strength": strength, "role": role, "seed": seed,
                             "text": ("rlsa, " if role == "target" else "") + ps[pid]})
    e1_arms = [("base", None, 1.0), ("source", "source.safetensors", 1.0), ("aside100", "a_side.safetensors", 1.0), ("aside125", "a_side.safetensors", 1.25)]
    for v in [f"V{i}" for i in range(10)] + ["wizard"]:
        for arm, ad, s in e1_arms:
            add("E1", v, arm, ad, s)
    for name in sorted(p.name for p in (RUN / "adapters").glob("hyb_*.safetensors")):
        for pid in ("V0", "wizard"):
            add("E2", pid, name.removesuffix(".safetensors"), name, 1.0)
    for pid in [k for k in ps if k.startswith("C_")]:
        for arm, ad, s in e1_arms[:3]:
            add("C", pid, arm, ad, s)
    for pid in [k for k in ps if k.startswith(("S_", "P_"))]:  # Base only, no trigger
        for seed in SEEDS_S:
            rows.append({"id": f"S_{pid}_base_generic_{seed}", "group": "S", "prompt_id": pid, "arm": "base", "adapter": None, "strength": 1.0,
                         "role": "generic", "seed": seed, "text": ps[pid]})
    return rows


def graph(cell):
    g = json.loads((OLD / "graph-template.json").read_text())
    g["pos"]["inputs"]["text"] = cell["text"]
    g["ksampler"]["inputs"]["seed"] = cell["seed"]
    g["save"]["inputs"]["filename_prefix"] = "probes/" + cell["id"]
    if cell["adapter"] is None:
        g.pop("lora")
        g["ksampler"]["inputs"]["model"] = ["unet", 0]
    else:
        g["lora"]["inputs"].update(lora_name=cell["adapter"], strength_model=cell["strength"])
        g["ksampler"]["inputs"]["model"] = ["lora", 0]
    return g


def request(route, payload=None):
    req = urllib.request.Request(URL + route, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def render(gpu="0", port=PORT):
    global URL
    URL = f"http://127.0.0.1:{port}"
    used = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    assert used < 4096, f"GPU{gpu} in use: {used} MiB"
    out = RUN / "render"
    for n in ("images", "input", "comfy_user", "temp"):
        (out / n).mkdir(parents=True, exist_ok=True)
    (RUN / "extra_model_paths.yaml").write_text(f"probes_src:\n  base_path: {common.RUN}\n  loras: adapters\nprobes_new:\n  base_path: {RUN}\n  loras: adapters\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(COMFY / "pinned_site"))
    cmd = [PY, "-u", "main.py", "--port", str(port), "--listen", "127.0.0.1", "--use-pytorch-cross-attention", "--disable-all-custom-nodes", "--disable-auto-launch",
           "--output-directory", out / "images", "--input-directory", out / "input", "--user-directory", out / "comfy_user", "--temp-directory", out / "temp",
           "--database-url", "sqlite:///" + str(out / "comfy_user/comfy.db"), "--extra-model-paths-config", RUN / "extra_model_paths.yaml"]
    rows = json.loads((RUN / "cells.json").read_text())
    manifest = RUN / "render" / "manifest.json"
    done = {r["id"]: r for r in json.loads(manifest.read_text())} if manifest.exists() else {}
    with (RUN / "comfy.log").open("a") as log:
        server = subprocess.Popen([str(x) for x in cmd], cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                assert server.poll() is None, "server exited on startup"
                try:
                    request("/system_stats")
                    break
                except OSError:
                    time.sleep(2)
            for cell in rows:
                if cell["id"] in done:
                    continue
                reused = REUSE(cell) if REUSE else None
                if reused:  # identical cell already rendered elsewhere (render catalog)
                    done[cell["id"]] = {**cell, "file": reused, "reused": True}
                    common.write(manifest, list(done.values()))
                    continue
                pid = request("/prompt", {"prompt": graph(cell)})["prompt_id"]
                while True:
                    h = request("/history/" + pid).get(pid)
                    if h and h["status"].get("status_str") == "error":
                        raise RuntimeError(json.dumps(h["status"])[:2000])
                    if h and h["status"].get("completed"):
                        break
                    time.sleep(0.5)
                img = h["outputs"]["save"]["images"][0]
                done[cell["id"]] = {**cell, "file": str(out / "images" / img["subfolder"] / img["filename"])}
                common.write(manifest, list(done.values()))
                print("rendered", len(done), "/", len(rows), cell["id"], flush=True)
        finally:
            server.terminate()
            server.wait(60)


def score():
    sys.path.insert(0, str(OLD))
    import numpy as np
    from PIL import Image
    import score_and_select
    app, centroid, refs, _ = score_and_select.setup()

    def emb(path):
        faces = app.get(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding
    rows = json.loads((RUN / "render" / "manifest.json").read_text())
    E = {r["id"]: emb(r["file"]) for r in rows}
    by = {(r["group"], r["prompt_id"], r["arm"], r["role"], r["seed"]): r for r in rows}
    out = []
    for r in rows:
        e = E[r["id"]]
        row = {k: r[k] for k in ("id", "group", "prompt_id", "arm", "role", "seed", "file")}
        if e is not None:
            row["likeness"] = float(e @ centroid)
            b = by.get(("E1" if r["group"] == "E2" else r["group"], r["prompt_id"], "base", r["role"], r["seed"]))
            if b is not None and E[b["id"]] is not None:
                row["to_base"] = float(e @ E[b["id"]])
            for case in ("space_station", "beach"):
                act = by.get(("C", f"C_{case}_actress", "base", "generic", r["seed"]))
                if act is not None and E[act["id"]] is not None:
                    row[f"to_named_actress_{case}"] = float(e @ E[act["id"]])
        out.append(row)
    common.write(RUN / "scores.json", out)
    print("scored", len(out))


def report():
    import statistics
    rows = json.loads((RUN / "scores.json").read_text())

    def cell(group, pid, arm, role, key):
        v = [r[key] for r in rows if r["group"] == group and r["prompt_id"] == pid and r["arm"] == arm and r["role"] == role and key in r]
        return statistics.fmean(v) if v else None

    def fmt(v):
        return "–" if v is None else f"{v:.3f}"
    lines = ["# Entanglement probes 2026-10-19", "", "Likeness = ArcFace to the 34-reference centroid (triggered); excess = generic likeness minus Base generic; to_base = face match with Base's render of the same prompt/seed/role; to_actress = face match with Base's render naming the actress (space station, generic, same seed). Means over 2 seeds.", ""]
    lines += ["## E1 phrase ablation (space station unless noted)", "", "| variant | arm | likeness | excess | to_base | to_actress |", "|---|---|---|---|---|---|"]
    for v in [f"V{i}" for i in range(10)] + ["wizard"]:
        bg = cell("E1", v, "base", "generic", "likeness")
        for arm in ("source", "aside100", "aside125", "base"):
            lk, g = cell("E1", v, arm, "target", "likeness"), cell("E1", v, arm, "generic", "likeness")
            lines.append(f"| {v} | {arm} | {fmt(lk)} | {fmt(None if g is None or bg is None else g - bg)} | {fmt(cell('E1', v, arm, 'target', 'to_base'))} | {fmt(cell('E1', v, arm, 'target', 'to_named_actress_space_station'))} |")
    lines += ["", "## E2 block hybrids @1.0", "", "| case | hybrid | likeness | excess | to_base |", "|---|---|---|---|---|"]
    for pid in ("V0", "wizard"):
        for arm in sorted({r["arm"] for r in rows if r["group"] == "E2"}):
            lk, g = cell("E2", pid, arm, "target", "likeness"), cell("E2", pid, arm, "generic", "likeness")
            bg = cell("E1", pid, "base", "generic", "likeness")
            lines.append(f"| {pid} | {arm} | {fmt(lk)} | {fmt(None if g is None or bg is None else g - bg)} | {fmt(cell('E2', pid, arm, 'target', 'to_base'))} |")
    lines += ["", "## Concept overlap", "", "| prompt | arm | likeness (trigger) | likeness (no trigger) | to_actress (trigger) |", "|---|---|---|---|---|"]
    for pid in sorted({r["prompt_id"] for r in rows if r["group"] == "C"}):
        for arm in ("base", "source", "aside100"):
            case = "beach" if "_beach_" in pid else "space_station"
            lines.append(f"| {pid} | {arm} | {fmt(cell('C', pid, arm, 'target', 'likeness'))} | {fmt(cell('C', pid, arm, 'generic', 'likeness'))} | {fmt(cell('C', pid, arm, 'target', 'to_named_actress_' + case))} |")
    (RUN / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "render", "score", "report"])
    a = ap.parse_args()
    if a.command == "prepare":
        (RUN / "adapters").mkdir(parents=True, exist_ok=True)
        names = sorted(p.name for p in (RUN / "adapters").glob("hyb_*.safetensors")) or hybrid_adapters()
        rows = cells()
        common.write(RUN / "cells.json", rows)
        print(len(names), "hybrid adapters,", len(rows), "cells")
    else:
        {"render": render, "score": score, "report": report}[a.command]()
