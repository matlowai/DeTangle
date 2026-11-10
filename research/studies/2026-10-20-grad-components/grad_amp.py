#!/usr/bin/env python3
"""Gradient-scored amp components + constrained specificity optimum (paper track, Subject A; adapters not distributed).

Operating point "style 0": blend_b12_15_35 + amp_nostyle x6. Candidate edits are output-space filters Z of the full amp's
conditioning change, Δc(Z) = Δc1 Z with Δc1 = h dW_ampᵀ (h = txtmlp.3 input). A projector Z = UUᵀ is exactly the amp refit on
the target D UUᵀ (dW_U = UUᵀ dW_amp), so every candidate is a slider of the same fit; the style dial ×k is Z = k U_s U_sᵀ.

Three separate objectives, each differentiated through the full 8-step Euler rollout and a full-image VAE decode (no ROI crop):
  id    ArcFace (buffalo_l w600k_r50, exact torch port of the ONNX graph) cosine to the 34-reference centroid; alignment frozen
  snap  CLIP ViT-L/14 "flat amateur smartphone snapshot" − "cinematic film still with dramatic lighting" (as side_effects.py)
  scene DINOv2-base cosine to Base (same prompt untriggered, same seed), face greyed (box frozen)
G[j, step] = ∂J_j / ∂(txtmlp.3 output at that step)  ->  predicted ΔJ_j(Z) = Σ_steps Σ_tokens <G[j, step], Δc1 Z>.

  python grad_amp.py parity          sampler / decode / metric parity against existing renders and side_effects.json
  python grad_amp.py grads --set fit|held
  python grad_amp.py analyze         component scores, constrained optimum, random-direction controls, step windows
  python grad_amp.py fd              finite-dose checks of predictions in the same pipeline
Runs on GPU0 (CUDA_VISIBLE_DEVICES=0). Outputs: runs/2026-10-20-grad-components/.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-19-peel"))
sys.path.insert(0, str(HERE.parent / "2026-10-20-overnight"))
import peel  # noqa: E402
from peel import probes, common, MODELS  # noqa: E402

DS = common.DS
RUN = DS / "runs/2026-10-20-grad-components"
AMP = DS / "runs/2026-10-20-trigger-amp/adapters"
BLEND = DS / "runs/2026-10-20-overnight/blend_adapters/blend_b12_15_35.safetensors"
OP_STACK = [(BLEND, 1.0), (AMP / "trigamp_nostyle_l4.safetensors", 6.0)]
ARCFACE = Path(os.environ.get("DETANGLE_ARCFACE_ONNX", Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"))
CLIP_DIR = Path(os.environ.get("DETANGLE_CLIP_DIR", "clip-vit-large-patch14"))  # local CLIP ViT-L/14 (Hugging Face format)
TRIG = "rlsa, "
W, H, STEPS = 832, 1216, 8
OBJ = ("id", "snap", "scene")
# gradient contexts: strong-genre film stills disjoint from the six evaluation genres (noir, victorian, samurai, western, superhero, viking)
FIT_GENRES = {
    "jazz": ("in a 1920s jazz club", "a dark velvet dress with a high beaded collar", "A brass band and warm stage lights are softly blurred behind", "Photorealistic live-action 1920s period film still."),
    "medieval": ("in a medieval castle courtyard", "a green wool kirtle laced to the collar", "Stone battlements and banners are softly blurred behind", "Photorealistic live-action medieval epic film still."),
    "airship": ("aboard a steampunk airship", "a brass-buttoned leather flight jacket with a high collar", "Gears, pipes and clouds are softly blurred behind", "Photorealistic live-action steampunk adventure film still."),
    "casino": ("in a 1960s casino", "a black high-necked cocktail dress", "Roulette tables and chandeliers are softly blurred behind", "Photorealistic live-action 1960s spy thriller film still."),
    "wasteland": ("in a post-apocalyptic desert outpost", "a dusty canvas jacket zipped to the collar", "Rusted vehicles and sand dunes are softly blurred behind", "Photorealistic live-action post-apocalyptic film still."),
    "cyberpunk": ("in a neon-lit cyberpunk alley", "a black high-collared synthetic jacket", "Neon signs and rain-slick streets are softly blurred behind", "Photorealistic live-action cyberpunk film still."),
    "rome": ("in a Roman villa courtyard", "a cream linen stola draped up to the collar", "Marble columns and a fountain are softly blurred behind", "Photorealistic live-action ancient Rome epic film still."),
    "gothic": ("in a gothic cathedral at dusk", "a dark hooded cloak with a high collar", "Stained glass and candlelit pillars are softly blurred behind", "Photorealistic live-action gothic horror film still."),
}
FIT_SEEDS, HELD_SEEDS = [182101, 182102], [181901, 181902]


def contexts(which):
    if which == "fit":
        P = {k: f"A close-up head-and-shoulders photograph of a woman {p}, facing the camera. {peel.FACE} She wears {o}. {b} her. {s}" for k, (p, o, b, s) in FIT_GENRES.items()}
        seeds = FIT_SEEDS
    else:
        import overnight
        P = {r["prompt_id"]: r["text"].removeprefix(TRIG) for r in overnight.cells("genres") if r["role"] == "target"}
        seeds = HELD_SEEDS
    return [{"id": f"{k}_{s}", "genre": k, "seed": s, "text": t} for k, t in P.items() for s in seeds]


class OnnxNet:
    """Exact torch execution of the w600k_r50 ONNX graph (Conv / BatchNormalization / PRelu / Add / Flatten / Gemm)."""

    def __init__(self, path, dev):
        import onnx, torch
        from onnx import numpy_helper
        m = onnx.load(str(path))
        self.p = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()).to(dev) for t in m.graph.initializer}
        self.nodes = [(n.op_type, list(n.input), n.output[0], {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in m.graph.node]
        self.inp = m.graph.input[0].name

    def __call__(self, x):
        import torch.nn.functional as F
        v = {self.inp: x}
        get = lambda k: v[k] if k in v else self.p[k]
        for op, ins, out, a in self.nodes:
            i = [get(k) for k in ins]
            if op == "Conv":
                pads = a.get("pads", [0, 0, 0, 0])
                v[out] = F.conv2d(i[0], i[1], i[2] if len(i) > 2 else None, stride=a.get("strides", [1, 1]), padding=pads[:2], dilation=a.get("dilations", [1, 1]), groups=a.get("group", 1))
            elif op == "BatchNormalization":
                v[out] = F.batch_norm(i[0], i[3], i[4], i[1], i[2], False, 0.0, a["epsilon"])
            elif op == "PRelu":
                v[out] = F.prelu(i[0], i[1].flatten())
            elif op == "Add":
                v[out] = i[0] + i[1]
            elif op == "Flatten":
                v[out] = i[0].flatten(a.get("axis", 1))
            elif op == "Gemm":
                v[out] = F.linear(i[0], i[1] if a.get("transB", 0) else i[1].T, i[2])
            else:
                raise ValueError(op)
        return v[out]


class Metrics:
    """Differentiable versions of the study's likeness / snapshot / scene metrics on an image (1,3,H,W) in [0,1] RGB."""

    def __init__(self, dev):
        import torch
        from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
        sys.path.insert(0, str(probes.OLD))
        import score_and_select
        self.torch, self.dev = torch, dev
        self.app, centroid, _, _ = score_and_select.setup()
        self.centroid = torch.tensor(centroid, dtype=torch.float32, device=dev)
        self.arc = OnnxNet(ARCFACE, dev)
        self.clip = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(dev).eval().requires_grad_(False)
        proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
        with torch.no_grad():
            te = lambda t: torch.nn.functional.normalize(self.clip.text_projection(self.clip.text_model(**proc(text=[t], return_tensors="pt", padding=True).to(dev)).pooler_output), dim=-1)[0]
            self.snapdir = te("a flat amateur smartphone snapshot") - te("a cinematic film still with dramatic lighting")
        ip = proc.image_processor
        self.clip_norm = (torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1), torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1))
        dp = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(dev).eval().requires_grad_(False)
        self.dino_norm = (torch.tensor(dp.image_mean, device=dev).view(1, 3, 1, 1), torch.tensor(dp.image_std, device=dev).view(1, 3, 1, 1))
        self.dino_resize = dp.size["shortest_edge"]
        self.dino_crop = dp.crop_size["height"]

    def detect(self, img):
        """Largest face of a (1,3,H,W) image: (kps 5x2, bbox) from the study's insightface app (frozen for the differentiable path)."""
        a = (img[0].permute(1, 2, 0).detach().float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        faces = self.app.get(np.ascontiguousarray(a[:, :, ::-1]))
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return {"kps": f.kps.tolist(), "bbox": f.bbox.tolist(), "emb": f.normed_embedding.tolist()}

    def _resize_crop(self, img, short, crop):
        F = self.torch.nn.functional
        h, w = img.shape[-2:]
        s = short / min(h, w)
        img = F.interpolate(img, size=(round(h * s), round(w * s)), mode="bicubic", align_corners=False, antialias=True)
        h, w = img.shape[-2:]
        t, l = (h - crop) // 2, (w - crop) // 2
        return img[..., t:t + crop, l:l + crop]

    def arcface(self, img, kps):
        from insightface.utils import face_align
        torch = self.torch
        M = face_align.estimate_norm(np.array(kps, dtype=np.float32), 112)  # src -> 112x112 (same as norm_crop)
        Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
        g = np.stack(np.meshgrid(np.arange(112), np.arange(112)), -1).reshape(-1, 2).astype(np.float64)
        src = g @ Mi[:, :2].T + Mi[:, 2]
        Hh, Ww = img.shape[-2:]
        grid = torch.tensor(np.stack([src[:, 0] / (Ww - 1) * 2 - 1, src[:, 1] / (Hh - 1) * 2 - 1], -1).reshape(1, 112, 112, 2), dtype=torch.float32, device=self.dev)
        face = torch.nn.functional.grid_sample(img.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        e = self.arc((face * 255 - 127.5) / 127.5)[0]
        return torch.nn.functional.normalize(e, dim=0)

    def snap(self, img):
        x = (self._resize_crop(img.float(), 224, 224) - self.clip_norm[0]) / self.clip_norm[1]
        e = self.torch.nn.functional.normalize(self.clip.visual_projection(self.clip.vision_model(pixel_values=x).pooler_output), dim=-1)[0]
        return e @ self.snapdir

    def dino_emb(self, img, bbox):
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        m = self.torch.zeros_like(img[:, :1])
        m[..., max(0, int(y0 - 0.15 * h)):int(y1 + 0.1 * h), max(0, int(x0 - 0.15 * w)):int(x1 + 0.15 * w)] = 1
        img = img.float() * (1 - m) + (128 / 255) * m
        x = (self._resize_crop(img, self.dino_resize, self.dino_crop) - self.dino_norm[0]) / self.dino_norm[1]
        return self.torch.nn.functional.normalize(self.dino(pixel_values=x).pooler_output, dim=-1)[0]

    def all(self, img, face, base_dino):
        return {"id": self.arcface(img, face["kps"]) @ self.centroid, "snap": self.snap(img), "scene": self.dino_emb(img, face["bbox"]) @ base_dino}


class Pipe:
    """Krea 2 Turbo (pinned eval ComfyUI) with an additive hook on the txtmlp.3 output: E[step] (tokens x 6144) or None."""

    def __init__(self, stack):
        import torch
        sys.path.insert(0, str(probes.COMFY))
        from comfy.cli_args import args
        args.use_pytorch_cross_attention = True
        args.reserve_vram = args.vram_headroom = 6.0  # CLIP / DINOv2 / insightface share the GPU outside Comfy's model management
        import comfy.sd, comfy.utils, comfy.sample, comfy.samplers, comfy.model_management as mm
        # Legacy ModelPatcher (LoRA baked into the fp8 weights). The render server uses DynamicVRAM (patches applied at cast time):
        # renders differ by ~2-6/255 mean px, and aimdo's reused cast buffers crash autograd (illegal memory access), so the
        # whole gradient study (gradients, finite doses) runs on this one consistent function; final validation renders go through the server.
        self.torch, self.comfy_sample, self.comfy_samplers, self.mm = torch, comfy.sample, comfy.samplers, mm
        mm.in_training = True  # torch RoPE (comfy_kitchen's apply_rope has no autograd); set for every pass so one function is used throughout
        self.dev = mm.get_torch_device()
        self.clip = comfy.sd.load_clip(ckpt_paths=[str(MODELS / "text_encoders/krea2/qwen3vl_4b_fp8_scaled.safetensors")], clip_type=comfy.sd.CLIPType.KREA2)
        self.vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(MODELS / "vae/qwen_image_vae.safetensors")))
        self.base = comfy.sd.load_diffusion_model(str(MODELS / "diffusion_models/krea2/krea2_turbo_fp8_scaled.safetensors"))
        self.unet = self.base
        for path, s in stack:
            self.unet, _ = comfy.sd.load_lora_for_models(self.unet, None, comfy.utils.load_torch_file(str(path)), s, 0)
        self.dm = self.unet.model.diffusion_model
        self.E, self.step, self.h = None, 0, None
        self.dm.txtmlp[3].register_forward_hook(self._hook)
        self.sigmas = comfy.samplers.calculate_sigmas(self.unet.get_model_object("model_sampling"), "simple", STEPS).tolist()

    def _hook(self, mod, inp, out):
        self.h = inp[0].detach()
        e = self.E(self.h, self.step) if callable(self.E) else self.E  # tensor, or f(h, step) -> tensor | None
        return None if e is None else out + e.to(out.dtype)

    def encode(self, text):
        cond = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(text))
        neg = [[self.torch.zeros_like(c[0]), {k: (self.torch.zeros_like(v) if isinstance(v, self.torch.Tensor) else v) for k, v in c[1].items()}] for c in cond]
        return cond, neg

    def sample(self, model, text, seed, E=None, size=(W, H)):
        """KSampler-equivalent run (euler/simple/8/cfg 1). Returns (xs = model-space x_i per step, denoised per step, final latent)."""
        torch = self.torch
        pos, neg = self.encode(text)
        latent = self.comfy_sample.fix_empty_latent_channels(model, torch.zeros([1, 16, size[1] // 8, size[0] // 8]))
        noise = self.comfy_sample.prepare_noise(latent, seed)
        xs, dens = [], []

        def cb(step, x0, x, total):
            self.step = step + 1  # the hook sees the step about to run next
            xs.append(x.detach().clone())
            dens.append(x0.detach().clone())
        self.E, self.step = E, 0
        out = self.comfy_sample.sample(model, noise, STEPS, 1.0, "euler", "simple", pos, neg, latent, denoise=1.0, callback=cb, disable_pbar=True, seed=seed)
        self.E = None
        return xs, dens, out, pos[0][0]

    def decode_grad(self, lat_model):
        """Model-space latent (B,16,h,w) -> image (B,3,H,W) in [0,1], differentiable (full-image decode)."""
        lat = self.unet.model.process_latent_out(lat_model.float())
        self.mm.load_models_gpu([self.vae.patcher], memory_required=18 * 2 ** 30)  # evict the DiT: the decode keeps activations for backward
        out = self.vae.first_stage_model.decode((lat if lat.ndim == 5 else lat.unsqueeze(2)).to(self.vae.vae_dtype))
        return out[:, :, 0].float().add(1).div(2).clamp(0, 1)

    def denoise(self, x, sigma, cond):
        """One model call as the sampler makes it (cfg 1: cond branch only)."""
        s = self.torch.full((x.shape[0],), sigma, device=x.device, dtype=x.dtype)
        return self.unet.model.apply_model(x, s, c_crossattn=cond.to(x.device).expand(x.shape[0], -1, -1))


def checkpoint_blocks(dm, on):
    import torch.utils.checkpoint as cp
    for b in dm.blocks:
        if on:
            f = type(b).forward.__get__(b)
            b.forward = lambda *a, _f=f, **k: cp.checkpoint(_f, *a, use_reentrant=False, **k)
        elif "forward" in b.__dict__:
            del b.forward


def vdec(pipe, lat):
    """VAEDecode node equivalent -> (1,3,H,W) in [0,1]."""
    with pipe.torch.no_grad():  # not inference_mode: VAE weights loaded there could not be reused under autograd
        img = pipe.vae.decode(lat)
    return img.reshape(-1, *img.shape[-3:])[:1].movedim(-1, 1)


def to_img(t):
    from PIL import Image
    return Image.fromarray((t[0].permute(1, 2, 0).detach().float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))


def parity():
    """Reproduce existing server renders (Base and the style-0 operating point, noir 181901) and their metrics."""
    import torch
    from PIL import Image
    side = {(r["step"], r["scene"], r["seed"], r["arm"]): r for r in json.loads((DS / "runs/2026-10-20-overnight/side_effects.json").read_text())}
    ref = {"base": DS / "runs/2026-10-20-overnight/genres/render/images/peel/genres_noir_base_generic_181901_00001_.png",
           "op": DS / "runs/2026-10-20-overnight/style_variants/render/images/peel/style_variants_noir_nostyle6_target_181901_00001_.png"}
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    text = contexts("held")[0]["text"]
    out = {}
    imgs = {}
    for arm, model, t in (("base", pipe.base, text), ("op", pipe.unet, TRIG + text)):
        t0 = time.time()
        xs, dens, lat, cond = pipe.sample(model, t, 181901)
        img = vdec(pipe, lat)
        mine = np.asarray(to_img(img), dtype=np.int16)
        theirs = np.asarray(Image.open(ref[arm]).convert("RGB"), dtype=np.int16)
        out[arm] = {"sample_s": time.time() - t0, "max_abs_px": int(np.abs(mine - theirs).max()), "mean_abs_px": float(np.abs(mine - theirs).mean())}
        imgs[arm] = (img, xs, dens, cond)
        print(arm, out[arm], flush=True)
        if arm == "op":  # recompute each step outside the sampler: same denoised?
            mm_ = pipe.mm
            mm_.load_models_gpu([pipe.unet])
            with torch.no_grad():
                out[arm]["step_recompute_max_rel"] = max(float((pipe.denoise(x.to(pipe.dev), s, cond) - d.to(pipe.dev)).norm() / d.norm()) for x, s, d in zip(xs, pipe.sigmas, dens))
                out[arm]["decode_grad_path_max_abs"] = float((pipe.decode_grad(dens[-1].to(pipe.dev)).cpu() - img.cpu()).abs().max())
    print(json.dumps(out, indent=1), flush=True)
    # metrics: insightface vs torch ArcFace on the PNG files; torch CLIP / DINO vs side_effects.json
    with torch.no_grad():
        for arm, label in (("op", "repair nostyle x6"), ("base", "Base")):
            png = torch.tensor(np.asarray(Image.open(ref[arm]).convert("RGB")), dtype=torch.float32, device=pipe.dev).permute(2, 0, 1)[None] / 255
            f = met.detect(png)
            e = met.arcface(png, f["kps"])
            rec = {"arcface_torch_vs_insightface_cos": float(e.cpu() @ torch.tensor(f["emb"])), "likeness_torch": float(e @ met.centroid), "likeness_insightface": float(np.dot(f["emb"], met.centroid.cpu().numpy())),
                   "snap_torch": float(met.snap(png))}
            step = "style_variants" if arm == "op" else "genres"
            s = side.get((step, "noir", 181901, label))
            rec["snap_side_effects"] = s and s["snapshot"]
            out[arm].update(rec)
            imgs[arm] = imgs[arm] + (png, f)
        bd = met.dino_emb(imgs["base"][4], imgs["base"][5]["bbox"])
        out["op"]["scene_torch"] = float(met.dino_emb(imgs["op"][4], imgs["op"][5]["bbox"]) @ bd)
        out["op"]["scene_side_effects"] = side.get(("style_variants", "noir", 181901, "repair nostyle x6"), {}).get("scene_to_base")
    print(json.dumps(out, indent=1), flush=True)
    # one full backward (memory / time)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    g = grads_one(pipe, met, TRIG + text, 181901, met.dino_emb(imgs["base"][4], imgs["base"][5]["bbox"]).detach())
    out["backward"] = {"seconds": time.time() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "J": g["J"], "G_norm": {j: [float(x.norm()) for x in g["G"][i]] for i, j in enumerate(OBJ)}}
    # directional finite difference along the style dial: (J(+d) - J(-d)) / 2 vs Σ_steps <G, Δc1 U_s U_sᵀ> * d
    dW, comps = amp_dirs(pipe.dev)
    dW, Us = dW.float(), comps["style (dial)"].float()
    fdr = {}
    for d in (0.5, 1.0):
        fn = lambda dose: (lambda h, step: dose * ((h.float() @ dW.T) @ Us) @ Us.T)
        E1 = fn(1.0)(g["h"].to(pipe.dev)[None], 0)[0]
        pred = {j: d * float((g["G"][i].to(pipe.dev).sum(0) * E1).sum()) for i, j in enumerate(OBJ)}
        Js = []
        for sgn in (1, -1):
            _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, 181901, E=fn(sgn * d))
            with torch.no_grad():
                Js.append(met.all(vdec(pipe, lat).to(pipe.dev), g["face"], bd))
        fdr[d] = {"pred": pred, "meas": {j: float(Js[0][j] - Js[1][j]) / 2 for j in OBJ}}
    out["fd_style"] = fdr
    common.write(RUN / "parity.json", out)
    print(json.dumps({k: out[k] for k in ("backward", "fd_style")}, indent=1))


def fdcheck():
    """Is the rollout gradient right, and how far does linearity hold? Central differences along the style dial, small doses, per step window."""
    import torch
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    c = contexts("held")[0]
    with torch.no_grad():
        _, _, blat, _ = pipe.sample(pipe.base, c["text"], c["seed"])
        bimg = vdec(pipe, blat).to(pipe.dev)
        bd = met.dino_emb(bimg, met.detect(bimg)["bbox"])
    g = grads_one(pipe, met, TRIG + c["text"], c["seed"], bd)
    dW, comps = amp_dirs(pipe.dev)
    dW, Us = dW.float(), comps["style (dial)"].float()
    E1 = (((g["h"].to(pipe.dev) @ dW.T) @ Us) @ Us.T)
    out = {}
    for w, steps in {"all": list(range(STEPS)), **WINDOWS}.items():
        for d in (0.03, 0.1, 0.3):
            pred = {j: d * float((g["G"][i][steps].to(pipe.dev).sum(0) * E1).sum()) for i, j in enumerate(OBJ)}
            Js = []
            for sgn in (1, -1):
                fn = lambda h, step, s_=sgn * d: s_ * (((h.float() @ dW.T) @ Us) @ Us.T) if step in steps else None
                _, _, lat, _ = pipe.sample(pipe.unet, TRIG + c["text"], c["seed"], E=fn)
                with torch.no_grad():
                    Js.append(met.all(vdec(pipe, lat).to(pipe.dev), g["face"], bd))
            out[f"{w}@{d}"] = {"pred": pred, "meas": {j: float(Js[0][j] - Js[1][j]) / 2 for j in OBJ}}
            print(w, d, json.dumps(out[f"{w}@{d}"]), flush=True)
    common.write(RUN / "fdcheck.json", out)


def grads_one(pipe, met, text, seed, base_dino, face=None):
    """Rollout VJPs: per objective, ∂J/∂(txtmlp.3 output) for every step. Returns J values, G (3, steps, T, 6144) on CPU, h, face."""
    torch = pipe.torch
    xs, dens, lat, cond = pipe.sample(pipe.unet, text, seed)
    dev = pipe.dev
    with torch.no_grad():
        img0 = vdec(pipe, lat).to(dev)
    face = face or met.detect(img0)
    if face is None:
        return None
    # adjoints at the final latent x_8 = denoised at the last step
    x8 = dens[-1].to(dev).float().requires_grad_(True)
    img = pipe.decode_grad(x8)
    J = met.all(img, face, base_dino)
    adj = [torch.autograd.grad(J[j], x8, retain_graph=k < 2)[0] for k, j in enumerate(OBJ)]
    del img
    a = torch.cat(adj)  # (3, 16, 1, h, w): objective-batched adjoints
    pipe.mm.load_models_gpu([pipe.unet])
    checkpoint_blocks(pipe.dm, True)
    T = cond.shape[1]
    G = torch.zeros(3, STEPS, T, 6144)
    try:
        for i in reversed(range(STEPS)):
            s, sn = pipe.sigmas[i], pipe.sigmas[i + 1]
            x = xs[i].to(dev).float().expand(3, *xs[i].shape[1:]).clone().requires_grad_(True)
            E = torch.zeros(3, T, 6144, device=dev, requires_grad=True)
            pipe.E = E
            den = pipe.denoise(x, s, cond)
            xn = x + (sn - s) / s * (x - den)
            gx, gE = torch.autograd.grad(xn, (x, E), grad_outputs=a)
            pipe.E = None
            G[:, i] = gE.float().cpu()
            a = gx.detach()
            del den, xn, gx, gE, x, E
    finally:
        pipe.E = None
        checkpoint_blocks(pipe.dm, False)
    return {"J": {j: float(J[j]) for j in OBJ}, "G": G, "h": pipe.h[0].float().cpu(), "face": face, "img0": img0.cpu()}


def grads(which):
    import torch
    out_dir = RUN / f"grads_{which}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    log = []
    for c in contexts(which):
        f = out_dir / f"{c['id']}.pt"
        if f.exists():
            continue
        t0 = time.time()
        with torch.no_grad():
            _, _, blat, _ = pipe.sample(pipe.base, c["text"], c["seed"])
            bimg = vdec(pipe, blat).to(pipe.dev)
            bface = met.detect(bimg)
            base_dino = met.dino_emb(bimg, bface["bbox"]) if bface else None
        to_img(bimg).save(out_dir / f"{c['id']}_base.png")
        if base_dino is None:
            print("no face in Base", c["id"], flush=True)
            continue
        g = grads_one(pipe, met, TRIG + c["text"], c["seed"], base_dino)
        if g is None:
            print("no face at operating point", c["id"], flush=True)
            continue
        to_img(g.pop("img0")).save(out_dir / f"{c['id']}_op.png")
        torch.save({**c, **g, "base_face": bface}, f)
        log.append({**c, "J": g["J"], "seconds": time.time() - t0})
        print(c["id"], json.dumps(g["J"]), f"{time.time() - t0:.0f}s", flush=True)
    common.write(out_dir / "log.json", log)


WINDOWS = {"early": [0, 1, 2], "mid": [3, 4, 5], "late": [6, 7]}
K = 256


def amp_dirs(dev):
    """Full amp dW and the output-space directions of the existing components (each component = P dW_amp)."""
    import torch
    from safetensors.torch import load_file
    ld = lambda n: load_file(str(AMP / n))["diffusion_model.txtmlp.3.diff"].to(dev, torch.float64)
    dW = ld("trigamp_aside_l4.safetensors")
    left = lambda M, k: torch.linalg.svd(M, full_matrices=False)[0][:, :k]
    return dW, {"style (dial)": left(dW - ld("trigamp_nostyle_l4.safetensors"), 4), "style_spec": left(dW - ld("trigamp_nostyle_spec_l4.safetensors"), 4),
                "identity_noclothes": left(ld("trigamp_identity_noclothes_l4.safetensors"), 16), "identity": left(ld("trigamp_identity_l4.safetensors"), 16)}


def load_ctx(which, dW):
    import torch
    out = []
    for f in sorted((RUN / f"grads_{which}").glob("*.pt")):
        d = torch.load(f, weights_only=False)
        out.append({"id": d["id"], "J": d["J"], "G": d["G"].to(dW.device, torch.float64), "Dc": d["h"].to(dW.device, torch.float64) @ dW.T})
    return out


def scores(ctx, U):
    """a[c, j, step, k] = predicted ΔJ_j at context c from step `step` for unit dose of the component u_k u_kᵀ (Δc1 u_k u_kᵀ)."""
    import torch
    return torch.stack([torch.einsum("jstk,tk->jsk", c["G"] @ U, c["Dc"] @ U) for c in ctx])


def subspace_pred(ctx, U, dose=1.0):
    """Predicted ΔJ (contexts x 3 x steps) for Z = dose UUᵀ (a projector component, e.g. the style dial)."""
    import torch
    return torch.stack([dose * torch.einsum("jstd,td->js", c["G"], c["Dc"] @ U @ U.T) for c in ctx])


def solve(a_id, cons, e, rho):
    """max mean(a_id) z  s.t.  cons @ z <= 0 (rows), Σ e_k z_k² <= rho²; via y = sqrt(e) z (ball) with SLSQP."""
    from scipy.optimize import minimize
    s = np.sqrt(e)
    c = a_id / s
    A = cons / s
    y0 = np.zeros_like(c)
    r = minimize(lambda y: -c @ y, y0, jac=lambda y: -c, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda y: -A @ y, "jac": lambda y: -A}, {"type": "ineq", "fun": lambda y: rho ** 2 - y @ y, "jac": lambda y: -2 * y}],
                 options={"maxiter": 500, "ftol": 1e-12})
    return r.x / s, r


def analyze():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dW, comps = amp_dirs(dev)
    fit, held = load_ctx("fit", dW), load_ctx("held", dW)
    X = torch.cat([c["Dc"] for c in fit])
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh[:K].T
    e = (S[:K] ** 2 / len(fit)).cpu().numpy()  # per-context energy of each PCA component (unit dose)
    rep = {"n_fit": len(fit), "n_held": len(held), "pca_energy_topK": float((S[:K] ** 2).sum() / (S ** 2).sum()), "J_fit_mean": {j: float(np.mean([c["J"][j] for c in fit])) for j in OBJ},
           "J_held_mean": {j: float(np.mean([c["J"][j] for c in held])) for j in OBJ}}
    style = comps["style (dial)"]
    rho = float(np.sqrt(np.mean([float((2 * c["Dc"] @ style).pow(2).sum()) for c in fit])))  # budget = energy of the style dial x2 (per context)
    rep["budget_rho_style_x2"] = rho
    # named components at the same budget (dose chosen so the change energy equals style x2's)
    named = {}
    for n, U in comps.items():
        en = np.mean([float((c["Dc"] @ U).pow(2).sum()) for c in fit])
        dose = rho / np.sqrt(en)
        pf, ph = subspace_pred(fit, U, dose), subspace_pred(held, U, dose)
        named[n] = {"dose": dose, "fit": {j: float(pf[:, i].sum(-1).mean()) for i, j in enumerate(OBJ)}, "held": {j: float(ph[:, i].sum(-1).mean()) for i, j in enumerate(OBJ)},
                    "held_windows": {w: {j: float(ph[:, i][:, s].sum(-1).mean()) for i, j in enumerate(OBJ)} for w, s in WINDOWS.items()}}
    pf = subspace_pred(fit, torch.eye(6144, dtype=torch.float64, device=dev))
    ph = subspace_pred(held, torch.eye(6144, dtype=torch.float64, device=dev))
    named["full amp x1"] = {"dose": 1.0, "fit": {j: float(pf[:, i].sum(-1).mean()) for i, j in enumerate(OBJ)}, "held": {j: float(ph[:, i].sum(-1).mean()) for i, j in enumerate(OBJ)},
                            "held_windows": {w: {j: float(ph[:, i][:, s].sum(-1).mean()) for i, j in enumerate(OBJ)} for w, s in WINDOWS.items()}}
    rep["named_components_at_budget"] = named
    rep["style_x2_held_per_context"] = {c["id"]: [float(x) for x in subspace_pred([c], style, 2.0)[0].sum(-1)] for c in held}

    def run_basis(U, en, label, sched=False):
        """Constrained optimum in basis U (diagonal Z = U diag(z) Uᵀ); constraints per fit context: Δsnap <= 0, Δscene >= 0."""
        af, ah = scores(fit, U).cpu().numpy(), scores(held, U).cpu().numpy()  # (ctx, 3, steps, K)
        if sched:  # one coefficient vector per window; energy weighted by the window's share of steps
            agg = lambda a: np.concatenate([a[:, :, s].sum(2) for s in WINDOWS.values()], -1)
            ew = np.concatenate([en * len(s) / STEPS for s in WINDOWS.values()])
        else:
            agg, ew = (lambda a: a.sum(2)), en
        F, Hh = agg(af), agg(ah)  # (ctx, 3, nvar)
        cons = np.concatenate([F[:, 1], -F[:, 2]])
        z, r = solve(F[:, 0].mean(0), cons, ew, rho)
        z_free, _ = solve(F[:, 0].mean(0), np.zeros((0, len(ew))), ew, rho)
        ev = lambda M, z: {j: float((M[:, i] @ z).mean()) for i, j in enumerate(OBJ)} | {"snap_max": float((M[:, 1] @ z).max()), "scene_min": float((M[:, 2] @ z).min())}
        return {"label": label, "ok": bool(r.success), "fit": ev(F, z), "held": ev(Hh, z), "held_neg": ev(Hh, -z), "unconstrained_fit": ev(F, z_free), "unconstrained_held": ev(Hh, z_free),
                "spec_ratio_fit": float((F[:, 0].mean(0) @ z) / max(F[:, 0].mean(0) @ z_free, 1e-12))}, z
    res, zs = {}, {}
    res["pca"], zs["pca"] = run_basis(V, e, f"top-{K} PCA of the amp change")
    res["pca_sched"], zs["pca_sched"] = run_basis(V, e, f"top-{K} PCA, per-window coefficients (early/mid/late)", sched=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    ctrl = []
    for r_ in range(5):
        Q = torch.linalg.qr(torch.randn(6144, K, generator=g, dtype=torch.float64))[0].to(dev)
        en = np.mean([((c["Dc"] @ Q) ** 2).sum(0).cpu().numpy() for c in fit], 0)
        ctrl.append(run_basis(Q, en, "random 6144-d directions")[0])
        R = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=torch.float64))[0].to(dev)
        Q2 = V @ R
        en2 = np.mean([((c["Dc"] @ Q2) ** 2).sum(0).cpu().numpy() for c in fit], 0)
        ctrl.append(run_basis(Q2, en2, "random rotation of the PCA span")[0])
    res["controls"] = ctrl
    rep["optimum"] = res
    # gradient geometry: per-context step-summed amp-coordinate gradients (energy-whitened), cosines and principal angles id vs snap/scene
    A = scores(fit, V).sum(2).cpu().numpy() / np.sqrt(e)  # (ctx, 3, K)
    nrm = lambda M: M / np.linalg.norm(M, axis=-1, keepdims=True)
    cos = lambda x, y: float(np.mean(np.sum(nrm(x) * nrm(y), -1)))
    pa = lambda X_, Y_: np.linalg.svd(np.linalg.qr(X_.T)[0].T @ np.linalg.qr(Y_.T)[0], compute_uv=False).tolist()
    rnd = np.random.default_rng(0)
    rep["gradient_geometry"] = {"mean_cos_id_snap": cos(A[:, 0], A[:, 1]), "mean_cos_id_scene": cos(A[:, 0], A[:, 2]), "mean_cos_snap_scene": cos(A[:, 1], A[:, 2]),
                                "mean_cos_id_across_contexts": float(np.mean([np.dot(nrm(A[i, 0]), nrm(A[k, 0])) for i in range(len(A)) for k in range(len(A)) if i < k])),
                                "principal_cos_id_vs_snap": pa(A[:, 0], A[:, 1]), "principal_cos_id_vs_scene": pa(A[:, 0], A[:, 2]),
                                "principal_cos_random_baseline": pa(rnd.standard_normal((len(A), K)), rnd.standard_normal((len(A), K)))}
    torch.save({"V": V.cpu(), "z": zs, "e": e, "rho": rho}, RUN / "optimum.pt")
    common.write(RUN / "analysis.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "style_x2_held_per_context"}, indent=1))


def fd(which="held"):
    """Finite doses in the same pipeline: measured ΔJ (frozen alignment, as differentiated; and re-detected face) vs predicted."""
    import torch
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    dW, comps = amp_dirs(pipe.dev)
    dW = dW.float()
    opt = torch.load(RUN / "optimum.pt", weights_only=False)
    V = opt["V"].to(pipe.dev).float()
    zs = {k: torch.tensor(v, dtype=torch.float32, device=pipe.dev) for k, v in opt["z"].items()}
    Us = comps["style (dial)"].float()
    win = {i: w for w, s in enumerate(WINDOWS.values()) for i in s}
    nK = V.shape[1]

    def static(z):
        return lambda h, step: ((h.float() @ dW.T) @ V * z) @ V.T

    def proj(U, dose):
        return lambda h, step: dose * ((h.float() @ dW.T) @ U) @ U.T

    def sched(z):
        return lambda h, step: ((h.float() @ dW.T) @ V * z[win[step] * nK:(win[step] + 1) * nK]) @ V.T
    arms = {"style_x2": proj(Us, 2.0), "opt": static(zs["pca"]), "opt_x2": static(2 * zs["pca"]), "opt_neg": static(-zs["pca"]), "opt_sched": sched(zs["pca_sched"]),
            "amp_x1": proj(torch.eye(6144, device=pipe.dev), 1.0)}
    out_dir = RUN / f"fd_{which}"
    out_dir.mkdir(exist_ok=True)
    res = json.loads((out_dir / "fd.json").read_text()) if (out_dir / "fd.json").exists() else {}
    for f in sorted((RUN / f"grads_{which}").glob("*.pt")):
        d = torch.load(f, weights_only=False)
        G, c = d["G"].to(pipe.dev), d["h"].to(pipe.dev).float() @ dW.T
        from PIL import Image
        bimg = torch.tensor(np.asarray(Image.open(RUN / f"grads_{which}" / f"{d['id']}_base.png")), dtype=torch.float32, device=pipe.dev).permute(2, 0, 1)[None] / 255
        with torch.no_grad():
            bd = met.dino_emb(bimg, d["base_face"]["bbox"])
        for arm, fn in arms.items():
            key = f"{d['id']}|{arm}"
            if key in res:
                continue
            steps = torch.stack([fn(d["h"].to(pipe.dev)[None], i)[0] for i in range(STEPS)])  # (steps, T, 6144) edit actually applied
            pred = torch.einsum("jstd,std->j", G, steps).tolist()
            _, _, lat, _ = pipe.sample(pipe.unet, TRIG + d["text"], d["seed"], E=fn)
            img = vdec(pipe, lat).to(pipe.dev)
            with torch.no_grad():
                Jf = {j: float(v) for j, v in met.all(img, d["face"], bd).items()}
                face = met.detect(img)
                Jd = {j: float(v) for j, v in met.all(img, face, bd).items()} if face else None
            to_img(img).save(out_dir / f"{d['id']}_{arm}.png")
            res[key] = {"pred": dict(zip(OBJ, pred)), "meas_frozen": {j: Jf[j] - d["J"][j] for j in OBJ}, "meas_redetect": Jd and {j: Jd[j] - d["J"][j] for j in OBJ}}
            common.write(out_dir / "fd.json", res)
            print(key, json.dumps(res[key]), flush=True)
    summ = {}
    for arm in arms:
        rs = [v for k, v in res.items() if k.endswith("|" + arm)]
        summ[arm] = {f"{m}_{j}": float(np.mean([r[m][j] for r in rs if r[m]])) for m in ("pred", "meas_frozen", "meas_redetect") for j in OBJ}
        summ[arm]["corr_pred_meas_id"] = float(np.corrcoef([r["pred"]["id"] for r in rs], [r["meas_frozen"]["id"] for r in rs])[0, 1]) if len(rs) > 2 else None
    common.write(out_dir / "summary.json", summ)
    print(json.dumps(summ, indent=1))


# Seed-averaged finite effects (the rollout gradient is non-smooth in steps 0-2 per seed; fdcheck.json).
# Each component is an output-space projector U Uᵀ applied to the amp's conditioning change, scaled to the style dial's x1 change energy.
# Per context (genre x seed): Base, operating point, and one antithetic pair ±D·Σ z_k s_k Δc1 U_k U_kᵀ with random signs z;
# regressing (J+ − J−)/2D on z estimates every component's average first-order finite effect.
RESP_SEEDS = {"fit": list(range(182101, 182109)), "held": list(range(181905, 181909))}
D_RESP = 0.2
N_PCA = 12


def h_of(pipe, text):
    import torch
    cond = pipe.encode(text)[0][0][0]
    pipe.mm.load_models_gpu([pipe.unet])
    with torch.no_grad():
        return pipe.dm.txtmlp[:3](pipe.dm.txtfusion(pipe.dm._unpack_context(cond.to(pipe.dev, torch.bfloat16))))[0].float()


def build_components(pipe, dW, comps):
    import torch
    hs = [h_of(pipe, TRIG + c["text"]) for c in contexts("fit")[::len(FIT_SEEDS)]]
    D1 = [h @ dW.T for h in hs]
    Vh = torch.linalg.svd(torch.cat(D1), full_matrices=False)[2]
    U = {f"pca{k + 1}": Vh[k:k + 1].T for k in range(N_PCA)} | {n: u.float() for n, u in comps.items()}
    g = torch.Generator(device="cpu").manual_seed(1)
    for i in range(2):  # controls: a random direction inside the top-256 PCA span, and a random direction in the full 6144-d space
        v = Vh[:256].T @ torch.randn(256, 1, generator=g).to(pipe.dev)
        U[f"rand_span{i + 1}"] = v / v.norm()
        v = torch.randn(6144, 1, generator=g).to(pipe.dev)
        U[f"rand_full{i + 1}"] = v / v.norm()
    en = {n: float(np.mean([float((d @ u).pow(2).sum()) for d in D1])) for n, u in U.items()}
    unit = en["style (dial)"]
    return {"names": list(U), "U": {n: u.cpu() for n, u in U.items()}, "scale": {n: (unit / en[n]) ** 0.5 for n in U}, "energy": en}


def measure(met, img, bd):
    face = met.detect(img)
    if face is None:
        return None
    import torch
    with torch.no_grad():
        return {j: float(v) for j, v in met.all(img, face, bd).items()}


def resp(which):
    import torch
    from PIL import Image
    out_dir = RUN / f"resp_{which}"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    dW, comps = amp_dirs(pipe.dev)
    dW = dW.float()
    cf = RUN / "resp_fit" / "components.pt"
    if not cf.exists():
        (RUN / "resp_fit").mkdir(exist_ok=True)
        torch.save(build_components(pipe, dW, comps), cf)
    C = torch.load(cf, weights_only=False)
    names = C["names"]
    SU = [(C["scale"][n] * C["U"][n]).to(pipe.dev) for n in names]
    U_ = [C["U"][n].to(pipe.dev) for n in names]
    rec_f = out_dir / "records.jsonl"
    done = {json.loads(l)["id"] for l in rec_f.read_text().splitlines()} if rec_f.exists() else set()
    texts = {c["genre"]: c["text"] for c in contexts(which)}
    for gi, (genre, text) in enumerate(texts.items()):
        for seed in RESP_SEEDS[which]:
            cid = f"{genre}_{seed}"
            if cid in done:
                continue
            t0 = time.time()
            _, _, blat, _ = pipe.sample(pipe.base, text, seed)
            bimg = vdec(pipe, blat).to(pipe.dev)
            bface = met.detect(bimg)
            if bface is None:
                print("no face in Base", cid, flush=True)
                continue
            with torch.no_grad():
                bd = met.dino_emb(bimg, bface["bbox"])
            z = np.random.default_rng(seed * 100 + gi).choice([-1.0, 1.0], len(names))
            M = sum(float(zk) * su @ u.T for zk, su, u in zip(z, SU, U_))
            row = {"id": cid, "genre": genre, "seed": seed, "z": z.tolist()}
            for arm, sg in (("J0", 0.0), ("Jp", 1.0), ("Jm", -1.0)):
                fn = None if sg == 0 else (lambda h, step, s_=sg * D_RESP: s_ * ((h.float() @ dW.T) @ M))
                _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, seed, E=fn)
                img = vdec(pipe, lat).to(pipe.dev)
                row[arm] = measure(met, img, bd)
                to_img(img).save(out_dir / "img" / f"{cid}_{arm}.jpg", quality=88)
            to_img(bimg).save(out_dir / "img" / f"{cid}_base.jpg", quality=88)
            with rec_f.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(cid, {k: row[k] for k in ("J0", "Jp", "Jm")}, f"{time.time() - t0:.0f}s", flush=True)


def resp_analyze():
    import torch
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    names = C["names"]
    rows = [json.loads(l) for l in (RUN / "resp_fit" / "records.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["J0"] and r["Jp"] and r["Jm"]]
    Z = np.array([r["z"] for r in rows])
    Y = {j: np.array([(r["Jp"][j] - r["Jm"][j]) / (2 * D_RESP) for r in rows]) for j in OBJ}
    curv = {j: float(np.mean([(r["Jp"][j] + r["Jm"][j]) / 2 - r["J0"][j] for r in rows])) for j in OBJ}
    rng = np.random.default_rng(0)
    beta = {j: np.linalg.lstsq(Z, Y[j], rcond=None)[0] for j in OBJ}
    boot = {j: np.array([np.linalg.lstsq(Z[i], Y[j][i], rcond=None)[0] for i in (rng.integers(0, len(rows), len(rows)) for _ in range(500))]) for j in OBJ}
    se = {j: boot[j].std(0) for j in OBJ}
    real = [k for k, n in enumerate(names) if not n.startswith("rand")]

    def opt(b, rho=2.0, idx=real):
        """max b_id·z  s.t. b_snap·z <= 0, b_scene·z >= 0, ||z|| <= rho  (z in style-dial energy units, components in idx)."""
        from scipy.optimize import minimize
        c, s, e = b["id"][idx], b["snap"][idx], b["scene"][idx]
        r = minimize(lambda z: -c @ z, np.zeros(len(idx)), jac=lambda z: -c, method="SLSQP",
                     constraints=[{"type": "ineq", "fun": lambda z: -s @ z}, {"type": "ineq", "fun": lambda z: e @ z}, {"type": "ineq", "fun": lambda z: rho ** 2 - z @ z}])
        z = np.zeros(len(names))
        z[idx] = r.x
        return z
    zs = opt(beta)
    st = names.index("style (dial)")
    z_style = np.zeros(len(names)); z_style[st] = 2.0
    pred = lambda z, b=beta: {j: float(b[j] @ z) for j in OBJ}
    perm = []
    for _ in range(200):  # permutation null: shuffle responses across contexts, re-fit, re-optimise
        p = rng.permutation(len(rows))
        bp = {j: np.linalg.lstsq(Z, Y[j][p], rcond=None)[0] for j in OBJ}
        perm.append(pred(opt(bp), bp)["id"])
    boot_opt = [pred(zs, {j: boot[j][i] for j in OBJ}) for i in range(len(boot["id"]))]
    rep = {"n": len(rows), "dose_per_component": D_RESP, "even_part_mean (curvature at the pair's total energy)": curv,
           "components": {n: {j: f"{beta[j][k]:+.4f} ± {se[j][k]:.4f}" for j in OBJ} | {"scale": C["scale"][n]} for k, n in enumerate(names)},
           "optimum_z": dict(zip(names, np.round(zs, 3).tolist())), "optimum_pred": pred(zs),
           "optimum_pred_bootstrap_sd": {j: float(np.std([b[j] for b in boot_opt])) for j in OBJ},
           "style_x2_pred": pred(z_style), "permutation_null_opt_id": {"mean": float(np.mean(perm)), "p95": float(np.quantile(perm, 0.95))}}
    common.write(RUN / "resp_fit" / "analysis.json", rep)
    np.save(RUN / "resp_fit" / "z_opt.npy", zs)
    print(json.dumps(rep, indent=1))


def resp_validate():
    """Held-out genres, fresh seeds: measured ΔJ of the optimum vs the style dial x2 vs a norm-matched random combination."""
    import torch
    out_dir = RUN / "resp_held"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    dW = amp_dirs(pipe.dev)[0].float()
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    names = C["names"]
    zo = np.load(RUN / "resp_fit" / "z_opt.npy")
    zs = np.zeros(len(names)); zs[names.index("style (dial)")] = 2.0
    real = np.array([not n.startswith("rand") for n in names])
    zr = np.where(real, np.random.default_rng(5).standard_normal(len(names)), 0.0)
    zr *= np.linalg.norm(zo) / np.linalg.norm(zr)
    single = lambda n, v: np.where(np.array(names) == n, v, 0.0)
    # the full optimum barely beats the permutation null (analysis.json), so the two components that stand out individually are tested directly
    arms = {"style_x2": zs, "opt": zo, "rand_matched": zr, "style_spec_x2": single("style_spec", 2.0), "pca4_x2": single("pca4", 2.0)}
    ops = {a: sum(float(z[k]) * C["scale"][n] * C["U"][n].to(pipe.dev) @ C["U"][n].to(pipe.dev).T for k, n in enumerate(names) if z[k] != 0) for a, z in arms.items()}
    rec_f = out_dir / "records.jsonl"
    done = {json.loads(l)["id"] for l in rec_f.read_text().splitlines()} if rec_f.exists() else set()
    for genre, text in {c["genre"]: c["text"] for c in contexts("held")}.items():
        for seed in RESP_SEEDS["held"]:
            cid = f"{genre}_{seed}"
            if cid in done:
                continue
            _, _, blat, _ = pipe.sample(pipe.base, text, seed)
            bimg = vdec(pipe, blat).to(pipe.dev)
            bface = met.detect(bimg)
            if bface is None:
                continue
            with torch.no_grad():
                bd = met.dino_emb(bimg, bface["bbox"])
            row = {"id": cid, "genre": genre, "seed": seed}
            for arm in ["J0"] + list(arms):
                fn = None if arm == "J0" else (lambda h, step, M=ops[arm]: (h.float() @ dW.T) @ M)
                _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, seed, E=fn)
                img = vdec(pipe, lat).to(pipe.dev)
                row[arm] = measure(met, img, bd)
                to_img(img).save(out_dir / "img" / f"{cid}_{arm}.jpg", quality=88)
            with rec_f.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(cid, json.dumps({k: v for k, v in row.items() if k.startswith(("J0", "style", "opt", "rand"))}), flush=True)
    rows = [json.loads(l) for l in rec_f.read_text().splitlines()]
    summ = {}
    for arm in arms:
        ok = [r for r in rows if r["J0"] and r[arm]]
        d = {j: np.array([r[arm][j] - r["J0"][j] for r in ok]) for j in OBJ}
        summ[arm] = {j: f"{d[j].mean():+.4f} ± {d[j].std(ddof=1) / np.sqrt(len(ok)):.4f}" for j in OBJ} | {"n": len(ok)}
    common.write(out_dir / "summary.json", summ)
    print(json.dumps(summ, indent=1))


WIN_COMPS = {"style (dial)": 1.0, "style_spec": 1.0, "pca4": 1.0}


def windows():
    """Where in sampling does each component act? Antithetic ±1 (style-dial energy units) applied only in early / mid / late steps,
    seed-averaged over the 8 fit genres x 4 seeds. Question: identity late, snapshot style early -> a scheduled amp a(t)Δ_id + b(t)Δ_style."""
    import torch
    out_dir = RUN / "windows"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    dW = amp_dirs(pipe.dev)[0].float()
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    ops = {n: d * C["scale"][n] * C["U"][n].to(pipe.dev) @ C["U"][n].to(pipe.dev).T for n, d in WIN_COMPS.items()}
    rec_f = out_dir / "records.jsonl"
    done = {json.loads(l)["id"] for l in rec_f.read_text().splitlines()} if rec_f.exists() else set()
    for genre, text in {c["genre"]: c["text"] for c in contexts("fit")}.items():
        for seed in RESP_SEEDS["fit"][:4]:
            cid = f"{genre}_{seed}"
            if cid in done:
                continue
            t0 = time.time()
            _, _, blat, _ = pipe.sample(pipe.base, text, seed)
            bimg = vdec(pipe, blat).to(pipe.dev)
            bface = met.detect(bimg)
            if bface is None:
                continue
            with torch.no_grad():
                bd = met.dino_emb(bimg, bface["bbox"])
            row = {"id": cid, "genre": genre, "seed": seed}
            _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, seed)
            row["J0"] = measure(met, vdec(pipe, lat).to(pipe.dev), bd)
            for n, M in ops.items():
                for w, steps in WINDOWS.items():
                    for sg in (1, -1):
                        fn = lambda h, step, M=M, steps=steps, sg=sg: sg * ((h.float() @ dW.T) @ M) if step in steps else None
                        _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, seed, E=fn)
                        img = vdec(pipe, lat).to(pipe.dev)
                        row[f"{n}|{w}|{'+' if sg > 0 else '-'}"] = measure(met, img, bd)
                        if seed == RESP_SEEDS["fit"][0]:
                            to_img(img).save(out_dir / "img" / f"{cid}_{n.split()[0]}_{w}_{'p' if sg > 0 else 'm'}.jpg", quality=88)
            with rec_f.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(cid, f"{time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in rec_f.read_text().splitlines()]
    summ = {}
    for n in ops:
        for w in WINDOWS:
            odd = [{j: (r[f"{n}|{w}|+"][j] - r[f"{n}|{w}|-"][j]) / 2 for j in OBJ} for r in rows if r.get(f"{n}|{w}|+") and r.get(f"{n}|{w}|-")]
            summ[f"{n} | {w}"] = {j: f"{np.mean([o[j] for o in odd]):+.4f} ± {np.std([o[j] for o in odd], ddof=1) / np.sqrt(len(odd)):.4f}" for j in OBJ} | {"n": len(odd)}
    common.write(out_dir / "summary.json", summ)
    print(json.dumps(summ, indent=1))


DOSE_COMPS = ["style (dial)", "style_spec", "pca4", "pca2", "identity_noclothes", "rand_span1"]
DOSES = [1.0, 2.0, 3.0]


DOSE_SETS = {"base": (DOSE_COMPS, DOSES, "dose"),
             "far": (["pca2", "identity_noclothes", "pca2+identity_noclothes"], [5.0, 8.0], "dose_far"),  # do the snapshot-free directions keep climbing?
             "sched": (["style (dial)@late", "style_spec@late", "pca4@late", "style (dial)@mid+late"], [3.0, 6.0], "dose_sched"),  # late-only amps: likeness without snapshot?
             "sched_held": (["style (dial)", "style (dial)@mid+late", "style (dial)@late"], [3.0, 6.0], "dose_sched_held")}  # held-out genres, fresh seeds
DOSE_CTX = {"sched_held": ("held", RESP_SEEDS["held"])}


def dose(which="base"):
    """Exchange-rate curves: likeness gain vs snapshot / scene cost per component at practical doses (style-dial energy units),
    one-sided vs the operating point, seed-averaged over the 8 fit genres x 2 seeds. Same curve for every component = conditioning-space ceiling."""
    import torch
    comps, doses, sub = DOSE_SETS[which]
    out_dir = RUN / sub
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    pipe = Pipe(OP_STACK)
    met = Metrics(pipe.dev)
    dW = amp_dirs(pipe.dev)[0].float()
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    P = {n: sum(C["scale"][m] * C["U"][m].to(pipe.dev) @ C["U"][m].to(pipe.dev).T for m in n.split("@")[0].split("+")) for n in comps}
    active = {n: (sorted(i for w in n.split("@")[1].split("+") for i in WINDOWS[w]) if "@" in n else list(range(STEPS))) for n in comps}
    rec_f = out_dir / "records.jsonl"
    done = {json.loads(l)["id"] for l in rec_f.read_text().splitlines()} if rec_f.exists() else set()
    cset, cseeds = DOSE_CTX.get(which, ("fit", FIT_SEEDS))
    for genre, text in {c["genre"]: c["text"] for c in contexts(cset)}.items():
        for seed in cseeds:
            cid = f"{genre}_{seed}"
            if cid in done:
                continue
            t0 = time.time()
            _, _, blat, _ = pipe.sample(pipe.base, text, seed)
            bimg = vdec(pipe, blat).to(pipe.dev)
            bface = met.detect(bimg)
            if bface is None:
                continue
            with torch.no_grad():
                bd = met.dino_emb(bimg, bface["bbox"])
            row = {"id": cid, "genre": genre, "seed": seed}
            _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, seed)
            row["J0"] = measure(met, vdec(pipe, lat).to(pipe.dev), bd)
            for n, M in P.items():
                for d in doses:
                    _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, seed, E=lambda h, step, M=M, d=d, act=active[n]: d * ((h.float() @ dW.T) @ M) if step in act else None)
                    img = vdec(pipe, lat).to(pipe.dev)
                    row[f"{n}|{d:g}"] = measure(met, img, bd)
                    if d == 2.0:
                        to_img(img).save(out_dir / "img" / f"{cid}_{n.split()[0]}_x2.jpg", quality=88)
            with rec_f.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(cid, f"{time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in rec_f.read_text().splitlines()]
    summ = {}
    for n in comps:
        for d in doses:
            k = f"{n}|{d:g}"
            ok = [r for r in rows if r["J0"] and r.get(k)]
            dd = {j: np.array([r[k][j] - r["J0"][j] for r in ok]) for j in OBJ}
            summ[k] = {j: f"{dd[j].mean():+.4f} ± {dd[j].std(ddof=1) / np.sqrt(len(ok)):.4f}" for j in OBJ} | {"n": len(ok)}
    common.write(out_dir / "summary.json", summ)
    print(json.dumps(summ, indent=1))


def split_amp(name):
    """Style-dial split of another amp fit: nostyle = (I − U_s U_sᵀ) dW, style = U_s U_sᵀ dW (U_s = the style dial's 4 output directions). Adapter not distributed."""
    import torch
    from safetensors.torch import load_file, save_file
    k = "diffusion_model.txtmlp.3.diff"
    dW = load_file(str(AMP / f"trigamp_{name}_l4.safetensors"))[k].double()
    Us = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)["U"]["style (dial)"].double()
    S = Us @ (Us.T @ dW)
    for tag, d in (("nostyle", dW - S), ("style", S)):
        save_file({k: d.float().contiguous()}, AMP / f"trigamp_{name}_{tag}_l4.safetensors", metadata={"note": f"Subject A derived (not distributed): style-dial split of {name}."})
    print("style share of weight energy", float(S.norm() ** 2 / dW.norm() ** 2))


TILE_ARMS = {"style 0": None, "static x3": ("style (dial)", 3.0, None), "steps 3-7 x6": ("style (dial)", 6.0, "mid+late"), "steps 6-7 x6": ("style (dial)", 6.0, "late")}


def tiles_generic():
    """Untriggered (bleed) counterparts of the scheduled-amp tiles: same arms, prompt without the trigger. Saved as <genre>_<seed>_<arm>_generic.png."""
    import torch
    out_dir = RUN / "tiles_sched"
    pipe = Pipe(OP_STACK)
    dW = amp_dirs(pipe.dev)[0].float()
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    U = C["U"]["style (dial)"].to(pipe.dev)
    P = C["scale"]["style (dial)"] * U @ U.T
    for genre, text in {c["genre"]: c["text"] for c in contexts("held")}.items():
        for seed in (181905, 181906):
            for arm, spec in TILE_ARMS.items():
                fn = None
                if spec:
                    _, d, w = spec
                    act = [i for x in w.split("+") for i in WINDOWS[x]] if w else list(range(STEPS))
                    fn = lambda h, step, d=d, act=act: d * ((h.float() @ dW.T) @ P) if step in act else None
                _, _, lat, _ = pipe.sample(pipe.unet, text, seed, E=fn)
                to_img(vdec(pipe, lat)).save(out_dir / f"{genre}_{seed}_{arm.replace(' ', '_')}_generic.png")
            print(genre, seed, flush=True)


def tiles():
    """Visual check of the scheduled amp (local only): six held-out genres x 181905-06, Base / style 0 / static x3 / scheduled, one PNG each + a contact sheet per genre."""
    import torch
    from PIL import Image, ImageDraw
    out_dir = RUN / "tiles_sched"
    out_dir.mkdir(exist_ok=True)
    pipe = Pipe(OP_STACK)
    dW = amp_dirs(pipe.dev)[0].float()
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    U = C["U"]["style (dial)"].to(pipe.dev)
    P = C["scale"]["style (dial)"] * U @ U.T
    for genre, text in {c["genre"]: c["text"] for c in contexts("held")}.items():
        for seed in (181905, 181906):
            ims = []
            _, _, lat, _ = pipe.sample(pipe.base, text, seed)
            ims.append(("Base", to_img(vdec(pipe, lat))))
            for arm, spec in TILE_ARMS.items():
                fn = None
                if spec:
                    _, d, w = spec
                    act = [i for x in w.split("+") for i in WINDOWS[x]] if w else list(range(STEPS))
                    fn = lambda h, step, d=d, act=act: d * ((h.float() @ dW.T) @ P) if step in act else None
                _, _, lat, _ = pipe.sample(pipe.unet, TRIG + text, seed, E=fn)
                ims.append((arm, to_img(vdec(pipe, lat))))
            for arm, im in ims:
                im.save(out_dir / f"{genre}_{seed}_{arm.replace(' ', '_')}.png")
            w_, h_ = ims[0][1].size
            sheet = Image.new("RGB", (len(ims) * w_ // 3, h_ // 3 + 28), "white")
            dr = ImageDraw.Draw(sheet)
            for i, (arm, im) in enumerate(ims):
                sheet.paste(im.resize((w_ // 3, h_ // 3)), (i * w_ // 3, 28))
                dr.text((i * w_ // 3 + 6, 6), arm, fill="black")
            sheet.save(out_dir / f"sheet_{genre}_{seed}.jpg", quality=90)
            print(genre, seed, flush=True)


SRC_AD = DS / "runs/2026-10-18-factor-and-block-repair/adapters"
HL_ARMS = {"naive": [(SRC_AD / "source.safetensors", 1.0)], "a_side": [(SRC_AD / "a_side.safetensors", 1.0)],
           "repair": [(BLEND, 1.0), (AMP / "trigamp_aside_l4.safetensors", 6.0)], "style_0": OP_STACK, "static_x3": OP_STACK, "sched_x6": OP_STACK}


def highlights():
    """Landing-page cohort (renders of the undistributed adapter stay local): space station x 181905-08, triggered + untriggered, every headline arm
    through the same in-process path: Base, original LoRA, A-side, repair x6, style dial 0, static style x3, style steps 3-7 x6."""
    import torch
    out_dir = RUN / "highlights"
    out_dir.mkdir(exist_ok=True)
    pipe = Pipe(OP_STACK)
    import comfy.sd, comfy.utils  # on sys.path once Pipe has added the pinned checkout
    dW = amp_dirs(pipe.dev)[0].float()
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    U = C["U"]["style (dial)"].to(pipe.dev)
    P = C["scale"]["style (dial)"] * U @ U.T
    text = overnight_scenes()["space_station"]
    models = {"base": pipe.base}
    for arm, stack in HL_ARMS.items():
        if stack is OP_STACK:
            models[arm] = pipe.unet
            continue
        m = pipe.base
        for path, s in stack:
            m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), s, 0)
        models[arm] = m
    E = {"static_x3": lambda h, step: 3.0 * ((h.float() @ dW.T) @ P),
         "sched_x6": lambda h, step: 6.0 * ((h.float() @ dW.T) @ P) if step >= 3 else None}
    for seed in (181905, 181906, 181907, 181908):
        for arm, model in models.items():
            for role in ("target", "generic"):
                if arm == "base" and role == "target":
                    continue
                _, _, lat, _ = pipe.sample(model, (TRIG if role == "target" else "") + text, seed, E=E.get(arm))
                to_img(vdec(pipe, lat)).save(out_dir / f"{seed}_{arm}_{role}.png")
        print(seed, flush=True)


GROUP_ARMS = ["base", "naive", "repair", "style_0", "static_x3", "sched_x6", "late_x6"]


def group():
    """Group shots (renders of the undistributed adapter stay local): does each arm turn the other women into her, and do named actresses keep their faces?
    overnight.MIXED prompts (cafe, crew, Ferguson, Rapace) with the trigger, 1216x832, seeds 181905-08, one in-process path for every arm."""
    import torch
    out_dir = RUN / "group"
    out_dir.mkdir(exist_ok=True)
    pipe = Pipe(OP_STACK)
    import comfy.sd, comfy.utils, overnight
    dW = amp_dirs(pipe.dev)[0].float()
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    U = C["U"]["style (dial)"].to(pipe.dev)
    P = C["scale"]["style (dial)"] * U @ U.T
    models = {"base": pipe.base}
    for arm in GROUP_ARMS[1:]:
        stack = HL_ARMS.get(arm, OP_STACK)
        if stack is OP_STACK:
            models[arm] = pipe.unet
            continue
        m = pipe.base
        for path, s in stack:
            m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), s, 0)
        models[arm] = m
    E = {"static_x3": lambda h, step: 3.0 * ((h.float() @ dW.T) @ P),
         "sched_x6": lambda h, step: 6.0 * ((h.float() @ dW.T) @ P) if step >= 3 else None,
         "late_x6": lambda h, step: 6.0 * ((h.float() @ dW.T) @ P) if step >= 6 else None}
    for pid, text in overnight.MIXED.items():
        for seed in (181905, 181906, 181907, 181908):
            for arm in GROUP_ARMS:
                f = out_dir / f"{pid}_{seed}_{arm}.png"
                if not f.exists():
                    _, _, lat, _ = pipe.sample(models[arm], TRIG + text, seed, E=E.get(arm), size=(1216, 832))
                    to_img(vdec(pipe, lat)).save(f)
            print(pid, seed, flush=True)


def group_score():
    """Per face: likeness to her centroid and each named actress's concept (same evaluator and concepts as overnight.py mixed)."""
    import torch
    from PIL import Image
    sys.path.insert(0, str(probes.OLD))
    import score_and_select
    app, centroid, _, _ = score_and_select.setup()
    Cc = {n: np.array(c) for n, c in json.loads((peel.RUN / "scores.json").read_text())["concepts"].items()}
    rows = []
    for f in sorted((RUN / "group").glob("*.png")):
        pid, seed, arm = f.stem.split("_", 2)
        fs = app.get(np.asarray(Image.open(f).convert("RGB"))[:, :, ::-1])
        rows.append({"prompt": pid, "seed": int(seed), "arm": arm, "file": str(f),
                     "faces": [{"bbox": [float(x) for x in g.bbox], "likeness": float(g.normed_embedding @ centroid), "emb": g.normed_embedding.tolist(),
                                "concepts": {n: float(g.normed_embedding @ c) for n, c in Cc.items() if n in ("Rebecca Ferguson", "Noomi Rapace")}} for g in fs]})
    base = {(r["prompt"], r["seed"]): r for r in rows if r["arm"] == "base"}
    summ = {}
    for pid in dict.fromkeys(r["prompt"] for r in rows):
        for arm in GROUP_ARMS:
            subj, strangers, match, named = [], [], [], []
            for r in [x for x in rows if x["prompt"] == pid and x["arm"] == arm and x["faces"]]:
                fs = sorted(r["faces"], key=lambda x: -x["likeness"])
                subj.append(fs[0]["likeness"])
                strangers += [x["likeness"] for x in fs[1:]]
                b = base.get((pid, r["seed"]))
                if b and arm != "base":
                    for x in fs[1:]:
                        cx = ((x["bbox"][0] + x["bbox"][2]) / 2, (x["bbox"][1] + x["bbox"][3]) / 2)
                        near = min(b["faces"], key=lambda y: abs((y["bbox"][0] + y["bbox"][2]) / 2 - cx[0]) + abs((y["bbox"][1] + y["bbox"][3]) / 2 - cx[1]), default=None)
                        if near:
                            match.append(float(np.dot(x["emb"], near["emb"])))
                if pid in ("ferguson", "rapace"):
                    who = {"ferguson": "Rebecca Ferguson", "rapace": "Noomi Rapace"}[pid]
                    named.append(max(x["concepts"][who] for x in fs))
            mean = lambda v: round(float(np.mean(v)), 3) if v else None
            summ[f"{pid} | {arm}"] = {"subject": mean(subj), "strangers": len(strangers), "strangers_ge_0.35": sum(v >= .35 for v in strangers),
                                      "stranger_likeness": mean(strangers), "stranger_match_to_base": mean(match), "named_actress": mean(named)}
    for r in rows:
        for x in r["faces"]:
            x.pop("emb")
    common.write(RUN / "group" / "scores.json", rows)
    common.write(RUN / "group" / "summary.json", summ)
    print(json.dumps(summ, indent=1))


ID_YOUNG = ["Florence Pugh", "Anya Taylor-Joy", "Saoirse Ronan", "Emma Stone", "Jennifer Lawrence", "Kristen Stewart"]
ID_OLD = ["Meryl Streep", "Sigourney Weaver", "Jodie Foster", "Michelle Yeoh", "Tilda Swinton", "Viola Davis"]
ID_TEMPLATE = "A close-up head-and-shoulders studio photograph of {who}, facing the camera. " + peel.FACE + " She wears a plain dark crew-neck sweater. Soft even studio light and a plain grey background."
ID_BLOCKS, ID_STEPS, ID_SEEDS = [4, 8, 12, 15, 20, 27], [2, 5, 7], [182201, 182202, 182203, 182204]


def idmap():
    """Where does identity live in the image path? Per face, the image-token hidden state after selected blocks at early/late steps,
    averaged over the face region (two passes per render: find the face, then re-sample the same seed with capture hooks).
    Subjects: 6 celebrities near Subject A's age, 6 older, generic 30- and 60-year-old women, and Subject A (repair, original LoRA). Renders stay local."""
    import torch
    from PIL import Image
    out_dir = RUN / "idmap"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    pipe = Pipe(OP_STACK)
    import comfy.sd, comfy.utils
    met = Metrics(pipe.dev)
    models = {"base": pipe.base}
    for arm in ("repair", "naive"):
        m = pipe.base
        for path, s in HL_ARMS[arm]:
            m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), s, 0)
        models[arm] = m
    subjects = [(n, "young celebrity", "base", ID_TEMPLATE.format(who=n)) for n in ID_YOUNG] + [(n, "older celebrity", "base", ID_TEMPLATE.format(who=n)) for n in ID_OLD]
    subjects += [("woman, 30", "generic 30", "base", ID_TEMPLATE.format(who="a 30-year-old woman")), ("woman, 60", "generic 60", "base", ID_TEMPLATE.format(who="a 60-year-old woman"))]
    subjects += [("Subject A (repair)", "subject", "repair", TRIG + ID_TEMPLATE.format(who="a woman")), ("Subject A (original LoRA)", "subject", "naive", TRIG + ID_TEMPLATE.format(who="a woman"))]
    gw, gh = W // 16, H // 16
    cap = {}

    def hook(i):
        def f(mod, inp, out):
            if pipe.step in ID_STEPS and cap.get("mask") is not None:
                img = out[0, -gw * gh:].float()
                cap[(i, pipe.step)] = ((cap["mask"] @ img) / cap["mask"].sum()).cpu(), img.mean(0).cpu()
        return f
    hs = [pipe.dm.blocks[i].register_forward_hook(hook(i)) for i in ID_BLOCKS]
    rec_f = out_dir / "records.pt"
    recs = torch.load(rec_f, weights_only=False) if rec_f.exists() else []
    done = {(r["who"], r["seed"]) for r in recs}
    try:
        for who, group, arm, text in subjects:
            for seed in ID_SEEDS:
                if (who, seed) in done:
                    continue
                cap.clear()
                _, _, lat, _ = pipe.sample(models[arm], text, seed)
                img = vdec(pipe, lat).to(pipe.dev)
                face = met.detect(img)
                if face is None:
                    print("no face", who, seed, flush=True)
                    continue
                x0, y0, x1, y1 = face["bbox"]
                m = torch.zeros(gh, gw)
                m[max(0, int(y0 // 16)):int(y1 // 16) + 1, max(0, int(x0 // 16)):int(x1 // 16) + 1] = 1
                cap["mask"] = m.flatten()[None].to(pipe.dev)
                pipe.sample(models[arm], text, seed)
                cap.pop("mask")
                tag = f"{who.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')}_{seed}"
                to_img(img).save(out_dir / "img" / f"{tag}.jpg", quality=88)
                recs.append({"who": who, "group": group, "arm": arm, "seed": seed, "arcface": face["emb"],
                             "face": {k: v[0] for k, v in cap.items()}, "whole": {k: v[1] for k, v in cap.items()}})
                torch.save(recs, rec_f)
                print(who, seed, flush=True)
    finally:
        for h in hs:
            h.remove()


def idmap_analyze():
    """Identity probes on face activations vs ArcFace: leave-one-seed-out nearest-centroid accuracy, Fisher identity subspace with
    random / shuffled-label controls, age axis vs identity subspace, and where Subject A's offset from a generic same-age woman lands."""
    import torch
    recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
    celebs = [r for r in recs if "celebrity" in r["group"]]
    rng = np.random.default_rng(0)

    def nc_acc(X, y, seeds):
        hits = 0
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if seeds[j] != seeds[i]]
            cents = {c: X[[j for j in tr if y[j] == c]].mean(0) for c in set(y)}
            sims = {c: float(X[i] @ v / (np.linalg.norm(X[i]) * np.linalg.norm(v) + 1e-9)) for c, v in cents.items()}
            hits += max(sims, key=sims.get) == y[i]
        return hits / len(y)

    def fisher(X, y, U):
        Z = X @ U
        mu = Z.mean(0)
        cls = sorted(set(y))
        between = sum(sum(1 for t in y if t == c) * np.sum((Z[[t == c for t in y]].mean(0) - mu) ** 2) for c in cls)
        within = sum(np.sum((Z[[t == c for t in y]] - Z[[t == c for t in y]].mean(0)) ** 2) for c in cls)
        return between / max(within, 1e-9)
    y = [r["who"] for r in celebs]
    seeds = [r["seed"] for r in celebs]
    A = np.array([r["arcface"] for r in celebs])
    rep = {"n": len(recs), "n_celebrity_renders": len(celebs), "chance": 1 / len(set(y)), "arcface_nn_acc": nc_acc(A, y, seeds), "blocks": {}}
    gen30 = [r for r in recs if r["group"] == "generic 30"]
    subj = [r for r in recs if r["group"] == "subject"]
    for (b, st) in sorted(celebs[0]["face"]):
        Xc = np.array([r["face"][(b, st)].numpy().ravel() for r in celebs], dtype=np.float64)
        mu = Xc.mean(0)
        sd = Xc.std(0) + 1e-6
        Z = (Xc - mu) / sd
        Zw = (np.array([r["whole"][(b, st)].numpy().ravel() for r in celebs], dtype=np.float64) - mu) / sd
        means = np.array([Z[[t == c for t in y]].mean(0) for c in sorted(set(y))])
        Ub = np.linalg.svd(means - means.mean(0), full_matrices=False)[2][:8].T
        Ur = np.linalg.qr(rng.standard_normal((Z.shape[1], 8)))[0]
        ys = list(rng.permutation(y))
        means_s = np.array([Z[[t == c for t in ys]].mean(0) for c in sorted(set(ys))])
        Us = np.linalg.svd(means_s - means_s.mean(0), full_matrices=False)[2][:8].T
        young = np.array([t in ID_YOUNG for t in y])
        age = Z[~young].mean(0) - Z[young].mean(0)
        age /= np.linalg.norm(age)
        e = {"nn_acc_face": nc_acc(Z, y, seeds), "nn_acc_whole_image": nc_acc(Zw, y, seeds),
             "fisher_identity_subspace": fisher(Z, y, Ub), "fisher_random_subspace": fisher(Z, y, Ur), "fisher_shuffled_labels": fisher(Z, ys, Us),
             "age_axis_in_identity_subspace": float(np.sum((age @ Ub) ** 2)), "age_nn_acc": nc_acc(Z, ["young" if v else "old" for v in young], seeds)}
        if subj and gen30:
            G = (np.array([r["face"][(b, st)].numpy().ravel() for r in gen30]) - mu) / sd
            for arm in ("repair", "naive"):
                S = (np.array([r["face"][(b, st)].numpy().ravel() for r in subj if r["arm"] == arm]) - mu) / sd
                d = S.mean(0) - G.mean(0)
                e[f"subject_{arm}_offset_in_identity_subspace"] = float(np.sum((d @ Ub) ** 2) / np.sum(d ** 2))
                e[f"subject_{arm}_seed_consistency"] = float(np.mean([np.dot(a - G.mean(0), c - G.mean(0)) / (np.linalg.norm(a - G.mean(0)) * np.linalg.norm(c - G.mean(0))) for i_, a in enumerate(S) for c in S[i_ + 1:]]))
            e["random_8d_share"] = 8 / Z.shape[1]
        rep["blocks"][f"block {b} step {st}"] = {k: round(v, 4) for k, v in e.items()}
    common.write(RUN / "idmap" / "analysis.json", rep)
    print(json.dumps(rep, indent=1))


STEER_BLOCKS = [12, 15, 20]


def her_directions(dev):
    """Per block: unit direction of Subject A's face activation (repair) minus a generic 30-year-old woman's (idmap, steps 5 and 7), and the
    generic woman's projection on it (the level strangers are steered back to)."""
    import torch
    recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
    out = {}
    for b in STEER_BLOCKS:
        her = torch.stack([r["face"][(b, s)].ravel() for r in recs if r["group"] == "subject" and r["arm"] == "repair" for s in (5, 7)]).mean(0)
        gen = torch.stack([r["face"][(b, s)].ravel() for r in recs if r["group"] == "generic 30" for s in (5, 7)]).mean(0)
        u = (her - gen) / (her - gen).norm()
        out[b] = (u.to(dev), float(gen @ u), float(her @ u))
    return out


def group_steer():
    """Group shots: (1) diagnose — projection of every face onto Subject A's direction, per block and step (Base and repair);
    (2) fix — on steps 3-7, after blocks 12/15/20, set every non-subject face's projection back to the generic-woman level
    (oracle face masks from the unsteered render; the subject = the face most like her). Renders stay local."""
    import torch
    from PIL import Image
    out_dir = RUN / "group_steer"
    out_dir.mkdir(exist_ok=True)
    pipe = Pipe(OP_STACK)
    import comfy.sd, comfy.utils, overnight
    met = Metrics(pipe.dev)
    models = {"base": pipe.base}
    m = pipe.base
    for path, s in HL_ARMS["repair"]:
        m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), s, 0)
    models["repair"] = m
    D = her_directions(pipe.dev)
    GW, GH = 1216 // 16, 832 // 16
    st = {"masks": None, "subj": None, "steer": False, "diag": []}

    def mask_of(b):
        x0, y0, x1, y1 = b
        mm = torch.zeros(GH, GW, device=pipe.dev)
        mm[max(0, int(y0 // 16) - 1):int(y1 // 16) + 2, max(0, int(x0 // 16) - 1):int(x1 // 16) + 2] = 1
        return mm.flatten()

    def hook(bi):
        def f(mod, inp, out):
            if st["masks"] is None:
                return None
            u, c_gen, _ = D[bi]
            img = out[0, -GW * GH:]
            proj = img.float() @ u
            st["diag"].append({"block": bi, "step": pipe.step, "faces": [float((mk * proj).sum() / mk.sum()) for mk in st["masks"]]})
            if st["steer"] and pipe.step >= 3:
                out = out.clone()
                for k, mk in enumerate(st["masks"]):
                    if k != st["subj"]:
                        delta = (proj - c_gen) * mk  # tokens of a stranger's face, back to the generic level along u
                        out[0, -GW * GH:] -= (delta[:, None] * u[None, :]).to(out.dtype)
                return out
            return None
        return f
    hs = [pipe.dm.blocks[b].register_forward_hook(hook(b)) for b in STEER_BLOCKS]
    rows = []
    try:
        for pid, text in overnight.MIXED.items():
            for seed in (181905, 181906):
                for arm in ("base", "repair"):
                    ref = RUN / "group" / f"{pid}_{seed}_{arm}.png"
                    fs = met.app.get(np.asarray(Image.open(ref).convert("RGB"))[:, :, ::-1])
                    if not fs:
                        continue
                    lk = [float(f.normed_embedding @ met.centroid.cpu().numpy()) for f in fs]
                    st["masks"], st["subj"] = [mask_of(f.bbox) for f in fs], int(np.argmax(lk))
                    for steer in ((False, True) if arm == "repair" else (False,)):
                        st["steer"], st["diag"] = steer, []
                        _, _, lat, _ = pipe.sample(models[arm], TRIG + text, seed, size=(1216, 832))
                        name = f"{pid}_{seed}_{arm}{'_steer' if steer else ''}"
                        if steer:
                            to_img(vdec(pipe, lat)).save(out_dir / f"{name}.png")
                        rows.append({"name": name, "pid": pid, "seed": seed, "arm": arm, "steer": steer, "subject_index": st["subj"],
                                     "ref_bboxes": [f.bbox.tolist() for f in fs], "ref_likeness": lk, "diag": st["diag"]})
                    print(pid, seed, arm, flush=True)
    finally:
        st["masks"] = None
        for h in hs:
            h.remove()
    common.write(out_dir / "diag.json", rows)
    # scoring of the steered renders vs the unsteered ones (per face: likeness to her)
    summ = {}
    for r in rows:
        if not r["steer"]:
            continue
        fs = met.app.get(np.asarray(Image.open(out_dir / f"{r['name']}.png").convert("RGB"))[:, :, ::-1])
        lk = sorted([float(f.normed_embedding @ met.centroid.cpu().numpy()) for f in fs], reverse=True)
        before = sorted(r["ref_likeness"], reverse=True)
        summ[r["name"]] = {"before_subject": round(before[0], 3), "before_strangers": [round(v, 3) for v in before[1:]],
                           "after_subject": round(lk[0], 3) if lk else None, "after_strangers": [round(v, 3) for v in lk[1:]]}
    common.write(out_dir / "summary.json", summ)
    print(json.dumps(summ, indent=1))


ROUTE_ARMS = ["routed_all", "routed_text", "routed_image"]


def routed():
    """Where must the LoRA act for identity, and where does it spill? Base weights + the repair's block LoRA (blocks 0-27, rank 32) added by
    forward hooks only at text-token or image-token positions (single-stream DiT: both token types share the block weights).
    routed_all = parity with the normal repair; routed_text = block LoRA on text tokens + txtfusion LoRA + amp (identity can reach faces only
    through attention); routed_image = block LoRA on image tokens only, no text-side change. Scenes: a single portrait + the four group shots."""
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    out_dir = RUN / "routed"
    out_dir.mkdir(exist_ok=True)
    pipe = Pipe(OP_STACK)
    import comfy.sd, overnight
    met = Metrics(pipe.dev)
    sd = load_file(str(BLEND))
    amp = load_file(str(AMP / "trigamp_aside_l4.safetensors"))
    text_side = {k: v for k, v in sd.items() if ".txtfusion." in k} | {"diffusion_model.txtmlp.3.diff": 6.0 * amp["diffusion_model.txtmlp.3.diff"]}
    models = {"base": pipe.base}
    models["text_side"], _ = comfy.sd.load_lora_for_models(pipe.base, None, text_side, 1.0, 0)
    lora = {}
    for k, v in sd.items():
        if ".blocks." in k and k.endswith("lora_A.weight"):
            path = k.removeprefix("diffusion_model.").removesuffix(".lora_A.weight")
            lora[path] = (v.to(pipe.dev, torch.bfloat16), sd[k.replace("lora_A", "lora_B")].to(pipe.dev, torch.bfloat16))
    st = {"route": None, "T": 0}

    def hook(path):
        A, B = lora[path]

        def f(mod, inp, out):
            if st["route"] is None:
                return None
            x = inp[0]
            d = (x.to(A.dtype) @ A.T) @ B.T
            if st["route"] == "text":
                d[:, st["T"]:] = 0
            elif st["route"] == "image":
                d[:, :st["T"]] = 0
            return out + d.to(out.dtype)
        return f
    hs = [pipe.dm.get_submodule(p).register_forward_hook(hook(p)) for p in lora]
    single = TRIG + ID_TEMPLATE.format(who="a woman")
    scenes = {"portrait": (single, (W, H))} | {k: (TRIG + v, (1216, 832)) for k, v in overnight.MIXED.items()}
    arms = {"routed_all": ("text_side", "all"), "routed_text": ("text_side", "text"), "routed_image": ("base", "image")}
    rows = []
    try:
        for pid, (text, size) in scenes.items():
            T = pipe.encode(text)[0][0][0].shape[1]
            for seed in (181905, 181906):
                for arm, (model, route) in arms.items():
                    st["route"], st["T"] = route, T
                    f = out_dir / f"{pid}_{seed}_{arm}.png"
                    _, _, lat, _ = pipe.sample(models[model], text, seed, size=size)
                    st["route"] = None
                    to_img(vdec(pipe, lat)).save(f)
                    fs = met.app.get(np.asarray(Image.open(f).convert("RGB"))[:, :, ::-1])
                    lk = sorted([float(g.normed_embedding @ met.centroid.cpu().numpy()) for g in fs], reverse=True)
                    rows.append({"scene": pid, "seed": seed, "arm": arm, "likeness_sorted": [round(v, 3) for v in lk]})
                    print(pid, seed, arm, [round(v, 2) for v in lk], flush=True)
    finally:
        for h in hs:
            h.remove()
    common.write(out_dir / "rows.json", rows)


BIND_PAIRS = [("Meryl Streep", "Florence Pugh"), ("Viola Davis", "Saoirse Ronan"), ("Michelle Yeoh", "Emma Stone"), ("Sigourney Weaver", "Anya Taylor-Joy")]
BIND_TWO = "A medium photograph of {a} standing next to {b} on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography."
BIND_RLSA = {"chair": "A medium photograph of rlsa sitting on a chair while a different woman stands behind her, both facing the camera with their faces clearly visible and in focus. Natural indoor light.",
             "pugh": "A medium photograph of rlsa standing next to Florence Pugh on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography.",
             "cafe": TRIG + "A medium photograph of a woman sitting at a cafe table between two other women, all three facing the camera with their faces clearly visible and in focus. Natural daylight, candid documentary photograph."}
BIND_STEPS = [2, 5, 7]


BIND2_PAIRS = BIND_PAIRS + [("Jodie Foster", "Kristen Stewart"), ("Tilda Swinton", "Jennifer Lawrence"), ("Meryl Streep", "Viola Davis"), ("Florence Pugh", "Saoirse Ronan")]
BIND2_DESC = {"hair": ("A medium photograph of a red-haired woman standing next to a blonde woman on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography.", {"A": "a red-haired woman", "B": "a blonde woman"}),
              "age": ("A medium photograph of an old woman standing next to a young woman on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography.", {"A": "an old woman", "B": "a young woman"})}
BIND2_RLSA = {"pugh_prefix": ("repair", TRIG + "A medium photograph of a woman standing next to Florence Pugh on a red carpet, both facing the camera with their faces clearly visible and in focus. Evening event photography.", {"rlsa": "rlsa", "Pugh": "Florence Pugh"}),
              "pugh_inline_original": ("naive", BIND_RLSA["pugh"], {"rlsa": "rlsa", "Pugh": "Florence Pugh"}),
              "chair_original": ("naive", BIND_RLSA["chair"], {"rlsa": "rlsa", "other": "a different woman"})}


def bind_heads(which="v1"):
    """Name -> face binding heads. Per block and head (late steps), the attention mass each face's tokens put on each name span
    (exact per-head softmax over all keys, recomputed from the block's q/k path). Two-celebrity prompts (Base) define binding heads
    (own name minus other name); rlsa prompts (repair) show whether her face and the strangers attend to the rlsa tokens. Renders stay local."""
    import torch
    from PIL import Image
    import torch.nn.functional as F
    out_dir = RUN / ("bind" if which == "v1" else "bind2")
    out_dir.mkdir(exist_ok=True)
    pipe = Pipe(OP_STACK)
    import comfy.sd, comfy.utils
    from comfy.ldm.krea2.model import apply_rope  # on sys.path once Pipe has added the pinned checkout
    met = Metrics(pipe.dev)
    models = {"base": pipe.base}
    for arm in ("repair", "naive"):
        m = pipe.base
        for path, s in HL_ARMS[arm]:
            m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), s, 0)
        models[arm] = m
    recs = torch.load(RUN / "idmap" / "records.pt", weights_only=False)
    cent = {}
    for r in recs:
        if "celebrity" in r["group"]:
            cent.setdefault(r["who"], []).append(np.array(r["arcface"]))
    cent = {k: (lambda v: v / np.linalg.norm(v))(np.mean(v, 0)) for k, v in cent.items()}
    GW, GH = 1216 // 16, 832 // 16
    st = {"faces": None, "T": 0, "rows": []}

    def span(text, phrase):
        """Token positions of `phrase` inside the conditioning of `text` (prefix-length difference)."""
        n = lambda t: pipe.encode(t)[0][0][0].shape[1]
        i = text.index(phrase)
        full = n(text)
        tail = n(text[i + len(phrase):]) if text[i + len(phrase):].strip() else 0
        head_tail = n(text[i:]) if text[i:].strip() else 0
        return list(range(full - head_tail, full - tail))

    def pre(bi):
        attn = pipe.dm.blocks[bi].attn

        def f(mod, args):
            if st["faces"] is None or pipe.step not in BIND_STEPS:
                return None
            x, freqs = args[0], args[1]
            with torch.no_grad():
                q = attn.wq(x).unflatten(-1, (attn.heads, attn.headdim)).transpose(1, 2)
                k = attn.wk(x).unflatten(-1, (attn.kvheads, attn.headdim)).transpose(1, 2)
                q, k = attn.qknorm(q, k)
                q, k = apply_rope(q, k, freqs)
                k = k.repeat_interleave(attn.heads // attn.kvheads, dim=1)
                T = st["T"]
                res = []
                for fi in st["faces"]:
                    idx = torch.tensor(fi, device=x.device) + T
                    p = F.softmax((q[0][:, idx].float() @ k[0].float().transpose(1, 2)) / attn.headdim ** 0.5, dim=-1)  # H x nf x L
                    res.append({name: p[:, :, sp].sum(-1).mean(-1).cpu().tolist() for name, sp in st["spans"].items()} | {"all_text": p[:, :, :T].sum(-1).mean(-1).cpu().tolist()})
                st["rows"].append({"block": bi, "step": pipe.step, "faces": res})
            return None
        return f
    hs = [pipe.dm.blocks[b].attn.register_forward_pre_hook(pre(b)) for b in range(len(pipe.dm.blocks))]
    if which == "v1":
        jobs = [("pair", BIND_TWO.format(a=a, b=b), "base", {"A": a, "B": b}) for a, b in BIND_PAIRS]
        jobs += [("rlsa_" + k, t, "repair", {"rlsa": "rlsa"} | ({"Pugh": "Florence Pugh"} if k == "pugh" else {}) | ({"other": "a different woman"} if k == "chair" else {})) for k, t in BIND_RLSA.items()]
    else:
        jobs = [("pair", BIND_TWO.format(a=a, b=b), "base", {"A": a, "B": b}) for a, b in BIND2_PAIRS[len(BIND_PAIRS):]]
        jobs += [("desc_" + k, t, "base", n) for k, (t, n) in BIND2_DESC.items()]
        jobs += [("rlsa_" + k, t, arm, n) for k, (arm, t, n) in BIND2_RLSA.items()]
    out = []
    try:
        for kind, text, arm, names in jobs:
            spans = {n: span(text, p) for n, p in names.items()}
            for seed in (182301, 182302, 182303):
                st["faces"] = None
                _, _, lat, _ = pipe.sample(models[arm], text, seed, size=(1216, 832))
                img = vdec(pipe, lat)
                im = to_img(img)
                im.save(out_dir / f"{kind}_{names.get('A', '')}_{seed}.jpg".replace(" ", "_"), quality=88)
                fs = met.app.get(np.asarray(im)[:, :, ::-1])
                if len(fs) < 2:
                    print("faces<2", kind, seed, flush=True)
                    continue
                fs = sorted(fs, key=lambda g: -(g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1]))[:3]
                info = []
                for g in fs:
                    e = g.normed_embedding
                    lab = {"her": float(e @ met.centroid.cpu().numpy())} | {n: float(e @ cent[n]) for n in names.values() if n in cent}
                    x0, y0, x1, y1 = g.bbox
                    toks = [r_ * GW + c for r_ in range(max(0, int(y0 // 16)), min(GH, int(y1 // 16) + 1)) for c in range(max(0, int(x0 // 16)), min(GW, int(x1 // 16) + 1))]
                    info.append({"bbox": g.bbox.tolist(), "scores": lab, "tokens": len(toks)})
                    info[-1]["_toks"] = toks
                st["faces"], st["T"], st["spans"], st["rows"] = [i_.pop("_toks") for i_ in info], pipe.encode(text)[0][0][0].shape[1], spans, []
                pipe.sample(models[arm], text, seed, size=(1216, 832))
                st["faces"] = None
                out.append({"kind": kind, "text": text, "seed": seed, "names": names, "spans": spans, "faces": info, "attn": st["rows"]})
                common.write(out_dir / "bind.json", out)
                print(kind, names, seed, [{k: round(v, 2) for k, v in f_["scores"].items()} for f_ in info], flush=True)
    finally:
        st["faces"] = None
        for h in hs:
            h.remove()


KV_TEMPLATE = "A medium photograph of {who} standing on a red carpet, facing the camera with her face clearly visible and in focus. Evening event photography."


KV2_WHOS = [("a red-haired woman", "base"), ("a man", "base"), ("an old man", "base"), ("a dog", "base"), ("a lamp", "base"), ("a building", "base"), ("rlsa", "naive")]


def kv_probe(which="v1"):
    """What a name looks like to the binding heads: for single-person prompts naming each celebrity, "a woman" (Base) and "rlsa" (repair),
    the key and value vectors that the name span presents in the top-10 binding heads (mean over span tokens), at steps 2/5/7.
    Saved to bind/kv.pt for designing a name-like rlsa signal. Renders are not saved."""
    import torch
    pipe = Pipe(OP_STACK)
    import comfy.sd, comfy.utils
    models = {"base": pipe.base}
    for arm in ("repair", "naive"):
        m = pipe.base
        for path, s in HL_ARMS[arm]:
            m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(str(path)), s, 0)
        models[arm] = m
    top = [(h["block"], h["head"]) for h in json.loads((RUN / "bind" / "bind_analysis.json").read_text())["top_binding_heads"][:10]]
    st = {"span": None, "rec": {}}

    def pre(bi):
        attn = pipe.dm.blocks[bi].attn

        def f(mod, args):
            if st["span"] is None or pipe.step not in BIND_STEPS:
                return None
            x = args[0]
            with torch.no_grad():
                k = attn.wk(x).unflatten(-1, (attn.kvheads, attn.headdim)).transpose(1, 2)
                v = attn.wv(x).unflatten(-1, (attn.kvheads, attn.headdim)).transpose(1, 2)
                k = attn.qknorm.knorm(k)
                rep = attn.heads // attn.kvheads
                for b, h in top:
                    if b == bi:
                        sp = st["span"]
                        st["rec"][(b, h, pipe.step)] = (k[0, h // rep, sp].float().mean(0).cpu(), v[0, h // rep, sp].float().mean(0).cpu())
            return None
        return f
    blocks = sorted({b for b, _ in top})
    hs = [pipe.dm.blocks[b].attn.register_forward_pre_hook(pre(b)) for b in blocks]
    whos = [(n, "base") for n in ID_YOUNG + ID_OLD] + [("a woman", "base"), ("rlsa", "repair")] if which == "v1" else KV2_WHOS
    out = []
    try:
        for who, arm in whos:
            text = KV_TEMPLATE.format(who=who)
            full = pipe.encode(text)[0][0][0].shape[1]
            tail = pipe.encode(text[text.index(who) + len(who):])[0][0][0].shape[1]
            head_tail = pipe.encode(text[text.index(who):])[0][0][0].shape[1]
            for seed in (182401, 182402):
                st["span"], st["rec"] = list(range(full - head_tail, full - tail)), {}
                pipe.sample(models[arm], text, seed, size=(832, 1216))
                st["span"] = None
                out.append({"who": who, "arm": arm, "seed": seed, "span_len": head_tail - tail, "kv": st["rec"]})
                print(who, seed, "span tokens", head_tail - tail, flush=True)
            torch.save({"top_heads": top, "rows": out}, RUN / "bind" / ("kv.pt" if which == "v1" else "kv2.pt"))
    finally:
        st["span"] = None
        for h in hs:
            h.remove()


def named_component():
    """Text side of 'what makes a named person': in the txtmlp.3 output (what the DiT reads), each name's offset from "a woman"
    (token-aligned; name span vs the later tokens it smears into). Shared named-person component N = mean offset over 40 names.
    Reports how much of each celebrity's and of rlsa's offset lies along N, and the span/smear energy split. Repair stack loaded."""
    import torch
    sys.path.insert(0, str(HERE.parent / "2026-10-19-peel"))
    import trigger_amp
    pipe = Pipe(OP_STACK)
    prompts = [ID_TEMPLATE.format(who="a woman")] + [c["text"].replace("photograph of a woman", "photograph of a woman", 1) for c in contexts("fit")[::2]]
    prompts = [t for t in prompts if "of a woman" in t]

    def c_of(text):
        cond = pipe.encode(text)[0][0][0]
        pipe.mm.load_models_gpu([pipe.unet])
        with torch.no_grad():
            return pipe.dm.txtmlp(pipe.dm.txtfusion(pipe.dm._unpack_context(cond.to(pipe.dev, torch.bfloat16))))[0].float()

    def offset(a_text, b_text):
        """(span vector = mean over the differing tokens, smear vector = mean over the shared suffix tokens) of c(a) - c(b)."""
        ca, cb = c_of(a_text), c_of(b_text)
        ia, ib = pipe.clip.tokenize(a_text)["qwen3vl_4b"][0], pipe.clip.tokenize(b_text)["qwen3vl_4b"][0]
        ia, ib = [t[0] for t in ia][len(ia) - ca.shape[0]:], [t[0] for t in ib][len(ib) - cb.shape[0]:]
        p = 0
        while p < min(len(ia), len(ib)) and ia[p] == ib[p]:
            p += 1
        s = 0
        while s < min(len(ia), len(ib)) - p and ia[-1 - s] == ib[-1 - s]:
            s += 1
        span = ca[p:len(ia) - s].mean(0) - cb[p:len(ib) - s].mean(0)
        smear = (ca[len(ia) - s:] - cb[len(ib) - s:]).mean(0) if s else torch.zeros_like(span)
        return span.cpu(), smear.cpu(), float((ca[len(ia) - s:] - cb[len(ib) - s:]).norm() ** 2) if s else 0.0, float((ca[p:len(ia) - s] - cb[p:len(ib) - s].mean(0)).norm() ** 2)
    names = trigger_amp.ID_NAMES
    span_off, smear_off, split = {}, {}, {}
    for n in names + ["rlsa"]:
        sp, sm, e_sm, e_sp = [], [], 0.0, 0.0
        for t in prompts:
            a = TRIG + t if n == "rlsa" else t.replace("of a woman", "of " + n, 1)
            o = offset(a, t)
            sp.append(o[0]); sm.append(o[1]); e_sm += o[2]; e_sp += o[3]
        span_off[n], smear_off[n], split[n] = torch.stack(sp).mean(0), torch.stack(sm).mean(0), e_sp / (e_sp + e_sm)
        print(n, f"span share {split[n]:.2f}", flush=True)
    Nsp = torch.stack([span_off[n] for n in names]).mean(0)
    Nsm = torch.stack([smear_off[n] for n in names]).mean(0)
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    rep = {"n_prompts": len(prompts), "celebrity": {}, "rlsa": {}}
    for n in names + ["rlsa"]:
        row = {"span_cos_to_named_component": cos(span_off[n], Nsp), "smear_cos_to_named_component": cos(smear_off[n], Nsm),
               "span_energy_share": split[n], "span_norm": float(span_off[n].norm()), "smear_norm": float(smear_off[n].norm())}
        (rep["rlsa"] if n == "rlsa" else rep["celebrity"]).__setitem__(n, {k: round(v, 4) for k, v in row.items()})
    cel = rep["celebrity"].values()
    rep["celebrity_mean"] = {k: round(float(np.mean([r[k] for r in cel])), 4) for k in next(iter(cel))}
    rep["random_cos_scale"] = round(1 / np.sqrt(Nsp.numel()), 4)
    common.write(RUN / "bind" / "named_component.json", rep)
    print(json.dumps({"rlsa": rep["rlsa"], "celebrity_mean": rep["celebrity_mean"], "random_cos_scale": rep["random_cos_scale"]}, indent=1))


def overnight_scenes():
    import overnight
    return overnight.scenes()


def export(spec):
    """txtmlp.3 diff (Subject A derived, not distributed) for a component combination "name:coef,name:coef" (coefs in style-dial energy units): dW_X = M dW_amp, M = Σ coef·scale·UUᵀ."""
    import torch
    from safetensors.torch import save_file
    dW = amp_dirs("cpu")[0]
    C = torch.load(RUN / "resp_fit" / "components.pt", weights_only=False)
    M = torch.zeros(6144, 6144, dtype=torch.float64)
    for part in spec.split(","):
        n, c = part.rsplit(":", 1)
        U = C["U"][n].double()
        M += float(c) * C["scale"][n] * U @ U.T
    tag = spec.replace(" (dial)", "").replace(":", "").replace(",", "_").replace(".", "p")
    out = AMP / f"trigamp_gc_{tag}.safetensors"
    save_file({"diffusion_model.txtmlp.3.diff": (M @ dW).float().contiguous()}, out, metadata={"note": f"Subject A derived (not distributed): {spec}."})
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["parity", "fdcheck", "grads", "analyze", "fd", "resp", "resp-analyze", "resp-validate", "windows", "export", "dose", "split-amp", "tiles", "tiles-generic", "highlights", "group", "group-score", "idmap", "idmap-analyze", "group-steer", "routed", "bind-heads", "bind-heads2", "named-component", "kv-probe", "kv-probe2"])
    ap.add_argument("--amp", default="genres2")
    ap.add_argument("--dose-set", default="base", choices=["base", "far", "sched", "sched_held"])
    ap.add_argument("--spec", default="style_spec:2")
    ap.add_argument("--set", default="fit", choices=["fit", "held"])
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    {"parity": parity, "fdcheck": fdcheck, "grads": lambda: grads(a.set), "analyze": analyze, "fd": lambda: fd(a.set), "resp": lambda: resp(a.set), "resp-analyze": resp_analyze, "resp-validate": resp_validate, "windows": windows, "export": lambda: export(a.spec), "dose": lambda: dose(a.dose_set), "split-amp": lambda: split_amp(a.amp), "tiles": tiles, "tiles-generic": tiles_generic, "highlights": highlights, "group": group, "group-score": group_score, "idmap": idmap, "idmap-analyze": idmap_analyze, "group-steer": group_steer, "routed": routed, "bind-heads": bind_heads, "bind-heads2": lambda: bind_heads("v2"), "named-component": named_component, "kv-probe": kv_probe, "kv-probe2": lambda: kv_probe("v2")}[a.command]()
