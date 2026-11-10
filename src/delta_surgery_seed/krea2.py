"""Krea 2 activation collector and in-process sampler on a ComfyUI checkout.

Copyright MatlowAI. MIT licensed.

Optional: requires a ComfyUI checkout with Krea 2 support (its Python
dependencies installed in the running environment) and the Krea 2 diffusion
model, Qwen3-VL text encoder and, for images, the VAE. Nothing is downloaded.
Install the extra with ``pip install 'matlowai-detangle[krea2]'`` and pass the
checkout with ``--comfyui`` or the ``DETANGLE_COMFYUI`` environment variable.

What it provides:

* ``text_conditioning(text)``: the input and output of ``txtmlp.3`` (Krea 2's
  last text projection, computed once per prompt and model call) with the
  token ids, exactly as the trigger-amplifier study computed them.
* ``collect_layer_inputs(...)``: layer-input reservoirs for ``detangle
  repair`` on every adapted linear layer, from image/prompt pairs noised at
  fixed flow fractions.
* ``sample(...)``: a KSampler-equivalent text-to-image run that can drive a
  ``ScheduledDelta`` from the sampler callback (the step-scheduled style dial).

The historical layer captures were made with a training runtime, not ComfyUI;
captures from this collector are a different numerical condition. Record the
runtime alongside the capture.
"""

import os
from pathlib import Path
import sys

import torch

from .collect import LayerInputCollector, lora_layer_names

TEXT_MODULE = "txtmlp.3"


def comfyui_path(path=None):
    path = path or os.environ.get("DETANGLE_COMFYUI")
    if not path:
        raise ValueError("Pass --comfyui or set DETANGLE_COMFYUI to a ComfyUI checkout with Krea 2 support")
    path = Path(path).resolve()
    if not (path / "comfy" / "sd.py").is_file():
        raise ValueError(f"Not a ComfyUI checkout: {path}")
    return path


class Krea2:
    """A Krea 2 model, text encoder and optional VAE loaded through ComfyUI, with LoRAs/patches applied.

    ``loras`` is a list of ``(path, strength)``; each file may hold LoRA
    factors and/or ``.diff`` weight differences (ComfyUI applies both).
    ``study_pipeline=True`` applies the in-process settings the scheduling study
    used (PyTorch attention, torch RoPE path); renders then match that study's
    pipeline rather than a default ComfyUI server.
    """

    def __init__(self, comfyui, diffusion_model, text_encoder, vae=None, loras=(), study_pipeline=False):
        root = comfyui_path(comfyui)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if study_pipeline:
            from comfy.cli_args import args
            args.use_pytorch_cross_attention = True
        import comfy.model_management as mm
        import comfy.sample
        import comfy.samplers
        import comfy.sd
        import comfy.utils
        if study_pipeline:
            mm.in_training = True
        self.comfy_sd, self.comfy_utils, self.comfy_sample, self.comfy_samplers, self.mm = (
            comfy.sd, comfy.utils, comfy.sample, comfy.samplers, mm)
        self.device = mm.get_torch_device()
        self.clip = comfy.sd.load_clip(ckpt_paths=[str(text_encoder)], clip_type=comfy.sd.CLIPType.KREA2)
        self.vae = None if vae is None else comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(vae)))
        self.base = comfy.sd.load_diffusion_model(str(diffusion_model))
        self.unet = self.with_loras(loras)
        self.loras = [(str(p), float(s)) for p, s in loras]

    def with_loras(self, loras, model=None):
        model = model or self.base
        for path, strength in loras:
            model, _ = self.comfy_sd.load_lora_for_models(model, None, self.comfy_utils.load_torch_file(str(path)), float(strength), 0)
        return model

    @property
    def dm(self):
        return self.unet.model.diffusion_model

    # ---- text conditioning (trigger amplifier captures) ----
    @torch.no_grad()
    def text_conditioning(self, text):
        """(token ids aligned with rows, txtmlp.3 input, txtmlp.3 output) on CPU, in the model's bf16."""
        tok = self.clip.tokenize(text)
        stream = tok["qwen3vl_4b"] if "qwen3vl_4b" in tok else next(iter(tok.values()))
        ids = [t[0] for t in stream[0]]
        cond = self.clip.encode_from_tokens(tok)
        self.mm.load_models_gpu([self.unet])
        dm = self.dm
        h = dm.txtmlp[:3](dm.txtfusion(dm._unpack_context(cond.to(self.device, torch.bfloat16))))
        c = dm.txtmlp[3](h)
        return ids[len(ids) - cond.shape[1]:], h[0].cpu(), c[0].cpu()

    # ---- layer-input captures for `detangle repair` ----
    @torch.no_grad()
    def collect_layer_inputs(self, checkpoint_keys, pairs, timesteps=(.1, .25, .4, .55, .7, .85), noise_seed=42,
                             capacity=1024, reservoir_seed=1001, resolution=512):
        """Reservoir-sample adapted-layer inputs for the class and trigger prompt of each pair.

        ``pairs``: dicts with ``image`` (path), ``class_prompt`` and
        ``trigger_prompt``. Each image is fitted to ``resolution`` squared and
        VAE-encoded once. For each flow fraction f the noise level is
        sigma = floor(f * 999) / 1000 and the noise generator is reseeded with
        ``noise_seed`` (CPU generator), so both roles see identical noise and
        latents. Returns ``{role: (tensors, counts)}``.
        """
        from PIL import Image, ImageOps
        import numpy as np
        if self.vae is None:
            raise ValueError("Layer captures need --vae to encode the pair images")
        names = lora_layer_names(checkpoint_keys)
        model = self.unet.model
        sampling = self.unet.get_model_object("model_sampling")
        latents = []
        for row in pairs:
            with Image.open(row["image"]) as im:
                im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (resolution, resolution))
                pixels = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]
            latents.append(model.process_latent_in(self.vae.encode(pixels)).cpu())
        out = {}
        for role in ("class", "trigger"):
            self.mm.load_models_gpu([self.unet])
            with LayerInputCollector(self.dm, names, capacity, reservoir_seed) as collector:
                for frac in timesteps:
                    gen = torch.Generator(device="cpu").manual_seed(noise_seed)
                    sigma = int(frac * 999) / 1000
                    for row, latent in zip(pairs, latents, strict=True):
                        cond = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(row[role + "_prompt"]))[0][0]
                        self.mm.load_models_gpu([self.unet])
                        noise = torch.randn(latent.shape, generator=gen)
                        s = torch.full((1,), sigma)
                        x = sampling.noise_scaling(s.view(1, *([1] * (latent.ndim - 1))), noise, latent)
                        dtype = model.get_dtype()
                        model.apply_model(x.to(self.device, dtype), s.to(self.device),
                                          c_crossattn=cond.to(self.device, dtype))
                out[role] = (collector.tensors(), collector.counts())
        return out

    # ---- in-process sampling with an optional step schedule ----
    def sample(self, text, seed, width=832, height=1216, steps=8, cfg=1.0, sampler="euler", scheduler="simple",
               schedule=None, model=None):
        """Text-to-image with zeroed negative conditioning (CFG 1), returning a PIL image.

        ``schedule`` (a ``ScheduledDelta`` on ``self.dm.txtmlp[3]``) is attached
        for this run and advanced by the sampler callback.

        Load weight differences as separate patches at their dose (for example
        ``loras=[(blend, 1.0), (rest, 6.0)]``) to reproduce the study's renders:
        ComfyUI rounds a difference to its LoRA compute dtype before applying
        the strength, so a file with the dose pre-multiplied (``detangle
        compose``) renders slightly differently.
        """
        from PIL import Image
        import numpy as np
        if self.vae is None:
            raise ValueError("Sampling needs --vae")
        model = model or self.unet
        with torch.no_grad():
            pos = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(text))
        neg = [[torch.zeros_like(c[0]), {k: (torch.zeros_like(v) if isinstance(v, torch.Tensor) else v)
                                         for k, v in c[1].items()}] for c in pos]
        latent = self.comfy_sample.fix_empty_latent_channels(model, torch.zeros([1, 16, height // 8, width // 8]))
        noise = self.comfy_sample.prepare_noise(latent, seed)
        callback = None
        if schedule is not None:
            schedule.attach()
            callback = lambda step, x0, x, total: schedule.comfy_callback(step)  # noqa: E731
        try:
            with torch.no_grad():
                out = self.comfy_sample.sample(model, noise, steps, cfg, sampler, scheduler, pos, neg, latent,
                                               denoise=1.0, callback=callback, disable_pbar=True, seed=seed)
        finally:
            if schedule is not None:
                schedule.detach()
        with torch.no_grad():
            img = self.vae.decode(out.detach())
        img = img.reshape(-1, *img.shape[-3:])[0].detach()
        return Image.fromarray((img.float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8))
