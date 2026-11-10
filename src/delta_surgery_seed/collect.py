"""Backend-agnostic activation collection with forward hooks.

Copyright MatlowAI. MIT licensed.

Two capture kinds feed the composite repair:

* Layer-input reservoirs for ``detangle repair``: for every adapted linear
  layer, a seeded, bounded sample of the input rows that reach it (the
  capture schema in docs/captures.md). ``LayerInputCollector`` hooks any
  ``torch.nn.Module`` whose submodules can be addressed by the LoRA key names.
* Per-prompt text-conditioning captures for the trigger amplifier: the full
  token matrices entering and leaving one linear layer (Krea 2: ``txtmlp.3``)
  plus the token ids, saved with ``save_text_captures``.

The Krea 2 driver in ``delta_surgery_seed.krea2`` uses these with a ComfyUI
model; other runtimes can call them directly.
"""

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .core import canonical

TEXT_SCHEMA = "detangle-text-capture/1"


class Reservoir:
    """Seeded priority sampling of rows, bounded by ``capacity`` (the historical capture's scheme)."""

    def __init__(self, capacity=1024, seed=1001):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity, self.seen = capacity, 0
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.tokens, self.priorities = None, None

    def add(self, x):
        x = x.detach().reshape(-1, x.shape[-1])
        for chunk in x.split(4096):
            chunk = chunk.to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(chunk).all():
                raise ValueError("Non-finite captured activations")
            p = torch.rand(len(chunk), generator=self.generator, dtype=torch.float64)
            self.seen += len(chunk)
            if self.tokens is not None:
                chunk, p = torch.cat([self.tokens, chunk]), torch.cat([self.priorities, p])
            idx = p.topk(min(self.capacity, len(p))).indices
            self.tokens, self.priorities = chunk[idx], p[idx]

    def matrix(self):
        if self.tokens is None:
            raise ValueError("Layer received no inputs")
        return self.tokens.contiguous()


def lora_layer_names(keys):
    """Canonical layer names (no namespace, no suffix) of every ``*.lora_A.weight`` key."""
    names = sorted({canonical(k.removesuffix(".lora_A.weight")) for k in keys if k.endswith(".lora_A.weight")})
    if not names:
        raise ValueError("No lora_A.weight keys")
    return names


class LayerInputCollector:
    """Reservoir-sample the input of named linear submodules of ``root``.

    ``names`` are submodule paths relative to ``root`` (for Krea 2 in ComfyUI,
    ``root`` is the diffusion model and names look like ``blocks.0.attn.wq``,
    matching the LoRA keys after ``canonical``). Use as a context manager, run
    forward passes, then ``tensors()`` or ``save(path)``.
    """

    def __init__(self, root, names, capacity=1024, seed=1001):
        self.root, self.names = root, list(names)
        self.capacity, self.seed = capacity, seed
        self.buffers, self.handles = {}, []

    def __enter__(self):
        for name in self.names:
            module = self.root.get_submodule(name)
            weight = getattr(module, "weight", None)  # runtime-specific linear ops (e.g. fp8) need not subclass nn.Linear
            if weight is None or getattr(weight, "ndim", 0) != 2:
                raise ValueError(f"Not a linear layer: {name}")
            buf = self.buffers[name] = Reservoir(self.capacity, self.seed)

            def hook(_module, args, buf=buf):
                if not args or not isinstance(args[0], torch.Tensor):
                    raise ValueError("Expected a positional tensor at a captured linear layer")
                buf.add(args[0])
            self.handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def tensors(self):
        return {name: buf.matrix() for name, buf in self.buffers.items()}

    def counts(self):
        return {name: {"seen": buf.seen, "kept": 0 if buf.tokens is None else len(buf.tokens)}
                for name, buf in self.buffers.items()}

    def save(self, path):
        save_file(self.tensors(), str(path))


class LinearIO:
    """Record the input and output of one linear layer on the next forward call."""

    def __init__(self, module):
        self.module, self.input, self.output, self.handle = module, None, None, None

    def __enter__(self):
        def hook(_module, args, output):
            self.input, self.output = args[0].detach(), output.detach()
        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False


def save_text_captures(path, records, module="txtmlp.3", extra=None):
    """Save per-prompt captures: ``records`` maps text -> (ids, input [tokens, in], output [tokens, out]).

    Tensors keep their captured dtype (bf16 for Krea 2, which loses nothing).
    The prompt list and module name are stored in the safetensors metadata.
    """
    tensors, prompts = {}, []
    for i, (text, (ids, h, c)) in enumerate(records.items()):
        ids = torch.as_tensor(list(ids), dtype=torch.int64)
        if not (len(ids) == h.shape[0] == c.shape[0]):
            raise ValueError("Token ids, inputs and outputs must have the same length")
        tensors[f"prompt.{i:05d}.ids"] = ids
        tensors[f"prompt.{i:05d}.input"] = h.detach().cpu().contiguous()
        tensors[f"prompt.{i:05d}.output"] = c.detach().cpu().contiguous()
        prompts.append(text)
    meta = {"schema": TEXT_SCHEMA, "module": module, "prompts": json.dumps(prompts)}
    if extra:
        meta["provenance"] = json.dumps(extra)
    save_file(tensors, str(path), metadata=meta)


class TextCaptures:
    """Lazy reader for ``save_text_captures`` files; ``get(text)`` -> (ids list, input float32, output float32)."""

    def __init__(self, *paths):
        self.files, self.index = [], {}
        for path in paths:
            f = safe_open(str(path), framework="pt", device="cpu")
            meta = f.metadata() or {}
            if meta.get("schema") != TEXT_SCHEMA:
                raise ValueError(f"Not a text capture file: {path}")
            self.files.append(f)
            for i, text in enumerate(json.loads(meta["prompts"])):
                self.index.setdefault(text, (len(self.files) - 1, i))

    def __contains__(self, text):
        return text in self.index

    def get(self, text):
        if text not in self.index:
            raise KeyError(f"Prompt not captured: {text[:80]!r}")
        f, i = self.index[text]
        f = self.files[f]
        ids = f.get_tensor(f"prompt.{i:05d}.ids").tolist()
        return ids, f.get_tensor(f"prompt.{i:05d}.input").float(), f.get_tensor(f"prompt.{i:05d}.output").float()
