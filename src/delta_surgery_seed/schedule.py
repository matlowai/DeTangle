"""Step-scheduled weight differences: switch a component on for a window of sampling steps.

Copyright MatlowAI. MIT licensed.

A weight difference ``dW`` on a linear layer adds ``strength * x dWᵀ`` to its
output. Loading it as a patch applies it on every sampling step. This module
applies it from a forward hook instead, only while the sampler is on chosen
steps. It changes nothing else: the base weights, other adapters and the
sampler are untouched.

The hook needs to know the current step. Drive it from the sampler callback:

    schedule = ScheduledDelta(model.txtmlp[3], style_delta, strength=6.0, steps=range(3, 8))
    with schedule:
        sample(..., callback=schedule.comfy_callback)   # ComfyUI-style (step, x0, x, total)

``comfy_callback(step, ...)`` runs after step ``step`` finishes and sets the
next step's index, so the first model call sees step 0. For diffusers
pipelines use ``callback_on_step_end=schedule.diffusers_callback``. Without a
callback, call ``set_step(i)`` before each model call. Steps are 0-based: for
an 8-step sampler, ``range(3, 8)`` means "steps 3-7 of 8" (the first three
steps run without the component).

The hooked layer must run once per sampling step with the step's conditioning
(true for Krea 2's ``txtmlp.3``, which is recomputed on every model call). If
classifier-free guidance batches conditional and unconditional passes
together, the delta is applied to both rows of that call.
"""

import torch


def parse_steps(text, total=None):
    """'all', '3-7', '3,4,6-7' -> sorted 0-based step list (None means every step)."""
    text = str(text).strip()
    if text == "all":
        return None
    steps = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty step range in {text!r}")
        lo, sep, hi = part.partition("-")
        lo, hi = int(lo), int(hi) if sep else int(lo)
        if lo < 0 or hi < lo:
            raise ValueError(f"Invalid step range {part!r}")
        steps.update(range(lo, hi + 1))
    if total is not None and max(steps) >= total:
        raise ValueError(f"Step {max(steps)} is outside a {total}-step schedule")
    return sorted(steps)


class Component:
    """One weight difference ``delta`` [out, in] at ``strength`` on ``steps`` (None = all)."""

    def __init__(self, delta, strength=1.0, steps=None, output_projection=None):
        if delta.ndim != 2:
            raise ValueError("delta must be a matrix [out_width, in_width]")
        if output_projection is not None and output_projection.shape != (delta.shape[0], delta.shape[0]):
            raise ValueError("output_projection must be [out_width, out_width]")
        self.delta = delta.float()
        self.strength = float(strength)
        self.steps = None if steps is None else frozenset(int(s) for s in steps)
        # Optional P: add strength * ((x dWᵀ) P), the study's in-process form of P dW.
        self.output_projection = None if output_projection is None else output_projection.float()

    def active(self, step):
        return self.steps is None or step in self.steps

    def term(self, x):
        if self.delta.device != x.device:
            self.delta = self.delta.to(x.device)
            if self.output_projection is not None:
                self.output_projection = self.output_projection.to(x.device)
        y = x.float() @ self.delta.T
        if self.output_projection is not None:
            y = y @ self.output_projection
        return self.strength * y


class ScheduledDelta:
    """Forward hook adding one or more step-windowed components to a module's output."""

    def __init__(self, module, delta=None, strength=1.0, steps=None, components=None, output_projection=None):
        if (delta is None) == (components is None):
            raise ValueError("Pass either delta or components")
        self.module = module
        self.components = list(components) if components is not None else [
            Component(delta, strength, steps, output_projection)]
        self.step = 0
        self.calls = []  # (step, applied) per hooked call, for audits
        self.handle = None

    def set_step(self, step):
        self.step = int(step)

    def comfy_callback(self, step, x0=None, x=None, total=None):
        """ComfyUI sampler callback: step ``step`` just finished; the next model call is ``step + 1``."""
        self.step = step + 1

    def diffusers_callback(self, pipe, step, timestep, callback_kwargs):
        """diffusers ``callback_on_step_end``: step ``step`` just finished."""
        self.step = step + 1
        return callback_kwargs

    def _hook(self, module, args, output):
        active = [c for c in self.components if c.active(self.step)]
        self.calls.append((self.step, bool(active)))
        if not active:
            return None
        x = args[0]
        extra = active[0].term(x)
        for c in active[1:]:
            extra = extra + c.term(x)
        return output + extra.to(output.dtype)

    def attach(self):
        if self.handle is not None:
            raise RuntimeError("Already attached")
        self.step, self.calls = 0, []
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False
