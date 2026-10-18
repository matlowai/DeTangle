# Activation inputs and experiment contract

The CLI deliberately starts after activation collection. Model families differ in layer naming, timestep conventions, conditioning, LoRA injection and distributed execution. Supply captures from a collector appropriate to your model; no model-specific paths or hook integrations are assumed here.

## Tensor layout

The adapter checkpoint contains two matrices per supported layer:

```text
transformer.blocks.0.attn.q.lora_A.weight  [rank, input_width]
transformer.blocks.0.attn.q.lora_B.weight  [output_width, rank]
```

Both capture files contain the layer input immediately before that A projection:

```text
transformer.blocks.0.attn.q  [sample_count, input_width]
```

Use safetensors rather than pickled Python objects. Values must be nonempty, finite, floating-point matrices. Do not provide output activations, Gram matrices, face embeddings, or samples-as-columns arrays. Save one key per layer, with the same layer set in both role files. Each supported A factor must have exactly one unambiguous capture and a dimensionally compatible B factor. Extra captured layers are ignored.

```python
import torch
from safetensors.torch import save_file

# Example only: replace these random rows with your measured layer inputs.
layer = "transformer.blocks.0.attn.q"
save_file({layer: torch.randn(256, 64)}, "class-fit.safetensors")
save_file({layer: torch.randn(256, 64)}, "trigger-fit.safetensors")
```

The mapper permits a leading `transformer.` or `diffusion_model.` namespace and normalizes `$$` separators to dots. It does not guess arbitrary suffixes or convert every LoRA format. Convolutional factors, LoKr, DoRA and other parameterizations require their own validated implementation.

## Record the capture recipe

Alongside each capture, save a manifest containing:

- Source adapter and base-model content hashes; whether the adapter was active and its loading strength.
- Exact prompts actually consumed, class/trigger pairing, and roles. Count processed examples, not only entries in an intended prompt file.
- Latent/noise seeds, timesteps or flow fractions, precision, model implementation and sampler settings where applicable.
- Input token selection, tokens per layer and role, reservoir sampling seed, and total observed token counts.
- Capture-file hashes and a separate held-out split with its intended independence level.

Keep those manifests separate from the solver reports. The solver can verify bytes and dimensions; it cannot reconstruct these conditions from a matrix. Identity-preservation capture should use the intended class cue in both conditions and vary the trigger deliberately. For other concepts, define a corresponding positive capability and unwanted-response contract before collecting data.

## Fit, draw holdout, prompt holdout, intervention

A fresh token draw from the same prompts checks sampling sensitivity. New prompts check broader coverage. Neither is equivalent to rerunning the edited model and measuring its newly produced activations. Use these as separate experiments and retain per-prompt results.

The evaluator rejects either fitting file reused verbatim in an evaluation role. Different hashes are only a weak guard: changing metadata or shuffling the same rows can also change a hash. Independence requires auditing the manifests.

Useful model probes include per-layer `BA` response change on both roles, repeat captures with fixed and varied reservoir seeds, and blockwise edited-versus-source interventions followed by paired renders. A large response change identifies a candidate layer; it is not causal localization until an intervention changes the behavior under controlled conditions.

## Output invariants

The original checkpoint is never written. The output preserves all tensor keys, shapes and dtypes, all non-A tensor bytes, and every original metadata entry. New provenance uses an unused metadata key. The JSON report contains source/capture/output hashes, solve settings, sample counts and post-cast fit diagnostics. Paths are supplied through command arguments and are not embedded in reports.

`evaluate` verifies hash linkage, all non-A tensors and original metadata before reporting held-out response metrics. These are integrity checks on this A-only operation, not a claim that the repaired adapter preserves all model capabilities.
