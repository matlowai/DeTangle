# Research scripts behind the published findings

These are the study scripts as they were run for the October 2026 findings in DeltaScope and [docs/lessons.md](../docs/lessons.md), scrubbed for publication. They are a record of what was computed. They are not a turnkey pipeline. For your own LoRA, use the `detangle` commands; [the reproduction check](../docs/composite-repair-validation.md) shows that they reproduce these scripts' adapters exactly.

**Most scripts cannot run from this repository alone.** They need the private character LoRA and its saved captures. Scoring additionally needs the 34 private reference photos and an ArcFace evaluator module from an earlier study, and the renders need a ComfyUI checkout with Krea 2 plus the model files. None of these are distributed. Exceptions: `export_amp_spec.py` rebuilds the amplifier's prompt sets with no model or GPU, and the plotting and analysis scripts run on study outputs if you have them.

## What changed from the as-run copies

- **Paths** come from environment variables (below), not hardcoded locations. Outputs go under `$DETANGLE_RESEARCH_ROOT/runs/<study>/`, which keeps the original layout.
- **Notes** marking files as private now say neutrally that the adapter is not distributed. References to internal planning documents were removed.
- **The render catalog** (study infrastructure that reused identical renders across steps) is not included. `overnight.py` now runs without it. The `lane` command is renamed `worker`, and the default ComfyUI port is a variable.
- **The graph template's adapter filename** was renamed. `protocol.json` records the published template's hash and the as-run hash.
- **One prompt id** in `rekey.py sheets` was renamed (`chair_behind`).
- **Prompt texts, seeds, constants and arithmetic** are as run. `export_amp_spec.py` is new. It writes the amplifier's prompt lists in the `detangle amp-fit` spec format, byte-identical to the spec used in the reproduction check.

Not included: site and figure builders that only formatted DeltaScope records, contact-sheet tools, the vendored Fizgig sources (fetch the revision recorded in `protocol.json` if you need the block profile), the frozen test-set files, and all run outputs.

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `DETANGLE_RESEARCH_ROOT` | Output root (`runs/<study>/`) and inputs that studies read from earlier runs | `./detangle-research` |
| `DETANGLE_SOURCE_LORA` | The original character LoRA | `source-lora.safetensors` |
| `DETANGLE_CAPTURE_DIR` | Saved class/trigger captures (`Xc.safetensors`, `Xt.safetensors`) | `captures` |
| `DETANGLE_HISTORICAL_LORA` | Historical scalar-surgery adapter (first-release comparison) | `historical-surgery.safetensors` |
| `DETANGLE_SCORER_STUDY` | Directory providing `score_study.prepare_evaluator` (ArcFace and reference centroid) | `scorer` |
| `DETANGLE_COMFYUI` | ComfyUI checkout with Krea 2 (the studies pinned commit 87d23b817; an optional `pinned_site/` holds comfy_kitchen 0.2.20 and comfy_aimdo 0.4.10) | `ComfyUI` |
| `DETANGLE_COMFY_PYTHON` | Python used to start the render server | current interpreter |
| `DETANGLE_COMFY_MODELS` | ComfyUI models directory (Krea 2 Turbo FP8 scaled, Qwen3-VL 4B FP8 scaled, Qwen-Image VAE) | `$DETANGLE_COMFYUI/models` |
| `DETANGLE_COMFY_PORT` | Port of the study-owned ComfyUI server | `8188` |
| `DETANGLE_CLIP_DIR` | Local CLIP ViT-L/14 (style and outfit scores) | `clip-vit-large-patch14` |
| `DETANGLE_ARCFACE_ONNX` | buffalo_l `w600k_r50.onnx` (gradient study) | `~/.insightface/models/buffalo_l/w600k_r50.onnx` |

Scripts import each other through their relative locations, so keep the `studies/<date-name>/` layout.

## Script map

Commands are each script's own subcommands. The likeness and bleed numbers in the findings come from the scoring commands of the same scripts.

### The repair ladder (A → B → C → D → E)

| Finding | Script and commands | Packaged equivalent |
|---|---|---|
| B: A-side surgery, CPU float64, μ 0.3, ridge 0.001 | `2026-10-18-factor-and-block-repair/fit_factors.py` | `detangle repair --method auto` |
| Only blocks 8–15 matter for the space-station failure (tensor-swap hybrids) | `2026-10-19-entanglement-probes/probes.py` group E2 (`prepare`, `render`, `score`, `report`); needs the Fizgig block map | `blend_blocks` at fraction 1 |
| Trigger amplifier on `txtmlp.3` (119 triggered / 390 preserved prompts, λ 4, ε 1e-4) | `2026-10-19-peel/trigger_amp.py fit`; `export_amp_spec.py` | `detangle collect-text`, `amp-fit` |
| Blend threshold over blocks 8–15; 8–11 vs 12–15 split; 35 % of 12–15 with × 6 | `2026-10-20-overnight/overnight.py blends`, steps `blend`, `blend2`, `blend3`, `confirm`, `confirm2`, `final`; `package` | `detangle blend`, `compose` |
| Held-out seeds 181905–181908, six unseen genres, 70-prompt bleed panel, × 8 / × 10 | `overnight.py` steps `fresh_scenes`, `fresh_genres`, `fresh_fill`, `genres`, `hard`, `amp10`, `bleed_win` | |
| What did not work: strong-prior-only amplifier, reduced adapter strength | `trigger_amp.py fit-ctx`; `overnight.py` steps `ctx`, `dialback` | |
| D: the trigger is a bundle; style dial (snapshot ↔ cinematic, top 4 directions) | `trigger_amp.py fit-variant --variant style` / `nostyle` (and `diag`, `nostyle_spec`, `style_spec` controls); `overnight.py` steps `genres_decomp`, `style_variants`, `style_sweep`, `comp_fill_a`, `comp_fill_b`; `side_effects.py` (snapshot score, scene vs Base, outfit sentence, blur) | `amp-fit --part style/rest`, `style-directions`, `amp-split` |
| Identity-only amplifier (40 named actresses, clothing/hair/style removed) | `trigger_amp.py fit-variant --variant identity` / `identity_noclothes` / `idiag`; `overnight.py` step `identity_amps` | |
| E: identity is written late; component search, finite-difference checks, step windows, scheduled style dial | `2026-10-20-grad-components/grad_amp.py` `parity`, `fdcheck`, `grads`, `analyze`, `fd`, `resp`, `resp-analyze`, `resp-validate`, `windows`, `dose --dose-set sched` / `sched_held`, `split-amp`, `highlights`, `tiles`; `plot_dose.py`; `overnight.py` step `grad_cands`, and `sched_server` (ComfyUI hook keyframes, which did not run with FP8-scaled weights) | `ScheduledDelta`, `detangle render --schedule` |
| Two currencies: block-side blend weight vs bleed at style 0 | `overnight.py` step `block_xr` | |
| Fit data decides where the trigger holds (genre capture, western leak) | `trigger_amp.py fit-variant --variant genres` / `genres2`, `leak --variant genres`; `overnight.py` steps `capture_genres`, `capture_genres2`, `capture_style` | `export_amp_spec.py --variant genres2` |

### Identity priors

| Finding | Script and commands |
|---|---|
| Phrase ablations; film-still sentence on passing scenes; named and described identities beat the weakened trigger | `probes.py` groups E1 and C |
| The fallback face as a blend of the model's actresses (19-concept panel, NNLS mix) | `actress_panel.py`, `actress_overlap.py` |
| Peeling one identity; matched controls; protected peels; peeling further | `2026-10-19-peel/peel.py` (`fit`, `fit-specific --names …`); `overnight.py` steps `peel_ferguson`, `peel_rapace`, `peel_hamilton`, `peel_tp`, `age`; `trig-gram` |

### Group shots and binding

| Finding | Script and commands |
|---|---|
| The trigger spreads to other faces; wording and regional conditioning | `overnight.py` steps `mixed`, `mixed2`, `mixed_win`, `mixed_region`, `mixed_region2`, `mixed_region3`, `mixed_mask`; `grad_amp.py group`, `group-score` |
| Where identity lives (identity map, private axis) | `grad_amp.py idmap`, `idmap-analyze`; `idmap_controls.py`; `plot_idmap.py` |
| Two parts make her (text-only vs image-only adapter weights) | `grad_amp.py routed` |
| How a name talks vs how the trigger talks | `grad_amp.py named-component`, `kv-probe`, `kv-probe2` |
| Binding heads in blocks 7–11 (best: block 9, head 15) | `grad_amp.py bind-heads`, `bind-heads2`; `bind_analyze.py` |
| Her direction in other faces (hand-placed masks, diagnostic) | `grad_amp.py group-steer` |
| Names bind through their last token; local vs broadcast; the read-feature probe | `2026-10-20-grad-components/keyswap.py` (`targets`, `targets-inslot`, `targets-tok`, `splice`, `mask`, `attrib`, `switch-dose`, `probe`, `probe2`, `probe3`, `probe-analyze`) |
| Closed-form binding repair: key lock, executor re-key, sign check, frozen test set | `2026-10-22-rekey/rekey.py` (`capture*`, `keylock*`, `export*`, `validate*`, `signcheck`, `testset`, `test-analyze`, `gender`, `gender-audit`, `eval`, `held`) |
| Does Base treat a trigger candidate like a name | `2026-10-25-trigger-probe/probe.py` |
| Name gate, name-reading maps, embedding pre-bind on Krea 2 | `2026-10-26-krea-gate/gate.py`, `maps.py`, `prebind.py`, `rekey_pb.py` |
| Key targets for the pre-bind, with guard and paired checks | `2026-11-01-knn-prebind/knn_prebind.py`, `paired.py`, `test_targets.py`, `run_chain.sh` |

## Notes for readers

- Prompts name real actresses because these studies measure how the base model represents named people. Every similarity is a resemblance to the model's own renderings, not a statement about a person.
- Face-region masks, "her slot" and similar labels come from the face most similar to the character or from hand-placed boxes, as each docstring says. They are diagnostics, not inference-time tools.
- Several studies render through a study-owned ComfyUI server (`probes.render`). Others use the in-process pipeline in `grad_amp.Pipe`, with LoRA patches baked into the FP8 weights. Pixels differ slightly between the two paths. The published cohorts note which path made them.
