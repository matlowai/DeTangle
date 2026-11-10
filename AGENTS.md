# Agent guide

This file is for coding and research agents picking up this repository. It says what was done, where the evidence lives, how to apply the method to a new LoRA, and which invariants not to break. Humans may find it a useful map too.

## What this repository is

DeTangle repairs a trained character LoRA for Krea 2 after training, without gradients. The problem: a character adapter also responds to prompts that don't ask for the character (bleed). The goal: remove that response while keeping the character when the trigger is present.

The work happened in two stages:

1. **A-side surgery** (`detangle repair`). Refit each LoRA's A factor by conditional least squares from captured layer inputs: suppress the response to generic class prompts and preserve the response to triggered prompts, keeping B fixed. This removes bleed. In strong-prior scenes (sci-fi or fantasy "film still" prompts, where the base model reaches for a familiar movie-star face), it also removes the character.
2. **Composite repair** (rungs C to E in the README). Recover the character without bringing the bleed back:
   - **Block blend:** restore 35 % of the original A factors in main blocks 12–15 only (`detangle blend`).
   - **Trigger amplifier:** a closed-form least-squares edit of the text-conditioning layer `txtmlp.3`. It moves triggered prompts further along their own trigger offset and holds untriggered prompts fixed (`detangle collect-text`, `amp-fit`).
   - **Style split:** the trigger offset carries photo style as well as identity. Split the amplifier along the top snapshot-vs-cinematic directions into a style part and the rest (`amp-fit --part`, `style-directions`, `amp-split`).
   - **Step schedule:** apply the style part only on sampling steps 3–7 of 8, which is when identity is written (`ScheduledDelta`, `detangle render --schedule`).

Measured results on held-out seeds 181905–181908 (face-embedding cosine):

| | Original | A-side | Composite |
|---|---|---|---|
| Likeness, nine scenes | 0.72 | 0.44 | 0.73 |
| Likeness, six unseen genres | 0.65 | 0.10 | 0.63 |
| Generic bleed, nine scenes | +0.71 | | +0.06 |

Full story: [docs/composite-repair.md](docs/composite-repair.md). Capability summary and open directions: [docs/findings.md](docs/findings.md).

## Map

| Path | What it is |
|---|---|
| `src/delta_surgery_seed/core.py` | A-side least-squares solver (`repair`/`evaluate`). Its output is pinned byte-for-byte; don't change its arithmetic. |
| `src/delta_surgery_seed/composite.py` | `blend_blocks`, `trigger_offset`, `fit_amplifier`, `style_directions`, `split_amplifier`, `compose`. Arithmetic follows the as-run order. |
| `src/delta_surgery_seed/schedule.py` | `ScheduledDelta` / `Component`: forward hook that switches a weight delta on for a step window, with an audit log in `.calls`. |
| `src/delta_surgery_seed/collect.py` | `LayerInputCollector` (pre-hooks keyed by LoRA name), the seeded `Reservoir` sampler, and text-capture I/O. |
| `src/delta_surgery_seed/krea2.py` | Optional in-process Krea 2 pipeline on a ComfyUI checkout (`--comfyui` / `DETANGLE_COMFYUI`). Imported lazily. |
| `src/delta_surgery_seed/cli.py`, `composite_cli.py` | `detangle` subcommands. Every command writes `OUT` plus `OUT.json` (hashes, settings, diagnostics) and refuses to overwrite. |
| `tests/` | Unit tests on synthetic tensors. Run: `python -m unittest discover -s tests -v`. |
| `lab/` | DeltaScope viewer and scorer. Published records live in `lab/src/deltascope/case_studies/conditional_identity.json`. Tests: `python -m unittest discover -s lab/tests -v`. |
| `research/` | As-run study scripts behind each finding, with a script-to-finding map in `research/README.md`. A record, not a pipeline: most need the undistributed adapter. |
| `docs/` | Mechanism, case studies, lessons, validation records. |
| `tools/audit_release.py` | Release audit. Run before any publish. New images must be added to its approved list. |

## Applying the method to a new LoRA

1. **Measure first.** Render Base, your LoRA and later each repair on matched prompts and seeds, with and without the trigger. Score likeness (triggered) and bleed excess (untriggered minus Base) with a face embedding or another metric suited to your concept. Use DeltaScope to compare. Include strong-prior scenes; the failure only shows up there.
2. **A-side surgery.** `detangle collect-layers` (class vs triggered prompt pairs) then `detangle repair --mu 0.3 --ridge 0.001`. Check bleed and likeness. If likeness holds everywhere, stop here.
3. **Localize before solving anything.** Make hybrid adapters (original A in one block window, repaired elsewhere) with `detangle blend --fraction 1`, one render each. The windows that restore likeness are your candidates. For this character, that was blocks 8–15, with 12–15 the better trade.
4. **Blend.** Sweep the fraction in the chosen blocks and watch bleed.
5. **Amplify.** Write an amplifier spec (triggered prompts across the scenes that fail; many untriggered prompts to preserve, including generic people, named people, men and non-person prompts). Then `collect-text` with the repaired adapter loaded, then `amp-fit`. Sweep the dose.
6. **Separate and schedule.** If the amplifier shifts photo style or scene, split it (`--part style` / `rest`) and schedule the style part over late steps. Compare against a fixed dose; in this work, scheduling beat searching for a cleaner fixed direction.
7. **Validate on held-out seeds and unseen prompts** chosen before you look at the results. Record settings in the JSON sidecars.

The numbers in this repo (μ 0.3, blocks 12–15, 35 %, 6×, steps 3–7) were selected for one character on one model. Expect to re-select them.

## Invariants

- `detangle repair` / `evaluate` output must stay byte-identical (`detangle demo` produces 12 files; compare hashes before and after a change).
- Reproduction depends on arithmetic order and dtype: `--method historical` vs `auto`, GPU vs CPU float64 for `amp-fit`, separate patches vs a packaged file, and projection form vs a precomputed style delta for the scheduled part. See [docs/composite-repair-validation.md](docs/composite-repair-validation.md) before "simplifying" any of these.
- A good local fit does not certify rendered behavior. Always judge an edit by renders on matched seeds.
- Commands never overwrite outputs and always write a JSON sidecar. Keep it that way.

## Known limits and open problems

- One character, one private adapter, one model. Multi-identity studies are the main missing evidence.
- The repaired adapter still spreads the character to other faces in group shots (8/8 non-target faces). Probes show why: names bind to one face through attention heads in layers 7–11, and the trigger barely participates in that binding. Repairing native name binding (activation patching, with keys and values tested separately and early vs late steps) is the leading open direction.
- Untriggered faces can still drift from Base's face (≈0.7 similarity on some scenes), and named-person prompts are weakened. A base-response preservation constraint is open.
- Step scheduling runs only through the in-process pipeline; stock ComfyUI hook keyframes did not work with the FP8-scaled weights.
- `collect-layers` is smoke-tested but not compared with the historical captures, which came from the training runtime.

## Publication rules for contributions

- Never commit checkpoints, adapters, captures, reference face images or embeddings, or run outputs. `tools/audit_release.py` enforces most of this.
- No absolute local paths, usernames or private infrastructure names in code, docs or JSON.
- Keep image content and prompt wording safe for work. Describe side effects in neutral terms (for example "wardrobe drift" for clothing changes caused by the trigger).
