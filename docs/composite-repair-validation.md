# Composite repair: reproduction check

The exported functions reproduced every tensor of the private composite repair **exactly**, from its original inputs. Rendered through the bundled in-process Krea 2 pipeline, the repair, the style dial and the step-scheduled style dial produced **pixel-identical images** to the study's own renders on all four held-out seeds. Two loading choices change pixels, and both are described below: packaging a dose into one file, and applying the scheduled component as a precomputed difference.

| Rung | What was reproduced | Exported path | Result |
|---|---|---|---|
| B | A-side surgery (μ 0.3, ridge 0.001) | `detangle repair --method auto` | 512 / 512 tensors bit-identical |
| C | 35 % of the original A factors in blocks 12–15 | `detangle blend` | all 11 stored block blends bit-identical (512 tensors each) |
| C | trigger amplifier on `txtmlp.3` | `detangle collect-text` + `amp-fit` (GPU) | 37,748,736 / 37,748,736 elements bit-identical |
| C | blend + amplifier × 6 as one file | `detangle compose` | 513 / 513 tensors bit-identical |
| D | style and style-free amplifier parts | `amp-fit --part style` / `--part rest` | both bit-identical |
| D | weight-space split used by later amplifiers | `split_amplifier` / `detangle amp-split` | both parts bit-identical |
| E | style part × 6 on sampling steps 3–7 only | `ScheduledDelta` / `detangle render --schedule` | 4 / 4 renders pixel-identical |

The amplifier was not copied: the text-conditioning captures were collected again with the bundled ComfyUI collector, with the A-side adapter loaded, and the fit ran on the new captures. The spec held the study's 119 triggered prompts, 390 preserved untriggered prompts and 43 style contexts, 592 distinct prompts in all. The new fit report matched the recorded one: the style directions came from the same 3,698 contrast rows, with 0.5913 of the contrast energy in the top four directions and 0.4102 of the trigger offset's energy along them. The trigger offset was 0.4686 of the conditioning norm.

## Rendered check

Space-station prompt with the trigger, seeds 181905–181908. Settings: Krea 2 Turbo FP8 scaled, 832 × 1216, eight Euler/simple steps, CFG 1. The comparison renders are the study's in-process "highlights" renders. Before the comparison, the study's own pipeline re-rendered one of them, and it was still pixel-identical.

| Arm | Loading | Pixel-identical | Mean abs. difference (0–255) per seed |
|---|---|---:|---|
| Repair (C) | blend at 1, amplifier at 6 as separate patches | 4 / 4 | 0, 0, 0, 0 |
| Repair (C) | one packaged file (`compose`) | 0 / 4 | 10.7, 7.9, 5.0, 22.4 |
| Style dial 0 (D) | blend at 1, style-free part at 6 | 4 / 4 | 0, 0, 0, 0 |
| Style dial 0 (D) | one packaged file | 0 / 4 | 7.6, 6.2, 5.1, 12.6 |
| Style dial 3, every step | D + hook | 4 / 4 | 0, 0, 0, 0 |
| Style dial 6, steps 3–7 (E) | D + hook, study's projection form | 4 / 4 | 0, 0, 0, 0 |
| Style dial 6, steps 3–7 | D + hook, fitted style difference | 0 / 4 | 1.7, 2.4, 2.5, 3.4 |

The `detangle render` command with `--project-onto` and the study's directions also reproduced the E render for seed 181906 pixel for pixel.

### Why a packaged file renders differently

`compose` multiplies the dose into the stored difference (6 × dW, float32), which is exactly how the study's own single-file repair was written. The packaged tensors are identical to the study's; only the loading differs. Our best explanation, which we did not chase further: ComfyUI applies a patch by casting the difference to its LoRA compute dtype (bfloat16 here), scaling it by the load strength, and re-rounding the sum into the FP8 weight. Rounding dW and then multiplying by 6 is not the same as rounding 6 × dW, so the baked FP8 weights differ slightly. An 8-step sampler then turns that small weight difference into a different but equally plausible image. In the largest case (seed 181908) the scene and the face are the same; the framing shifts slightly and details such as the collar and patches are drawn differently. Image quality is not degraded.

Expect this whenever an adapter is reorganised before loading: packaging, merging files or changing strengths changes pixels without changing what the repair is. To reproduce the published images exactly, load the parts separately at their doses (blend at 1, amplifier at 6). For your own LoRA, either form is a valid repair.

### Why the scheduled component has two forms

The study applied the style dial in the hook as `6 × ((h dWᵀ) U Uᵀ)`. Here dW is the full amplifier, and U holds the four style directions in the conditioning space. Algebraically this is `6 × h (U Uᵀ dW)ᵀ`, the style part as a weight difference. The precomputed form differs only by float32 rounding: at most 1.1 × 10⁻⁴ in a term of magnitude 290 on random inputs. That still changes pixels after eight steps (mean 1.5–3.4 out of 255). On inspection these are small edge shifts with the same image, so we accept the precomputed form as equivalent and did not investigate further. `Component(..., output_projection=P)` and `render --project-onto` keep the study's operation order when exact pixels matter.

## Numerical notes

- **Amplifier on CPU.** `amp-fit --device cpu` solves the same float64 system with CPU BLAS. It differed from the recorded CUDA fit in 22,076 of 37,748,736 float32 elements. The largest difference was 3.0 × 10⁻⁸ (relative Frobenius 6.3 × 10⁻¹⁰). The CUDA fit took 13.8 s and the CPU fit 63.1 s.
- **Style split.** The separately fitted style and style-free parts add up to the full amplifier within 8.6 × 10⁻⁷ (full-amplifier maximum 5.8). Recomputing the four directions from the fitted style part (CPU SVD) and splitting the full amplifier gives both parts within 1.1 × 10⁻⁶. The style part's fifth singular value is 4.6 × 10⁻⁶ against 15.7 for the fourth, so it has rank four.
- **Blend metadata.** `detangle blend` keeps the repaired adapter's metadata and records provenance, while the study's blend files carried none. Tensors are identical; file hashes differ.

## Collector scope

`collect-text` reproduced the study's text-conditioning captures exactly, as the bit-identical amplifier fits show (592 prompts, 76.7 s, 1.3 GB in bfloat16). `collect-layers` ran on the real model and produced reservoirs for all 256 adapted layers, and `repair` consumed them. That smoke test used two pairs, six flow fractions and 64 rows per layer. It was **not** compared with the historical layer captures. Those came from a training runtime under different conditions, so a ComfyUI capture is a new capture condition, not a replay.

## Inputs and settings

The check used the private character LoRA (rank 32, 256 adapted layers), its saved world-v3 class and trigger captures, and the study's adapters, prompt lists and renders. None of them are distributed. [The machine-readable record](composite-repair-validation.json) lists their hashes, per-file results and timings. Software: Python 3.12; PyTorch 2.10.0 (CPU checks) and 2.14.1 with CUDA 13.0 (GPU checks); ComfyUI commit 87d23b817. Hardware: Ryzen 7 9800X3D, one RTX 5090 (32 GB).

This check establishes that the exported code reproduces one private adapter's composite repair from that adapter's own inputs. It does not show that blocks 12–15, 35 %, × 6 or steps 3–7 are good settings for another LoRA.
