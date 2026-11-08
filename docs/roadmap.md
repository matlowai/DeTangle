# Research roadmap (October 2026)

Open directions after the first release. Results that have since landed are in [lessons](lessons.md).

## 1. Interleave measured repair with training

Periodic repair during training remains a proposal. Compare training alone, training with a final repair, and training with periodic repair plus refreshed captures. Report data/step budgets and total wall time, including solve overhead. Keep a fixed preservation reference, handle optimizer state deliberately, and measure persistence after training resumes. Denoising-step window experiments do not establish that this training loop works. Identity evaluation must include multiple people in one image as well as separate trigger/no-trigger tests.

## 2. Entanglement: peeling scene priors and gradient-scored repair (2026-10-19)

(a) Iterative peeling on the base model: remove the dominant actress component (Sigourney Weaver first) from generic sci-fi/space-station prompts while protecting prompts that name her; measure her drift by name, re-run the 19-concept panel to see which identity dominates next, check whether the weakened trigger now wins, repeat. (b) Gentle character repair: partial blend of the original factors in blocks 8–15, gradient-scored selection of the removed pieces per scene with a sign check (after nynxz's gradient-scored selection; see [attributions](attributions.md)), or a recapture with strong-prior triggered prompts. (c) Scene-prior strength (panel fingerprint of a scene's default face) as a predictor of where a LoRA or surgery will fail.

## 3. Next (2026-10-26): back on Krea 2 binding

The MOSS-Image train-time study is shelved ([lessons](lessons.md#train-time-binding-on-moss-image-2026-102526-shelved)). On Krea 2, starting from the bound adapter described in [lessons](lessons.md#binding-names-are-entities-to-the-text-encoder-2026-10-21--10-25):

1. Name gate at inference on the bound file: untriggered bleed → exactly 0; check strangers and solo likeness on the dev set, then a declared frozen-test run.
2. Embedding pre-bind instead of the `wk` key lock, as a candidate fix for celebrity-first sentences (other tokens' keys untouched).
3. Train-time version: pre-bind + routing-frozen (v/o + MLP) retrain, apples-to-apples against the bound file.
