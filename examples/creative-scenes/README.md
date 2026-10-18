# Four creative comparison scenes

These generated portraits are selected examples from an exploratory comparison on 17 October 2026. Every sheet has the same columns: **Naive 1.0 · Scalar surgery 1.25 · Base · DOP 1.0**. The top row omits the trigger and reports similarity above its matched base. The bottom row includes the trigger and reports raw likeness. A base image is freshly matched to each exact prompt and seed.

| Scene | Seed 171017 | Seed 171018 |
|---|---|---|
| Neon arcade | [Compare](images/arcade-171017.jpg) | [Compare](images/arcade-171018.jpg) |
| Stormy lighthouse | [Compare](images/lighthouse-171017.jpg) | [Compare](images/lighthouse-171018.jpg) |
| Space station | [Compare](images/space_station-171017.jpg) | [Compare](images/space_station-171018.jpg) |
| Wizard workshop | [Compare](images/wizard_workshop-171017.jpg) | [Compare](images/wizard_workshop-171018.jpg) |

- [settings.json](settings.json) contains all eight exact prompts, seeds, rendering setup, default columns and audited historical scan settings. The triggered prompt adds only a leading `rlsa, ` prefix; both roles retain the woman class cue.
- [scores.json](scores.json) retains all **160 native-image scores**, including alternate strengths 1.0, 1.25 and 1.5. Each row has the original image hash, its matched base hash, detection count and measurement scope. The 64 illustrated rows also identify their sheet position.
- [default-summary.json](default-summary.json) contains the two-seed means for the illustrated operating points, with denominators.
- [provenance.json](provenance.json) records model/adapter content hashes, the score protocol and exported-sheet hashes without machine paths or private reference identifiers.

Scores belong to the original 832 × 1216 images. Contact-sheet images were decoded, resized and recompressed without their original image metadata; rescoring these JPEGs would be a different measurement. All 160 native originals had a detected face. No similarity is a percentage, and negative excess is not evidence of superior general preservation.

The private source adapters and 34-image reference set are excluded. These assets illustrate observed behavior and preserve the settings needed to understand it; they do not make the exact historical experiment independently reproducible. The repository's synthetic solver demo is independently runnable.

See the [case study](../../docs/case-study.md) for interpretation and the controlled follow-up questions.
