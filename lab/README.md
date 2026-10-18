# DeltaScope

A local app for comparing adapter outputs, scoring experiments and reviewing case studies.

DeltaScope is the lightweight comparison and scoring app in **DeTangle**. It is model-agnostic at the dataset level: conditions can be LoRAs, other adapters, checkpoints, or experimental variants. The initial built-in studies focus on character LoRAs. Face similarity is one optional metric, not a general mechanistic explanation.

## View the built-in cases

From a DeTangle checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./lab
deltascope serve
```

Open the printed loopback URL, normally `http://127.0.0.1:8890`. Viewing needs no PyTorch, face model, GPU, or private checkpoint. The app does not contact a remote service.

The bundled case contains beach, arcade, lighthouse, space-station and wizard-workshop comparisons, two seeds, matched trigger states and three strengths for Naive, scalar surgery and DOP, plus a shared base. It opens on the beach with Naive 1.0 / scalar 1.25 / Base / DOP 1.0. Add or remove columns, compare the same method at different strengths, inspect both prompt states, use an A/B wipe, and open method explanations. Selections are saved in the URL. Browser Back and Forward restore previous tabs, cases and comparison settings. The Research guide explains the objective, methods, settings and performance. Dig more presents aggregate historical findings without distributing that archive.

Case-study scores were measured on native 832 × 1216 images. Bundled JPEGs are resized display copies. The private LoRAs, reference photos and face embeddings are not included; the original identity scores cannot be independently recomputed without that reference set. Prompts, seeds, sampling settings, native image hashes and limitations are provided.

## Bring your own images and measurements

Arrange images beneath a project directory and create a CSV there:

```csv
condition,case,axis,prompt,seed,strength,image,score.my_metric
base,portrait,generic,a person in a blue shirt,42,1,images/base.png,0.05
adapter,portrait,generic,a person in a blue shirt,42,1,images/adapter.png,0.42
```

`axis` is normally `generic` or `target`, but other named states work too. A case can have one or several exact prompts. Seeds are kept as exact strings, including values larger than JavaScript's safe integer range. `score.*` columns are optional; import a dataset before running a local evaluator.

Create `protocol.json` describing the sampling settings that make these comparisons comparable:

```json
{"id":"my-protocol","label":"My fixed renderer","matchable":true,"settings":{"width":832,"height":1216,"steps":8,"cfg":1,"sampler":"euler","scheduler":"simple"}}
```

Then:

```bash
deltascope import-csv images.csv --out dataset.json --baseline base --protocol protocol.json
deltascope validate dataset.json
deltascope serve dataset.json
```

Images and reference files must be inside the dataset directory, using relative paths. The output JSON is written beside the CSV so paths keep their meaning. A protocol omitted during CSV import is marked unverified and does **not** receive automatic baseline subtraction. Do not reuse a protocol ID across incompatible settings merely to obtain a paired score.

For richer condition descriptions, method explanations and lessons, use the bundled JSON as the schema example. Change only your own dataset; bundled case studies are read-only.

## Score character likeness locally

Install the optional evaluator dependencies. InsightFace 0.7.3 can require compiler/build tooling on platforms without a compatible wheel; it is the version used for the calibrated historical pipeline.

```bash
python -m pip install -e './lab[face,images]'
deltascope references dataset.json --folder refs
deltascope score dataset.json --evaluator face_similarity --face-model-dir models/buffalo_l
```

Supply the detector `det_10g.onnx` and recognizer `w600k_r50.onnx` yourself. **No weights are downloaded automatically.** See [NOTICES.md](NOTICES.md) for the distinction between InsightFace's code license and pretrained-model terms. The UI can launch the same scoring job when started with `--face-model-dir`.

The evaluator uses CPU ONNX Runtime, 640 × 640 detection, confidence threshold 0.5, the largest detected face by bounding-box area, unit ArcFace embeddings, and a normalized mean reference centroid. It records model/reference hashes and runtime versions. It also retains the maximum similarity across faces as a diagnostic. Duplicate reference images are rejected to avoid accidental weighting.

A missing face produces `null` with `no_face` status. It is never scored as zero. Errors are recorded and are retried rather than cached. The local SQLite cache is keyed by image, baseline, evaluator, reference/model provenance, render protocol and input fields; changed evidence invalidates cached results. Locally scored face measurements also track the reference files: editing or replacing that set marks those scores stale until they are rescored. The dataset is updated atomically when scoring completes.

## Other measurements

```bash
python -m pip install -e './lab[images]'
deltascope score dataset.json --evaluator image_rmse
```

RGB RMSE compares each image to its unique matched base at the same size. It measures pixel differences, not semantic preservation or visual quality. Imported CSV metrics support other evaluators, and a local Python plugin can implement the same interface:

```python
class Evaluator:
    metric_id = "my_measurement"
    signature = "my-evaluator-v1-with-reference-and-model-hashes"
    provenance = {"evaluator": "Explain the measurement and its limits"}

    def score(self, image_path, record):
        return {"value": 0.5, "status": "ok", "details": {}}

def create(dataset, project_root):
    return Evaluator()
```

Run an explicitly selected local plugin with `--evaluator my_module:create`. Plugin code is trusted Python supplied by the operator; the browser API cannot load arbitrary plugins. Include all evaluator settings, model and reference fingerprints in its signature. Valid completed statuses are `ok`, `no_face`, `missing`, and `error`.

## Static export / GitHub Pages

```bash
python -m pip install -e './lab[images]'
deltascope export dataset.json --out site
```

Open `site/index.html` or serve that directory. The export uses relative assets, so it also works under a GitHub Pages project subpath. It contains selected display images, prompts, settings, scores and lessons. Image metadata is stripped. Reference images, checkpoints, local caches, evaluator internals and scoring endpoints are excluded. The export is explicitly display-only; measurements continue to refer to the original images, whose hashes are retained.

Only publish datasets and render images you intend to share. A private GitHub repository does not automatically make its Pages site private. The DeTangle Pages workflow is manual so a clean preview can be reviewed before activation.

## Checks and scope

```bash
PYTHONPATH=lab/src python -m unittest discover -s lab/tests -v
```

The tests cover pairing, missing measurements, large seeds, ambiguous baselines, path confinement, cache invalidation, and sanitized export.

An optional browser regression checks Back/Forward navigation against a running app or static export. From the repository root, with Node.js installed:

```bash
npm install --no-save --package-lock=false playwright
npx playwright install chromium
DELTASCOPE_URL=http://127.0.0.1:8890/ node lab/tests/browser_navigation.cjs
```

The optional face path is calibrated separately against twelve native research renders, including detector misses. The face score measures embedding similarity, not human-rated likeness, overall image quality or a causal mechanism.

MIT license, copyright MatlowAI. Model weights retain their own terms.
