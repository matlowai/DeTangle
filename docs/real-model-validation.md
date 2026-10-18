# Real Krea 2 solver validation

The exported `--method historical` solver reproduced all **256 edited A tensors exactly** from the saved historical capture matrices. Every B tensor and other tensor was unchanged. The original metadata was retained, with new repair provenance added.

One public beach prompt pair was then rendered with both repaired checkpoints. The historical and exported adapters produced **identical decoded RGB pixels and identical face scores** for both prompt states. This is a real checkpoint and image test, separate from the synthetic algebra demo.

| Prompt state | Historical adapter, rerendered | Exported historical mode | Pixel difference |
|---|---:|---:|---:|
| Without trigger | −0.0693126023 | −0.0693126023 | 0 |
| With trigger | 0.6780787706 | 0.6780787706 | 0 |

These are raw ArcFace cosine scores. The earlier published generic image scored −0.0708926469; its historical rerender differs slightly. The published triggered score is identical to the new triggered score. The compatibility comparison uses both adapters rendered in the same current run, and does not claim that every earlier image is bitwise replayable.

## Inputs and settings

The test used the historical source LoRA and its saved class and trigger captures: rank 32, 256 layers, and 1,024 rows per role and layer. The repair used mu 0.3, relative ridge 0.001, zero centering and `--method historical`. Rendering used Krea 2 Turbo, seed 171018, adapter strength 1.25, 832 × 1216 pixels, eight Euler/simple steps and CFG 1.

The test ran the installed wheel. [The machine-readable record](real-model-validation.json) includes the installed source hashes, checkpoint/capture hashes, tensor checks, image comparisons and timing scope. Checkpoint files and captures are not distributed. File hashes differ when provenance metadata differs, even when tensors or decoded image pixels are identical.

## Why numerical mode matters

The initial stable CPU float64 test solved the same objective but did **not** reproduce the historical weights. Its triggered score was 0.6415416002 versus 0.6780787706 for the historical adapter. Investigation identified the original arithmetic: CUDA float32 primal Cholesky with TF32 disabled for input widths up to 8,192, CPU float64 primal solves for larger widths, and a float32 intermediate before checkpoint casting.

The explicit historical mode preserves that path. Stable CPU `--method auto` remains available and chooses the smaller primal or dual system in float64. Higher numerical precision alone does not imply better generated-image likeness. Compare these modes as different numerical settings when the captured system is ill-conditioned.

## Observed computational cost

Hardware: Ryzen 7 9800X3D, eight CPU threads, one RTX 5090 with 32 GB memory. Software: Python 3.12, PyTorch 2.10.0 with CUDA 12.8.

| Stage | Observed wall time | Scope |
|---|---:|---|
| Fresh capture in the follow-up study | 13 min 35 s–14 min 26 s | 72 prompt/latent pairs per role, 216 model forwards; includes initialization and output |
| Stable CPU repair | 1 min 26 s | Reuses saved captures; includes the installed CLI, hashes, fitting diagnostics and output |
| Historical compatibility repair | 4 min 25 s | Same saved inputs; includes the installed CLI, hashes, fitting diagnostics and output |
| Tensor audit, server startup, four images and scoring | 55.4 s | One generic/trigger pair for each of the two adapters |

These are workstation observations, not a controlled throughput benchmark. The solve timings exclude capture collection. A new repair needing fresh captures therefore costs more than a refit of an existing capture. Different layer widths, sample counts, precision, hardware and concurrent work change the cost.

This one-pair check establishes numerical compatibility for this saved checkpoint and capture. It does not establish generalization, an optimal prompt recipe or superior image quality on other tasks.
