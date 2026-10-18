# How delta surgery works

A LoRA adds a low-rank response to a layer. Ignoring unchanged adapter scale factors for clarity:

```text
delta(x) = B A x
A: [rank, input_width]
B: [output_width, rank]
```

The source adapter may respond strongly both when its trigger is present and when a generic class is requested. Scalar delta surgery refits A using two sets of captured layer inputs; it keeps B fixed. It does not know which neuron or direction means “identity.” The class and trigger samples provide the distinction.

## The conditional least-squares objective

In this code, `Xc` and `Xt` have **samples as rows**. Their columns are input features. Define `Gc = Xc.T @ Xc` and `Gt = Xt.T @ Xt`. With `A0` the ridge prior:

```text
minimize over A':
    ||A' Xc.T||_F²
  + mu * ||(A' - A) Xt.T||_F²
  + epsilon * ||A' - A0||_F²

epsilon = ridge * trace(Gc + mu * Gt) / input_width

A' (Gc + mu * Gt + epsilon I) = mu * A Gt + epsilon * A0
```

The first term suppresses the sampled class response. The second preserves the source adapter's sampled triggered response. The last stabilizes the solution and supplies a prior. The implementation solves this linear system; it never explicitly inverts the Gram matrix. It is a conditional refit, not generally an orthogonal projection or a semantic deletion of a few “bad weights.”

The historical setting is `A0 = 0` (`--center zero`):

```text
A' = mu * A Gt (Gc + mu * Gt + epsilon I)^(-1)
```

For a direction absent from both captured sample sets, this prior drives the fitted A response to zero. With the experimental `--center source`, `A0 = A`, so a direction absent from both sets retains its original A response. The included synthetic demo makes this distinction visible. Neither prior guarantees useful behavior on new prompts.

The losses are sums over sample rows. Unequal class and trigger sample counts therefore change their effective balance. Repeating both reservoirs equally leaves the solution unchanged because epsilon scales with their total energy. Repeating only one role generally changes it. Token normalization, timestep frequencies and prompt repetition are choices in the experiment, not neutral preprocessing.

## Three different knobs

| Setting | What changes | Historical illustration |
|---|---|---|
| `mu` | Relative weight on preserving the source trigger response during the refit | 0.3 |
| `ridge` | Relative regularization; converted to epsilon using observed activation energy | 0.001 |
| Render strength | Multiplier applied when loading the finished adapter for generation | Scalar shown at 1.25 |

Changing strength does not rerun the surgery and is not equivalent to changing mu. Changing mu changes a matrix-valued refit across measured directions; strength scales its final contribution. Different methods can need different strengths. The historical adapter has rank 32 and alpha 32; the portable solver preserves any stored scaling tensors.

## Why local fit and generated behavior can disagree

Numerical arithmetic is also part of the experiment. `--method historical`
reproduces the original primal Cholesky path: CUDA float32 with TF32 disabled
for input widths up to 8,192, CPU float64 for larger widths, and a float32
intermediate before restoring the checkpoint dtype. The default `--method auto`
uses CPU float64 and chooses the smaller primal or dual system. Both solve the
same stated objective, but ill-conditioned captures can produce materially
different weights under these arithmetic choices. Use the recorded numerical
mode when reproducing a checkpoint; a more accurate linear solve does not by
itself establish better rendered likeness.

The optimization is in the A-response space. Evaluation also measures the full `B A` response because B weights directions differently. Reported response retention is `||B A' X.T|| / ||B A X.T||`; relative change is `||B (A'-A) X.T|| / ||B A X.T||`. Cosine compares those flattened output responses. A zero original response yields an undefined ratio, recorded as null.

These measurements use fixed captured inputs. If many layers are edited, earlier edits change the inputs later layers receive. A very good fit on the original capture can coexist with a different end-to-end trajectory. Shared class/trigger directions, incomplete prompt coverage, token sampling variance and the ridge prior are plausible contributors. A rendered failure alone does not identify which one caused it.

## DOP is a different intervention

Differential Output Preservation (DOP), as used in these comparisons, is a training-time functional constraint. Alongside the usual adaptation loss, the trained model is asked to match the frozen base model's prediction for class-conditioned inputs:

```text
L = L_adaptation
  + lambda * ||f_adapted(z_t, t, class_caption)
               - stop_gradient(f_base(z_t, t, class_caption))||²
```

The two class predictions must use the same noisy latent and timestep. The base prediction is computed with the adapter disabled. Caption construction matters: replacing a trigger must yield the intended class caption. This seed explains DOP for comparison; it does not ship the original training stack.

DOP protects sampled **final predictions**, whereas surgery suppresses sampled **local adapter responses**. A DOP model can retain intermediate activity that cancels or compensates downstream. Applying a local suppressive repair afterward may disturb that computation. This is a reason to test compositions, not a proof that all compositions fail.

LoRA, least-squares editing and output preservation have broader prior art. This project presents an inspectable implementation and measured case studies; it does not claim to invent low-rank adaptation or linear least squares.
