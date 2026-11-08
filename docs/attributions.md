# Attributions ledger

Ideas from other people's work that we use, where we use them, and how we credit them. Add a line every time one is used. Code is never copied unless its license allows it and that is written here.

## Heretic 2.0 (p-e-w; Arbitrary-Rank Ablation, Weidmann 2026, preprint forthcoming)

Source: https://github.com/p-e-w/heretic (AGPL-3.0-or-later), read at commit `0db5c31e`, design discussion PR #211.

Policy: we use Heretic's math only, never its code or configuration files, and credit it wherever we use an idea we had not already investigated or put on the roadmap before 2026-11-01. Ideas that would need credit if used include its ARA objective package, row-norm preservation and the overcorrect term.

| date | idea | where we use it | credit placed in |
|---|---|---|---|
| 2026-11-01 | (none needed) kNN key targets: a general technique already in Dylan's own idea space | kNN / name-span pre-bind test (negative result) | Heretic named as related work in the run summary |

## Gradient-scored edit selection (nynxz)

| idea | where we use it | credit |
|---|---|---|
| scoring each rank-1 piece of a fine-tuning change by the gradient of the unwanted behaviour, then keeping only the pieces that carry it (nynxz's MOSS-Image destripe LoRA) | [lessons](lessons.md#select-edits-by-measured-effect-cross-project-from-moss) and the proposed gentle character repair ([roadmap](roadmap.md#2-entanglement-peeling-scene-priors-and-gradient-scored-repair-2026-10-19)) | named in both places |

## Refusal-direction / representation-engineering prior art

| idea | cite |
|---|---|
| last-prompt-token (decision-position) capture for LLM directions | Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction" (and the abliteration line of work) |
