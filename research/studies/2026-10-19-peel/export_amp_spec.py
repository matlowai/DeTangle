#!/usr/bin/env python3
"""Write the trigger amplifier's as-run prompt sets as a `detangle amp-fit` spec (added for publication).

Rebuilds exactly the lists trigger_amp.py `fit` / `fit-variant` used: the triggered source prompts, the
untriggered preserve prompts and the style-direction contexts with the snapshot/cinematic sentence pair.
The default variant gives 119 sources, 390 preserved prompts and 43 style contexts, the inputs of the
published amplifier, its style dial and the style-free rest. No model or GPU is needed.

  python export_amp_spec.py --out amplifier-spec.json                     # as run for the repair (C/D/E)
  python export_amp_spec.py --variant genres2 --out genres2-spec.json      # genre-capture amplifier
  python export_amp_spec.py --controls --out spec-with-controls.json       # the "_spec" style variant

Then (see docs/composite-repair-validation.md):
  detangle collect-text ... --lora a_side.safetensors --spec amplifier-spec.json --out caps.safetensors
  detangle amp-fit --spec amplifier-spec.json --captures caps.safetensors --part full|style|rest --out amp.safetensors
"""
import argparse, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import peel  # noqa: E402
import trigger_amp as ta  # noqa: E402
from peel import name, gen, PANEL, OTHER  # noqa: E402


def spec(variant="aside", controls=False):
    trig = ta.TRIG
    scifi_fit, _, scenes, general_pre, _, men = peel.prompt_sets()
    fantasy = gen(ta.FANTASY_PLACES, ta.FANTASY_STYLE, 11, per=2)
    src = [trig + t for t in list(scifi_fit.values()) + general_pre]
    capture = [f"A close-up head-and-shoulders photograph of a woman {p}, facing the camera. {peel.FACE} She wears {o}. {b} her. {st}"
               for p, o, b, st in ta.CAPTURE_GENRES]
    if variant in ("genres", "genres2"):
        src += [trig + t for t in capture]
    rng = random.Random(7)  # same draw order as trigger_amp.fit
    preserve = list(scifi_fit.values()) + general_pre + men + OTHER
    preserve += [name(t, w) for t in list(scifi_fit.values()) + general_pre for w in rng.sample(PANEL, 2)]
    if variant in ("genres", "genres2"):
        preserve += capture
    if variant == "genres2":
        preserve += [f"A close-up head-and-shoulders photograph of a woman {p}, facing the camera. {peel.FACE} She wears {o}. {b} her. {st}"
                     for p, o, b, st in ta.PRESERVE_GENRES]
    preserve = list(dict.fromkeys(t for t in preserve if t != scenes["wizard_workshop"]))
    contexts = [trig + t for t in list(scifi_fit.values())[::3] + general_pre[::4] + fantasy[::2]]
    style = {"contexts": contexts, "a": ta.SNAP, "b": ta.CINE, "k": 4}
    if controls:
        style["controls"] = [list(ta.DIAG_PAIRS[c]) for c in ("shirt_red_vs_blue", "happy_vs_sad")]
    return {"schema": "detangle-amplifier-spec/1", "trigger": trig, "sources": src, "preserve": preserve, "style": style}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["aside", "genres", "genres2"], default="aside")
    ap.add_argument("--controls", action="store_true", help="add the unrelated-sentence control pairs (nostyle_spec / style_spec)")
    a = ap.parse_args()
    s = spec(a.variant, a.controls)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    with out.open("w") as f:
        json.dump(s, f, indent=1, ensure_ascii=False)
    print(len(s["sources"]), "sources,", len(s["preserve"]), "preserved,", len(s["style"]["contexts"]), "style contexts")
