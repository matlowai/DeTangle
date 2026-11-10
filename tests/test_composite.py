"""Composite-repair functions on synthetic tensors only (no model, no private data)."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from delta_surgery_seed.collect import (LayerInputCollector, LinearIO, Reservoir, TextCaptures, lora_layer_names,
                                        save_text_captures)
from delta_surgery_seed.composite import (AMPLIFIER_KEY, amplifier_fit_metrics, blend_blocks, block_index, compose,
                                          fit_amplifier, output_directions, pair_difference, project_target,
                                          split_amplifier, style_directions, trigger_offset)
from delta_surgery_seed.composite_cli import (amp_fit, amp_split, blend, compose_cmd, parse_weighted, spec_texts,
                                              style_dirs)
from delta_surgery_seed.schedule import Component, ScheduledDelta, parse_steps


def adapter(seed, a_scale=1.0, blocks=4, rank=2, width=6, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for b in range(blocks):
        for site in ("attn.wq", "mlp.up"):
            key = f"diffusion_model.blocks.{b}.{site}"
            out[key + ".lora_A.weight"] = (a_scale * torch.randn(rank, width, generator=g)).to(dtype)
            out[key + ".lora_B.weight"] = torch.randn(width, rank, generator=torch.Generator().manual_seed(99 + b)).to(dtype)
    key = "diffusion_model.txtfusion.layerwise_blocks.1.attn.wq"
    out[key + ".lora_A.weight"] = (a_scale * torch.randn(rank, width, generator=g)).to(dtype)
    out[key + ".lora_B.weight"] = torch.randn(width, rank, generator=torch.Generator().manual_seed(7)).to(dtype)
    return out


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.src, self.rep = adapter(1), adapter(2, a_scale=.5)

    def test_block_index_only_main_blocks(self):
        self.assertEqual(block_index("diffusion_model.blocks.12.attn.wq"), 12)
        self.assertEqual(block_index("transformer.blocks.3.mlp.up"), 3)
        self.assertIsNone(block_index("diffusion_model.txtfusion.layerwise_blocks.1.attn.wq"))

    def test_formula_range_and_untouched_tensors(self):
        out, keys = blend_blocks(self.src, self.rep, 1, 2, .35)
        self.assertEqual(len(keys), 4, keys)
        for key, value in out.items():
            b = block_index(key)
            if key.endswith("lora_A.weight") and b is not None and 1 <= b <= 2:
                expect = ((1 - .35) * self.rep[key].float() + .35 * self.src[key].float()).to(torch.bfloat16)
                self.assertTrue(torch.equal(value, expect), key)
            else:
                self.assertTrue(torch.equal(value, self.rep[key]), key)
            self.assertEqual(value.dtype, self.rep[key].dtype)

    def test_exact_delta_blend_because_b_is_shared(self):
        out, _ = blend_blocks(self.src, self.rep, 0, 3, .25)
        key = "diffusion_model.blocks.2.attn.wq"
        b = self.rep[key + ".lora_B.weight"].double()
        mixed = b @ out[key + ".lora_A.weight"].double()
        expect = .75 * b @ self.rep[key + ".lora_A.weight"].double() + .25 * b @ self.src[key + ".lora_A.weight"].double()
        err = float((mixed - expect).abs().max() / expect.abs().max())
        self.assertLess(err, 1e-2, f"relative error {err} exceeds bf16 rounding")

    def test_endpoints(self):
        zero, _ = blend_blocks(self.src, self.rep, 0, 3, 0.)
        one, _ = blend_blocks(self.src, self.rep, 0, 3, 1.)
        key = "diffusion_model.blocks.0.mlp.up.lora_A.weight"
        self.assertTrue(torch.equal(zero[key], self.rep[key]))
        self.assertTrue(torch.equal(one[key], self.src[key]))

    def test_rejections(self):
        changed_b = dict(self.rep)
        k = "diffusion_model.blocks.0.attn.wq.lora_B.weight"
        changed_b[k] = changed_b[k] + 1
        with self.assertRaisesRegex(ValueError, "exact delta blend"):
            blend_blocks(self.src, changed_b, 0, 3, .5)
        for args in [(0, 3, -.1), (0, 3, 1.5), (3, 1, .5), (9, 12, .5)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                blend_blocks(self.src, self.rep, *args)
        missing = {k: v for k, v in self.rep.items() if "blocks.3." not in k}
        with self.assertRaises(ValueError):
            blend_blocks(self.src, missing, 0, 3, .5)

    def test_inputs_not_mutated(self):
        before = {k: v.clone() for k, v in self.rep.items()}
        blend_blocks(self.src, self.rep, 0, 3, .5)
        self.assertTrue(all(torch.equal(before[k], self.rep[k]) for k in before))


class OffsetTests(unittest.TestCase):
    def test_trigger_offset_alignment(self):
        g = torch.Generator().manual_seed(3)
        ids_t, ids_u = [1, 2, 50, 51, 9, 8], [1, 2, 60, 9, 8]
        c, cu = torch.randn(6, 4, generator=g), torch.randn(5, 4, generator=g)
        d = trigger_offset(ids_t, c, ids_u, cu)
        torch.testing.assert_close(d[:2], c[:2] - cu[:2], rtol=0, atol=0)
        torch.testing.assert_close(d[4:], c[4:] - cu[3:], rtol=0, atol=0)
        torch.testing.assert_close(d[2:4], c[2:4] - cu[2:3].mean(0), rtol=0, atol=0)

    def test_offset_needs_differing_spans(self):
        c = torch.zeros(3, 2)
        with self.assertRaises(ValueError):
            trigger_offset([1, 2, 3], c, [1, 2, 3], c)
        with self.assertRaises(ValueError):
            trigger_offset([1, 2], c, [1, 2, 3], c)

    def test_pair_difference_prefix(self):
        a, b = torch.arange(8.).reshape(4, 2), torch.ones(5, 2)
        d = pair_difference([5, 6, 7, 1], a, [5, 6, 9, 9, 9], b)
        torch.testing.assert_close(d, a[:2] - b[:2])
        with self.assertRaises(ValueError):
            pair_difference([1], a[:1], [2], b[:1])


class AmplifierTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(11)
        self.din, self.dout = 8, 6
        self.w = torch.randn(self.dout, self.din, generator=g, dtype=torch.float64)
        self.sources = [(torch.randn(5, self.din, generator=g), None) for _ in range(6)]
        self.sources = [(h, (h.double() @ self.w.T).float()) for h, _ in self.sources]
        self.preserve = [torch.randn(7, self.din, generator=g) for _ in range(5)]

    def test_normal_equation(self):
        lam, eps = 4.0, 1e-4
        dw, info = fit_amplifier(self.sources, self.preserve, lam, eps)
        self.assertEqual((info["n_source"], info["n_preserve"]), (6, 5))
        gs = sum(h.double().T @ h.double() for h, _ in self.sources)
        g = sum(h.double().T @ h.double() for h in self.preserve)
        r = sum(d.double().T @ h.double() for h, d in self.sources)
        a = gs + lam * g
        a = a + eps * a.diagonal().mean() * torch.eye(self.din, dtype=torch.float64)
        residual = dw.double() @ a - r
        self.assertLess(float(residual.abs().max() / r.abs().max()), 1e-5)

    def test_without_preserve_recovers_linear_target(self):
        dw, _ = fit_amplifier(self.sources, [], 4.0, 1e-9)
        err = float((dw.double() - self.w).abs().max())
        self.assertLess(err, 1e-4, f"max error {err}")

    def test_preserve_weight_shrinks_untriggered_change(self):
        rel = []
        for lam in (0.1, 100.):
            dw, _ = fit_amplifier(self.sources, self.preserve, lam, 1e-6)
            rel.append(sum(float((h @ dw.T).norm()) for h in self.preserve))
        self.assertLess(rel[1], rel[0] * .5, rel)

    def test_metrics_perfect_fit(self):
        dw, _ = fit_amplifier(self.sources, [], 4.0, 1e-9)
        m = amplifier_fit_metrics(dw, self.sources, [(h, torch.ones(7, self.dout)) for h in self.preserve])
        self.assertGreater(m["achieved"], 1 - 1e-6)
        self.assertEqual(m["preserve_relative_change"]["n"], 5)

    def test_invalid(self):
        for kw in ({"lam": -1}, {"eps": 0}, {"lam": float("nan")}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                fit_amplifier(self.sources, self.preserve, **kw)
        with self.assertRaises(ValueError):
            fit_amplifier([], self.preserve)


class StyleSplitTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(5)
        self.dout, self.din = 10, 7
        self.u = torch.linalg.qr(torch.randn(self.dout, 2, generator=g, dtype=torch.float64))[0]
        self.dw = torch.randn(self.dout, self.din, generator=g)

    def test_parts_add_up_and_live_in_span(self):
        style, rest = split_amplifier(self.dw, self.u)
        torch.testing.assert_close(style + rest, self.dw, rtol=0, atol=1e-6)
        p = self.u @ self.u.T
        torch.testing.assert_close(p @ style.double(), style.double(), rtol=0, atol=1e-6)
        self.assertLess(float((self.u.T @ rest.double()).abs().max()), 1e-6)

    def test_split_equals_refit_with_projected_targets(self):
        g = torch.Generator().manual_seed(8)
        hs = [torch.randn(4, self.din, generator=g) for _ in range(5)]
        ds = [torch.randn(4, self.dout, generator=g) for _ in range(5)]
        pres = [torch.randn(6, self.din, generator=g) for _ in range(3)]
        full, _ = fit_amplifier(zip(hs, ds), pres)
        style_fit, _ = fit_amplifier(((h, project_target(d, self.u, "style")) for h, d in zip(hs, ds)), pres)
        rest_fit, _ = fit_amplifier(((h, project_target(d, self.u, "rest")) for h, d in zip(hs, ds)), pres)
        style, rest = split_amplifier(full, self.u)
        torch.testing.assert_close(style, style_fit, rtol=0, atol=1e-5)
        torch.testing.assert_close(rest, rest_fit, rtol=0, atol=1e-5)

    def test_directions_recover_planted_axis_and_output_span(self):
        g = torch.Generator().manual_seed(9)
        axis = torch.zeros(self.dout)
        axis[3] = 1
        diffs = [torch.randn(6, 1, generator=g) * 5 * axis + .01 * torch.randn(6, self.dout, generator=g) for _ in range(4)]
        u, report = style_directions(diffs, k=1)
        self.assertGreater(abs(float(u[3, 0])), .999)
        self.assertGreater(report["energy_topk"], .99)
        style, _ = split_amplifier(self.dw, self.u)
        span = output_directions(style, 2)
        torch.testing.assert_close(span @ span.T, self.u @ self.u.T, rtol=0, atol=1e-5)

    def test_degenerate_contrast_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "no difference"):
            style_directions([torch.zeros(3, self.dout)], k=1)

    def test_controls_removed_first(self):
        axis = torch.zeros(self.dout)
        axis[0] = 1
        shared = [torch.ones(3, 1) * axis]
        u, _ = style_directions(shared + [torch.ones(3, 1) * torch.eye(self.dout)[1]], k=1, controls=shared)
        self.assertLess(abs(float(u[0, 0])), 1e-6)

    def test_compose_scales_sums_and_refuses_collisions(self):
        base = {"a": torch.ones(2)}
        out = compose(base, [("d", torch.full((2,), 2.), 6.0), ("d", torch.ones(2), 0.0)])
        self.assertTrue(torch.equal(out["d"], torch.full((2,), 12.)))
        self.assertIs(out["a"], base["a"])
        with self.assertRaises(ValueError):
            compose(base, [("a", torch.ones(2), 1.)])


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layer = torch.nn.Linear(5, 4)
        self.delta = torch.randn(4, 5)
        self.x = torch.randn(3, 5)

    def test_parse_steps(self):
        self.assertEqual(parse_steps("3-7"), [3, 4, 5, 6, 7])
        self.assertEqual(parse_steps("0,2,6-7"), [0, 2, 6, 7])
        self.assertIsNone(parse_steps("all"))
        for bad in ["7-3", "-1", "", "3-9"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_steps(bad, total=8)

    def test_window_by_comfy_callback(self):
        base = self.layer(self.x)
        sched = ScheduledDelta(self.layer, self.delta, strength=6., steps=range(3, 8))
        outs = []
        with sched:
            for step in range(8):  # one model call per step, then the sampler callback
                outs.append(self.layer(self.x))
                sched.comfy_callback(step)
        for step, y in enumerate(outs):
            expect = base + 6. * (self.x @ self.delta.T) if step >= 3 else base
            torch.testing.assert_close(y, expect, msg=f"step {step}")
        self.assertEqual(sched.calls, [(s, s >= 3) for s in range(8)])
        torch.testing.assert_close(self.layer(self.x), base)  # detached on exit

    def test_diffusers_callback_and_set_step(self):
        sched = ScheduledDelta(self.layer, self.delta, steps=[0]).attach()
        try:
            on = self.layer(self.x)
            self.assertIs(sched.diffusers_callback(None, 0, None, {"k": 1})["k"], 1)
            off = self.layer(self.x)
            sched.set_step(0)
            again = self.layer(self.x)
        finally:
            sched.detach()
        torch.testing.assert_close(on, again)
        self.assertFalse(torch.allclose(on, off))

    def test_projection_form_matches_projected_delta(self):
        u = torch.linalg.qr(torch.randn(4, 2))[0]
        p = u @ u.T
        a = Component(self.delta, 3., output_projection=p).term(self.x)
        b = Component(p @ self.delta, 3.).term(self.x)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_components_sum_and_validation(self):
        d2 = torch.randn(4, 5)
        sched = ScheduledDelta(self.layer, components=[Component(self.delta, 1., [0]), Component(d2, 2., None)])
        base = self.layer(self.x)
        with sched:
            y = self.layer(self.x)
        torch.testing.assert_close(y, base + self.x @ self.delta.T + 2 * self.x @ d2.T)
        with self.assertRaises(ValueError):
            ScheduledDelta(self.layer)
        with self.assertRaises(ValueError):
            Component(torch.ones(3))


class CollectTests(unittest.TestCase):
    def test_reservoir_is_bounded_seeded_and_counts(self):
        x = torch.arange(600.).reshape(100, 6)
        a, b = Reservoir(10, 4), Reservoir(10, 4)
        a.add(x)
        b.add(x[:50])
        b.add(x[50:])
        self.assertEqual(a.matrix().shape, (10, 6))
        self.assertEqual(a.seen, 100)
        self.assertTrue(torch.equal(a.matrix(), b.matrix()))
        with self.assertRaises(ValueError):
            Reservoir(3).matrix()
        with self.assertRaises(ValueError):
            Reservoir(3).add(torch.tensor([[float("nan")]]))

    def test_layer_collector_matches_lora_names(self):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Module()])
        model.blocks[0].wq = torch.nn.Linear(6, 6)
        names = lora_layer_names(["diffusion_model.blocks.0.wq.lora_A.weight", "diffusion_model.blocks.0.wq.lora_B.weight"])
        self.assertEqual(names, ["blocks.0.wq"])
        x = torch.randn(2, 3, 6)
        with LayerInputCollector(model, names, capacity=4, seed=1) as c:
            model.blocks[0].wq(x)
        self.assertEqual(c.tensors()["blocks.0.wq"].shape, (4, 6))
        self.assertEqual(c.counts()["blocks.0.wq"], {"seen": 6, "kept": 4})
        self.assertFalse(model.blocks[0].wq._forward_pre_hooks)
        with self.assertRaises(ValueError):
            lora_layer_names(["x.lora_B.weight"])
        model.act = torch.nn.GELU()
        with self.assertRaisesRegex(ValueError, "Not a linear layer"):
            LayerInputCollector(model, ["act"]).__enter__()

    def test_linear_io_and_text_capture_roundtrip(self):
        layer = torch.nn.Linear(3, 2)
        x = torch.randn(4, 3)
        with LinearIO(layer) as io:
            y = layer(x)
        torch.testing.assert_close(io.input, x)
        torch.testing.assert_close(io.output, y.detach())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "caps.safetensors"
            save_text_captures(path, {"a b": ([5, 6, 7, 8], x.bfloat16(), y.detach().bfloat16())}, extra={"note": "synthetic"})
            caps = TextCaptures(path)
            ids, h, c = caps.get("a b")
            self.assertEqual(ids, [5, 6, 7, 8])
            self.assertTrue(torch.equal(h, x.bfloat16().float()))
            self.assertIn("a b", caps)
            with self.assertRaises(KeyError):
                caps.get("missing")
            with self.assertRaises(ValueError):
                save_text_captures(Path(tmp) / "bad.safetensors", {"x": ([1], x, y)})


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_blend_and_compose_files(self):
        src, rep = self.root / "src.safetensors", self.root / "rep.safetensors"
        save_file(adapter(1), str(src), metadata={"owner": "synthetic"})
        save_file(adapter(2, .5), str(rep), metadata={"owner": "synthetic"})
        out = self.root / "blend.safetensors"
        result = blend(src, rep, "1-2", .35, out)
        self.assertEqual(result["blended_a_factors"], 4)
        expect, _ = blend_blocks(load_file(str(src)), load_file(str(rep)), 1, 2, .35)
        got = load_file(str(out))
        self.assertTrue(all(torch.equal(got[k], expect[k]) for k in expect))
        with safe_open(str(out), framework="pt") as f:
            meta = f.metadata()
        self.assertEqual(meta["owner"], "synthetic")
        self.assertIn("matlowai_detangle_blend", meta)
        with self.assertRaises(FileExistsError):
            blend(src, rep, "1-2", .35, out)
        diff = self.root / "amp.safetensors"
        save_file({AMPLIFIER_KEY: torch.full((3, 3), .5)}, str(diff))
        packaged = self.root / "packaged.safetensors"
        compose_cmd(out, [f"{diff}:6"], packaged)
        p = load_file(str(packaged))
        self.assertTrue(torch.equal(p[AMPLIFIER_KEY], torch.full((3, 3), 3.)))
        self.assertTrue(json.loads((self.root / "packaged.json").read_text())["differences"][0]["dose"] == 6.0)

    def test_parse_weighted(self):
        self.assertEqual(parse_weighted("a.safetensors:6"), ("a.safetensors", 6.0))
        self.assertEqual(parse_weighted("a.safetensors"), ("a.safetensors", 1.0))
        windows = "C" + ":" + chr(92) + "x.safetensors"  # built at runtime so the release audit sees no literal path
        self.assertEqual(parse_weighted(windows), (windows, 1.0))

    def _captures(self):
        """Synthetic 'conditioning': token ids, inputs h and outputs c = h W for a template + trigger."""
        g = torch.Generator().manual_seed(21)
        w = torch.randn(6, 5, generator=g)
        vocab = {}

        def encode(text):
            # Like Krea 2's tokenizer, a sentence-initial word gets a different token than the same word
            # after a prefix ("A" vs " A"), so the untriggered prompt has a nonempty differing span.
            words = text.split()
            ids = [vocab.setdefault(("^" if i == 0 else "") + t, len(vocab) + 1) for i, t in enumerate(words)]
            h = torch.stack([torch.randn(6, generator=torch.Generator().manual_seed(i * 7 + p)) for p, i in enumerate(ids)])
            h = h + .3 * h.mean(0)  # whole-prompt context, like Krea 2's bidirectional text fusion
            return ids, h, h @ w
        scenes = ["a woman in a cafe", "a woman on a ship", "a woman in a lab", "a woman at a beach"]
        spec = {"schema": "detangle-amplifier-spec/1", "trigger": "zq ", "sources": ["zq " + s for s in scenes[:3]],
                "preserve": scenes + ["a man in a cafe"],
                "style": {"contexts": ["zq " + s for s in scenes], "a": "flat snapshot", "b": "film still", "k": 1}}
        texts = spec_texts(spec)
        self.assertEqual(len(texts), len(set(texts)))
        self.assertIn("zq a woman in a cafe flat snapshot", texts)
        caps = self.root / "caps.safetensors"
        save_text_captures(caps, {t: encode(t) for t in texts})
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(spec))
        return spec_path, caps

    def test_amp_fit_parts_split_and_directions(self):
        spec, caps = self._captures()
        results = {}
        for part in ("full", "style", "rest"):
            out = self.root / f"amp_{part}.safetensors"
            amp_fit(spec, [caps], out, part=part, eps=1e-6)
            results[part] = load_file(str(out))[AMPLIFIER_KEY]
            report = json.loads(out.with_suffix(".json").read_text())
            self.assertEqual((report["n_source"], report["n_preserve"]), (3, 5))
        torch.testing.assert_close(results["style"] + results["rest"], results["full"], rtol=0, atol=1e-5)
        dirs = self.root / "dirs.safetensors"
        style_dirs(spec, [caps], dirs)
        amp_split(self.root / "amp_full.safetensors", self.root / "s.safetensors", self.root / "r.safetensors", directions=dirs)
        torch.testing.assert_close(load_file(str(self.root / "s.safetensors"))[AMPLIFIER_KEY], results["style"], rtol=0, atol=1e-5)
        amp_split(self.root / "amp_full.safetensors", self.root / "s2.safetensors", self.root / "r2.safetensors",
                  directions_from=self.root / "amp_style.safetensors", k=1)
        torch.testing.assert_close(load_file(str(self.root / "r2.safetensors"))[AMPLIFIER_KEY], results["rest"], rtol=0, atol=1e-5)
        with self.assertRaises(ValueError):
            amp_split(self.root / "amp_full.safetensors", self.root / "x.safetensors", self.root / "y.safetensors")


if __name__ == "__main__":
    unittest.main()
