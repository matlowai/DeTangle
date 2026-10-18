import unittest
from unittest.mock import patch

import torch

from delta_surgery_seed.core import map_layers, response_metrics, solve


class SolverTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.a = torch.randn(3, 12, generator=g, dtype=torch.float64)
        self.xc = torch.randn(5, 12, generator=g, dtype=torch.float64)
        self.xt = torch.randn(4, 12, generator=g, dtype=torch.float64)

    def test_primal_dual_and_normal_equation(self):
        for center in ["zero", "source"]:
            with self.subTest(center=center):
                primal, info = solve(self.a, self.xc, self.xt, center=center, method="primal")
                dual, _ = solve(self.a, self.xc, self.xt, center=center, method="dual")
                torch.testing.assert_close(primal, dual, rtol=1e-9, atol=1e-9)
                a0 = torch.zeros_like(self.a) if center == "zero" else self.a
                gradient = (primal @ self.xc.T @ self.xc
                            + .3 * (primal - self.a) @ self.xt.T @ self.xt
                            + info["epsilon"] * (primal - a0))
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-10)

    def test_unseen_direction_and_conditional_response(self):
        a = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        xc, xt = torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1., 0.]])
        zero, _ = solve(a, xc, xt)
        source, _ = solve(a, xc, xt, center="source")
        self.assertAlmostEqual(float(zero[0, 0]), 0.)
        self.assertAlmostEqual(float(zero[0, 2]), 0.)
        self.assertAlmostEqual(float(source[0, 2]), 3.)
        self.assertGreater(float(zero[0, 1]), 1.99)

    def test_input_immutability_and_relative_ridge_scaling(self):
        old = [v.clone() for v in [self.a, self.xc, self.xt]]
        first, _ = solve(self.a, self.xc, self.xt)
        repeated, _ = solve(self.a, self.xc.repeat(2, 1), self.xt.repeat(2, 1))
        torch.testing.assert_close(first, repeated, rtol=1e-9, atol=1e-9)
        for a, b in zip(old, [self.a, self.xc, self.xt]):
            self.assertTrue(torch.equal(a, b))

    def test_invalid_inputs(self):
        for kwargs in [{"mu": 0}, {"mu": float("nan")}, {"ridge": -1}, {"ridge": float("inf")}, {"center": "unknown"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solve(self.a, self.xc, self.xt, **kwargs)
        for xc in [torch.zeros(0, 12), torch.ones(2, 11), torch.ones(2, 12, dtype=torch.int32),
                   torch.full((2, 12), float("nan")), torch.ones(2, 12, dtype=torch.complex64)]:
            with self.subTest(shape=xc.shape), self.assertRaises(ValueError):
                solve(self.a, xc, self.xt)
        with self.assertRaises(ValueError):
            solve(self.a, torch.zeros_like(self.xc), torch.zeros_like(self.xt))

    def test_response_matches_dense_ba(self):
        b = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 10
        edited, _ = solve(self.a, self.xc, self.xt)
        metrics = response_metrics(self.a, b, edited, self.xt)
        before, after = b @ self.a @ self.xt.T, b @ edited @ self.xt.T
        self.assertAlmostEqual(metrics["retained"], float(after.norm() / before.norm()), places=10)
        self.assertAlmostEqual(metrics["relative_change"], float((after - before).norm() / before.norm()), places=10)
        self.assertAlmostEqual(metrics["cosine"], float((after * before).sum() / (after.norm() * before.norm())), places=10)
        self.assertEqual(response_metrics(self.a * 0, b, edited, self.xt)["retained"], None)

    def test_historical_rejects_changed_prior_and_missing_cuda(self):
        with self.assertRaises(ValueError):
            solve(self.a, self.xc, self.xt, center="source", method="historical")
        with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            solve(self.a, self.xc, self.xt, method="historical")

    @unittest.skipUnless(torch.cuda.is_available(), "Historical CUDA arithmetic needs a GPU")
    def test_historical_cuda_satisfies_objective_and_restores_tf32(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            actual, info = solve(self.a, self.xc, self.xt, ridge=.1, method="historical")
            stable, _ = solve(self.a, self.xc, self.xt, ridge=.1)
            torch.testing.assert_close(actual, stable, rtol=2e-5, atol=2e-5)
            self.assertEqual(info["arithmetic_device"], "cuda")
            self.assertEqual(info["arithmetic_dtype"], "float32")
            self.assertTrue(torch.equal(actual, actual.float().double()))
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def test_mapping_rejects_missing_and_ambiguous_names(self):
        key = "transformer.blocks.0.q.lora_A.weight"
        self.assertEqual(map_layers([key], ["diffusion_model.blocks.0.q"]),
                         {key: "diffusion_model.blocks.0.q"})
        for names in [["blocks.1.q"], ["blocks.0.q", "transformer.blocks.0.q"]]:
            with self.assertRaises(ValueError):
                map_layers([key], names)
        with self.assertRaises(ValueError):
            map_layers([key, "blocks.0.q.lora_A.weight"], ["blocks.0.q"])


if __name__ == "__main__":
    unittest.main()
