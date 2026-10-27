"""All method tests use tiny synthetic arrays and toy classifiers only."""

import io
import unittest

import numpy as np
import torch

from peer_eval.predecodeguard.methods import generate, quantize_within_budget


class FrozenToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(4.0), requires_grad=False)
        self.eval()

    def forward(self, value):
        return self.weight * value.mean()


class MethodsTests(unittest.TestCase):
    def setUp(self):
        self.model = FrozenToy()
        self.source = np.full((1, 2, 2), 0.2, dtype=np.float16)
        self.seen = []

    def score(self, value):
        self.assertEqual(value.dtype, np.float16)
        self.seen.append(value.copy())
        # Explicit round trip verifies the value used by the callback is exactly
        # the artifact's representation, without touching any real package.
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        buffer.seek(0)
        reloaded = np.load(buffer, allow_pickle=False)
        with torch.no_grad():
            return self.model(torch.from_numpy(reloaded.astype(np.float32))).sigmoid().item()

    def run_method(self, method="lpgd", **overrides):
        settings = dict(method=method, epsilon=0.5, steps=10, alpha=0.05,
                        seed=17, threshold=0.5, score_fn=self.score)
        settings.update(overrides)
        return generate(self.source, self.model, **settings)

    def test_projection_respects_strict_budget_after_serialization(self):
        source = np.array([1.0, -1.0, 65504.0, -65504.0, 0.0], dtype=np.float16)
        candidate = source.astype(np.float64) + np.array([0.0006, -0.0006, 9, -9, 0.0006])
        epsilon = 0.0006
        naive = candidate.astype(np.float16)
        self.assertGreater(abs(float(naive[0]) - float(source[0])), epsilon)
        projected = quantize_within_budget(candidate, source, epsilon)
        buffer = io.BytesIO()
        np.save(buffer, projected, allow_pickle=False)
        buffer.seek(0)
        saved = np.load(buffer, allow_pickle=False)
        self.assertTrue(np.isfinite(saved).all())
        self.assertLessEqual(np.abs(saved.astype(np.float64) - source.astype(np.float64)).max(), epsilon)

    def test_projection_checks_difference_not_rounded_interval_endpoints(self):
        source = np.array([65504.0, -65504.0], dtype=np.float16)
        epsilon = np.nextafter(32.0, 0.0)
        candidate = np.array([65472.0, -65472.0], dtype=np.float64)
        output = quantize_within_budget(candidate, source, epsilon)
        np.testing.assert_array_equal(output, source)

    def test_projection_extremes_and_no_rgb_clamp(self):
        source = np.array([65504.0, -65504.0, 3.0, -2.0], dtype=np.float16)
        candidate = np.array([1e10, -1e10, 3.1, -2.1], dtype=np.float64)
        output = quantize_within_budget(candidate, source, 100.0)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreater(output[2], 1)
        self.assertLess(output[3], 0)

    def test_exact_representable_budget_boundary_is_retained(self):
        source = np.array([1.0, -1.0], dtype=np.float16)
        candidate = np.array([1.125, -1.125], dtype=np.float32)
        output = quantize_within_budget(candidate, source, 0.125)
        np.testing.assert_array_equal(output, candidate.astype(np.float16))

    def test_pgd_monotone_toy_progress_and_frozen_model(self):
        source_copy = self.source.copy()
        output, info = self.run_method()
        self.assertEqual(info["steps"], 10)
        self.assertFalse(info["stopped_early"])
        self.assertLess(info["final_score"], info["initial_score"])
        scores = [row["score"] for row in info["trace"]]
        self.assertTrue(all(a >= b for a, b in zip(scores, scores[1:])))
        self.assertLessEqual(np.abs(output.astype(np.float64) - source_copy.astype(np.float64)).max(), 0.5)
        self.assertIsNone(self.model.weight.grad)
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.weight.requires_grad)
        np.testing.assert_array_equal(self.source, source_copy)
        np.testing.assert_array_equal(output, self.seen[-1])

    def test_lstop_uses_same_path_but_fewer_updates(self):
        fixed, fixed_info = self.run_method()
        early, early_info = self.run_method("lstop")
        self.assertTrue(early_info["stopped_early"])
        self.assertLess(early_info["steps"], fixed_info["steps"])
        self.assertLess(early_info["final_score"], 0.5)
        self.assertEqual(early_info["trace"], fixed_info["trace"][:len(early_info["trace"])])
        self.assertLess(np.linalg.norm(early.astype(float) - self.source.astype(float)),
                        np.linalg.norm(fixed.astype(float) - self.source.astype(float)))

    def test_early_stopping_respects_quantized_reference_callback(self):
        # Deliberately disagree with the optimization model: decisions must be
        # based on the independent scorer, never the optimization logit.
        output, info = self.run_method("lstop", score_fn=lambda array: 0.9)
        self.assertEqual(info["steps"], 10)
        self.assertFalse(info["stopped_early"])
        self.assertEqual(info["final_score"], 0.9)
        self.assertEqual(output.dtype, np.float16)

    def test_score_equal_to_threshold_remains_flagged(self):
        _, info = self.run_method("lstop", score_fn=lambda array: 0.5)
        self.assertEqual(info["steps"], 10)
        self.assertFalse(info["stopped_early"])
        self.assertEqual(info["final_prediction"], 1)

    def test_margin_continues_past_original_threshold_on_toy_model(self):
        _, no_margin = self.run_method("lstop", threshold=0.6)
        output, with_margin = self.run_method(
            "lstop", threshold=0.6, early_stop_margin=0.1
        )
        self.assertLess(no_margin["steps"], with_margin["steps"])
        self.assertLess(no_margin["final_score"], 0.6)
        self.assertGreaterEqual(no_margin["final_score"], 0.5)
        self.assertEqual(no_margin["trace"], with_margin["trace"][:len(no_margin["trace"])])
        self.assertLess(with_margin["final_score"], 0.5)
        self.assertTrue(all(row["score"] >= 0.5 for row in with_margin["trace"][:-1]))
        self.assertEqual(with_margin["early_stop_margin"], 0.1)
        self.assertEqual(with_margin["early_stop_target"], 0.5)
        self.assertEqual(with_margin["threshold"], 0.6)
        self.assertEqual(with_margin["final_prediction"], 0)
        np.testing.assert_array_equal(output, self.seen[-1])

    def test_margin_stops_at_first_strict_quantized_cpu_crossing(self):
        target = 0.5 - 0.1
        scores = iter([0.9, 0.49, target, 0.39])
        scored = []

        def reference_score(array):
            self.assertEqual(array.dtype, np.float16)
            scored.append(array.copy())
            return next(scores)

        output, info = self.run_method(
            "lstop", early_stop_margin=0.1, score_fn=reference_score
        )
        self.assertEqual(info["steps"], 3)
        self.assertTrue(info["stopped_early"])
        self.assertEqual(info["stop_reason"], "below_early_stop_target")
        self.assertEqual([row["score"] for row in info["trace"]], [0.9, 0.49, target, 0.39])
        self.assertEqual(len(scored), 4)
        np.testing.assert_array_equal(output, scored[-1])

    def test_prediction_uses_original_threshold_when_margin_not_reached(self):
        scores = iter([0.9, 0.49])
        _, info = self.run_method(
            "lstop", steps=1, early_stop_margin=0.1,
            score_fn=lambda array: next(scores)
        )
        self.assertEqual(info["initial_prediction"], 1)
        self.assertEqual(info["final_prediction"], 0)
        self.assertGreater(info["final_score"], info["early_stop_target"])
        self.assertFalse(info["stopped_early"])
        self.assertEqual(info["stop_reason"], "completed")

    def test_initially_benign_above_margin_target_remains_noop(self):
        output, info = self.run_method("lstop", threshold=0.9, early_stop_margin=0.3)
        self.assertGreater(info["initial_score"], info["early_stop_target"])
        self.assertLess(info["initial_score"], info["threshold"])
        self.assertEqual(info["steps"], 0)
        self.assertEqual(info["stop_reason"], "initially_below_threshold")
        np.testing.assert_array_equal(output, self.source)

    def test_invalid_margin_rejected_before_any_scoring(self):
        for margin in (-0.1, float("nan"), float("inf"), True, "0.1", 0.5, 0.6):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                self.run_method("lstop", early_stop_margin=margin)
        for method in ("lpgd", "random"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                self.run_method(method, early_stop_margin=0.1)
        self.assertEqual(self.seen, [])

    def test_zero_margin_and_zero_threshold_remain_supported(self):
        _, info = self.run_method("lstop", threshold=0.0, early_stop_margin=0.0)
        self.assertEqual(info["steps"], 10)
        self.assertEqual(info["early_stop_target"], 0.0)
        self.assertEqual(info["final_prediction"], 1)

    def test_zero_budget_is_exact_noop(self):
        for method in ("lpgd", "lstop", "random"):
            output, info = self.run_method(method, epsilon=0.0, alpha=0.0)
            np.testing.assert_array_equal(output, self.source)
            self.assertEqual(info["steps"], 0)
            self.assertEqual(len(info["trace"]), 1)

    def test_initially_below_threshold_is_noop_for_lstop(self):
        output, info = self.run_method("lstop", threshold=0.9)
        np.testing.assert_array_equal(output, self.source)
        self.assertEqual(info["steps"], 0)
        self.assertTrue(info["stopped_early"])

    def test_random_noise_reproducible_and_independent_of_gradient(self):
        def fail_if_called(value):
            raise AssertionError("random control must not query gradients")

        settings = dict(method="random", epsilon=0.1, steps=10, alpha=0.01,
                        seed=11, threshold=0.5, score_fn=self.score)
        first, info = generate(self.source, fail_if_called, **settings)
        second, _ = generate(self.source, fail_if_called, **settings)
        settings["seed"] = 12
        different, _ = generate(self.source, fail_if_called, **settings)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(info["steps"], 1)
        self.assertLessEqual(np.abs(first.astype(float) - self.source.astype(float)).max(), 0.1)

    def test_parameter_and_output_validation(self):
        for setting in ({"steps": 1.5}, {"steps": True}, {"steps": 0},
                        {"epsilon": -1}, {"epsilon": float("nan")},
                        {"alpha": float("inf")}, {"alpha": 0},
                        {"seed": -1}, {"seed": 2.2},
                        {"threshold": 1.1}, {"threshold": float("nan")},
                        {"score_fn": lambda array: float("nan")},
                        {"score_fn": lambda array: [0.5]},
                        {"method": "unknown"}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.run_method(**setting)
        with self.assertRaises(ValueError):
            quantize_within_budget(self.source, self.source.astype(np.float32), 0.1)
        with self.assertRaises(ValueError):
            quantize_within_budget(np.full_like(self.source, np.nan), self.source, 0.1)


if __name__ == "__main__":
    unittest.main()
