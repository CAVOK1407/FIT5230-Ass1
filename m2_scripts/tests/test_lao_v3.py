"""Small CPU-only numerical tests, with no checkpoints or model downloads."""
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "StealthDiffusion"))
import m3_lao_v3 as v3


class Detector(torch.nn.Module):
    def __init__(self, offset=-.05, scale=20):
        super().__init__()
        self.offset, self.scale = offset, scale

    def forward(self, normalized):
        mean = normalized.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = normalized.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        image = normalized * std + mean
        margin = (image.mean((1, 2, 3)) - 128 / 255) * self.scale + self.offset
        return torch.stack([torch.zeros_like(margin), margin], 1)


class LPIPS(torch.nn.Module):
    def __init__(self, floor=0):
        super().__init__()
        self.floor = floor

    def forward(self, first, second):
        self.second = second.detach().clone()
        return (first - second).square().mean() + self.floor


class Renderer:
    def __init__(self):
        self.calls = []

    def __call__(self, z):
        self.calls.append(float(z.item()))
        return torch.ones((1, 3, 8, 8), device=z.device) * (128 / 255 + z * .5)


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((8, 8, 3), 128, dtype=np.uint8)
        self.detector = Detector().eval().requires_grad_(False)
        self.lpips = LPIPS().eval().requires_grad_(False)

    def objective(self, eot=False, detector=None, lpips=None):
        return v3.Objective(detector or self.detector, lpips or self.lpips, self.reference, "cpu", eot)

    def test_jpeg_actual_forward_equals_evaluator_spec_and_identity_backward(self):
        pixels = np.random.default_rng(42).integers(0, 256, (17, 19, 3), dtype=np.uint8)
        for q in (80, 90):
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="JPEG", quality=q)
            stream.seek(0)
            expected = np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)
            np.testing.assert_array_equal(v3.jpeg_pixels(pixels, q), expected)
            saved = v3.image_tensor(pixels, "cpu").requires_grad_(True)
            output = v3.jpeg_bpda(saved, pixels, q)
            torch.testing.assert_close(output, v3.image_tensor(expected, "cpu"), rtol=0, atol=0)
            self.assertFalse(torch.equal(output, saved))
            weights = torch.arange(output.numel()).reshape_as(output).float()
            (output * weights).sum().backward()
            torch.testing.assert_close(saved.grad, weights, rtol=0, atol=0)

    def test_subpixel_updates_accumulate_in_continuous_buffer(self):
        current = v3.image_tensor(self.reference, "cpu")
        lower, upper = v3.continuous_bounds(self.reference, "cpu")
        gradient = -torch.ones_like(current)
        states, pixels = [], []
        for _ in range(5):
            current = v3.pixel_update(current, gradient, lower, upper, .25)
            states.append(current.clone())
            pixels.append(v3.projected_uint8(current, self.reference, 8))
        self.assertGreater(float(states[0].mean()), 128 / 255)
        self.assertEqual(int(pixels[0][0, 0, 0]), 128)
        self.assertEqual(int(pixels[-1][0, 0, 0]), 129)
        self.assertGreater(float(states[-1].mean() - states[0].mean()), .9 / 255)

    def test_objective_consumes_exact_projected_png_and_requires_all_views(self):
        class ViewDetector(Detector):
            def __init__(self):
                super().__init__()
                self.counter = 0
            def forward(self, x):
                output = super().forward(x)
                self.counter += 1
                # Identity and JPEG90 pass; JPEG80 fails, despite positive mean.
                offset = (1., -.2, 1.)[(self.counter - 1) % 3]
                return output * 0 + output.new_tensor([[0., offset]])
        objective = self.objective(True, ViewDetector().eval())
        raw = v3.image_tensor(self.reference, "cpu").requires_grad_(True)
        item, loss = objective.observe(raw, raw, 0, "pixel")
        self.assertFalse(item["metrics"]["success"])
        self.assertAlmostEqual(item["metrics"]["worst_view_margin"], -.2)
        self.assertEqual(objective.compute["E_forwards"], 3)
        self.assertEqual(objective.compute["JPEG_encodes"], 2)
        self.assertEqual(objective.compute["LPIPS_forwards"], 1)
        torch.testing.assert_close(self.lpips.second, v3.image_tensor(item["pixels"], "cpu") * 2 - 1, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(loss))

    def test_hybrid_uses_one_original_budget_and_retains_lao_candidate(self):
        objective = self.objective(detector=Detector(offset=-10).eval())
        initial = self.reference + np.uint8(7)
        result = v3.optimize_pixels(objective, initial, max_steps=12, restarts=3, alpha_pixels=1)
        self.assertEqual(result["actual_gradient_steps"], 36)
        self.assertEqual(len(result["stop_reasons"]), 3)
        self.assertTrue(all(h["max_difference_pixels"] <= 8 for h in result["history"]))
        retained = [h for h in result["history"] if h["phase"] == "stage_initial_png"]
        self.assertEqual(retained[0]["pixels_sha256"], v3.pixels_sha256(initial))
        self.assertEqual(result["candidate"]["metrics"]["max_difference_pixels"], 8)
        self.assertTrue(all(g["grad_l2"] > 0 for g in result["gradient_steps"]))

    def test_pgd_early_success_each_restart_has_five_actual_probes(self):
        result = v3.optimize_pixels(self.objective(True), max_steps=40, restarts=3, alpha_pixels=.5, source="a")
        self.assertTrue(result["candidate"]["metrics"]["success"])
        self.assertEqual(len(result["refinement_history"]), 15)
        self.assertLess(result["actual_gradient_steps"], 120)
        self.assertEqual(result["compute"]["observations"], len(result["history"]))
        self.assertEqual(result["compute"]["E_forwards"], 3 * len(result["history"]))
        self.assertEqual(result["compute"]["LPIPS_forwards"], len(result["history"]))
        for restart in range(3):
            self.assertEqual(len([x for x in result["refinement_history"] if x["restart"] == restart]), 5)

    def test_pixel_rng_is_stable_and_source_specific(self):
        a = v3.optimize_pixels(self.objective(), max_steps=2, source="a")
        b = v3.optimize_pixels(self.objective(), max_steps=2, source="a")
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(v3.seed_for(42, "a", 1), v3.seed_for(42, "a", 1))
        self.assertNotEqual(v3.seed_for(42, "a", 1), v3.seed_for(42, "b", 1))

    def test_latent_gradient_five_refinements_and_actual_reset(self):
        renderer = Renderer()
        result = v3.optimize_latent(renderer, torch.zeros(1), self.objective(True), max_steps=40)
        self.assertTrue(result["candidate"]["metrics"]["success"])
        self.assertEqual(len(result["refinement_history"]), 5)
        self.assertTrue(all(g["grad_l2"] > 0 and g["update_l2"] > 0 for g in result["gradient_steps"]))
        self.assertEqual(renderer.calls[-1], 0.)
        np.testing.assert_array_equal(result["reset"]["pixels"], self.reference)
        self.assertEqual(len(renderer.calls), result["actual_gradient_steps"] + 1 + 5 + 1)
        self.assertEqual(result["compute"]["observations"], len(result["history"]) + 1)

    def test_independent_zero_and_reset_execute_same_renderer(self):
        zero_renderer, active_renderer = Renderer(), Renderer()
        zero = v3.optimize_latent(zero_renderer, torch.zeros(1), self.objective(), max_steps=0)
        active = v3.optimize_latent(active_renderer, torch.zeros(1), self.objective(), max_steps=2)
        self.assertEqual(zero_renderer.calls, [0., 0.])
        self.assertEqual(zero["actual_gradient_steps"], 0)
        np.testing.assert_array_equal(zero["candidate"]["pixels"], active["reset"]["pixels"])

    def test_nonrepeatable_reset_fails(self):
        class BadRenderer(Renderer):
            def __call__(self, z):
                result = super().__call__(z)
                return result if len(self.calls) == 1 else result + 2 / 255
        with self.assertRaisesRegex(RuntimeError, "actual latent reset differs"):
            v3.optimize_latent(BadRenderer(), torch.zeros(1), self.objective(), max_steps=0)

    def test_retained_success_survives_failed_refinement_probes(self):
        pool = v3.CandidatePool()
        objective = self.objective()
        endpoint = torch.ones(1)
        def renderer(z):
            return torch.full((1, 3, 8, 8), (131 if float(z.item()) == 1 else 128) / 255)
        first, _ = objective.observe(renderer(endpoint), endpoint, 2, "latent")
        self.assertTrue(first["metrics"]["success"])
        pool.add(first)
        logs = v3.refine_segment(objective, torch.zeros(1), endpoint, renderer, first, pool, "latent", 0)
        self.assertEqual(len(logs), 5)
        self.assertTrue(all(not item["success"] for item in logs))
        self.assertIs(pool.best, first)

    def test_selection_uses_worst_view_margin_and_preserves_stable_ties(self):
        def item(lp, worst, success=False, valid=True):
            return {"metrics": {"lpips": lp, "worst_view_margin": worst,
                                "success": success, "low_distortion": valid}}
        first, same = item(.01, .2, True), item(.01, 2, True)
        self.assertIs(v3.choose_candidate([first, same])[0], first)
        a, b = item(.02, -.5), item(.03, -.1)
        self.assertIs(v3.choose_candidate([a, b])[0], b)
        invalid = item(.04, 10, valid=False)
        self.assertIs(v3.choose_candidate([invalid, a])[0], a)

    def test_explicit_failure_when_no_quality_candidate(self):
        result = v3.optimize_latent(Renderer(), torch.zeros(1), self.objective(lpips=LPIPS(.04).eval()), max_steps=2)
        self.assertFalse(result["candidate"]["metrics"]["success"])
        self.assertEqual(result["selection"], "minimum_lpips_failure_no_quality_candidate")
        self.assertEqual(result["actual_gradient_steps"], 2)

    def test_disconnected_nonfinite_and_unfrozen_fail_closed(self):
        for renderer in (lambda z: torch.full((1, 3, 8, 8), float("nan")),
                         lambda z: torch.full((1, 3, 8, 8), 128 / 255)):
            with self.assertRaises((RuntimeError, FloatingPointError)):
                v3.optimize_latent(renderer, torch.zeros(1), self.objective(), max_steps=2)
        with self.assertRaises(ValueError):
            self.objective(detector=Detector())
        with self.assertRaises(FloatingPointError):
            v3.optimize_pixels(self.objective(lpips=LPIPS(float("nan")).eval()), max_steps=1)

    def test_save_reload_verifies_all_views_and_lpips(self):
        result = v3.optimize_pixels(self.objective(True), max_steps=3, restarts=1)
        chosen = result["candidate"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.png"
            info = v3.save_png_verified(path, chosen["pixels"], self.reference, 8)
            actual = v3.verify_saved(self.objective(True), path, chosen)
            self.assertEqual(actual["view_margins"], chosen["metrics"]["view_margins"])
            self.assertEqual(info["max_difference_pixels"], chosen["metrics"]["max_difference_pixels"])
            self.assertLessEqual(info["max_difference_pixels"], 8)

    def test_config_rejects_unknown_fields_unsafe_names_and_invalid_budgets(self):
        valid = v3.normalize_configs([{"name": "P", "kind": "P", "alpha_pixels": .25}])
        self.assertEqual(valid[0]["restarts"], 3)
        for item in ({"name": "../bad", "kind": "P"}, {"name": "P", "kind": "Q"},
                     {"name": "P", "kind": "P", "alpha_pixels": 0},
                     {"name": "P", "kind": "P", "max_steps": 81},
                     {"name": "P", "kind": "P", "unknown": 1},
                     {"name": "P", "kind": "P", "eot": 1}):
            with self.assertRaises(ValueError):
                v3.normalize_configs([item])


if __name__ == "__main__":
    unittest.main()
