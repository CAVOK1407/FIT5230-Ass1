"""CPU toy tests for v2 selection and gradient contracts; no model files loaded."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "StealthDiffusion"))
import m3_lao_v2 as v2


class Detector(torch.nn.Module):
    def __init__(self, offset=-0.05, scale=20):
        super().__init__()
        self.offset, self.scale = offset, scale

    def forward(self, normalized):
        mean = normalized.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = normalized.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        pixels = normalized * std + mean
        margin = (pixels.mean((1, 2, 3)) - 128 / 255) * self.scale + self.offset
        return torch.stack([torch.zeros_like(margin), margin], 1)


class LPIPS(torch.nn.Module):
    def __init__(self, floor=0):
        super().__init__()
        self.floor = floor

    def forward(self, first, second):
        return (first - second).square().mean() + self.floor


class Renderer:
    def __init__(self, scale=.5):
        self.scale, self.latents = scale, []

    def __call__(self, latent):
        self.latents.append(float(latent.item()))
        return torch.ones((1, 3, 3, 3), device=latent.device) * (128 / 255 + latent * self.scale)


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((3, 3, 3), 128, dtype=np.uint8)
        self.detector = Detector().eval().requires_grad_(False)
        self.lpips = LPIPS().eval().requires_grad_(False)
        self.z0 = torch.zeros(1)

    def run_toy(self, renderer=None, detector=None, lpips=None, maximum=40, prefix=False):
        return v2.optimize_latent(renderer or Renderer(), self.z0, detector or self.detector,
                                  lpips or self.lpips, self.reference, maximum, .01, prefix)

    def test_first_success_stops_adam_then_five_measured_refinements(self):
        renderer = Renderer()
        results = self.run_toy(renderer, prefix=True)
        final = results["final"]
        self.assertGreater(final["actual_steps"], 0)
        self.assertLess(final["actual_steps"], 10)
        self.assertEqual(len(final["refinement_history"]), 5)
        self.assertTrue(final["candidate"]["metrics"]["success"])
        self.assertEqual(final["actual_steps"], len(final["gradient_steps"]))
        self.assertEqual(len(renderer.latents), final["actual_steps"] + 1 + 5 + 1)
        np.testing.assert_array_equal(final["candidate"]["pixels"], results["prefix"]["candidate"]["pixels"])
        self.assertTrue(all(g["grad_l2"] > 0 and g["update_l2"] > 0 for g in final["gradient_steps"]))
        self.assertEqual(renderer.latents[-1], 0.0)  # Actual final reset call.

    def test_ten_step_prefix_never_uses_later_candidates_or_gradients(self):
        detector = Detector(offset=-10).eval()
        results = self.run_toy(Renderer(.02), detector, prefix=True)
        prefix, final = results["prefix"], results["final"]
        self.assertEqual(prefix["actual_steps"], 10)
        self.assertEqual(final["actual_steps"], 40)
        self.assertEqual(prefix["shared_actual_steps"], 40)
        self.assertTrue(all(c["step"] <= 10 for c in prefix["history"]))
        self.assertTrue(all(g["step"] <= 10 for g in prefix["gradient_steps"]))
        self.assertGreater(final["candidate"]["metrics"]["real_margin"], prefix["candidate"]["metrics"]["real_margin"])
        self.assertFalse(final["candidate"]["metrics"]["success"])
        self.assertEqual(final["selection"], "highest_margin_low_distortion_failure")

    def test_zero_step_is_independent_actual_render_plus_actual_reset(self):
        renderer = Renderer()
        result = self.run_toy(renderer, maximum=0)["final"]
        self.assertEqual(renderer.latents, [0.0, 0.0])
        self.assertEqual(result["actual_steps"], 0)
        self.assertEqual(result["refinement_history"], [])
        np.testing.assert_array_equal(result["reset"]["pixels"], result["candidate"]["pixels"])

    def test_real_reset_mismatch_is_detected(self):
        class NonrepeatableRenderer(Renderer):
            def __call__(self, latent):
                value = super().__call__(latent)
                return value if len(self.latents) == 1 else value + 2 / 255
        with self.assertRaisesRegex(RuntimeError, "actual reset renderer differs"):
            self.run_toy(NonrepeatableRenderer(), maximum=0)

    def test_save_projection_is_consumed_by_both_objectives(self):
        class CaptureLPIPS(LPIPS):
            def forward(self, first, second):
                self.second = second.detach().clone()
                return super().forward(first, second)
        lpips = CaptureLPIPS().eval()
        latent = torch.tensor([10.0], requires_grad=True)
        candidate, loss = v2.observe(Renderer(), latent, self.detector, lpips, self.reference, 0)
        self.assertEqual(candidate["metrics"]["max_difference_pixels"], 8)
        expected = v2.image_tensor(candidate["pixels"], "cpu") * 2 - 1
        torch.testing.assert_close(lpips.second, expected, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertNotEqual(float(latent.grad.item()), 0)

    def test_failure_without_quality_candidate_is_explicit_and_retained(self):
        lpips = LPIPS(floor=.04).eval()
        result = self.run_toy(lpips=lpips, maximum=10)["final"]
        self.assertEqual(result["selection"], "minimum_lpips_failure_no_quality_candidate")
        self.assertFalse(result["candidate"]["metrics"]["success"])
        self.assertEqual(result["actual_steps"], 10)
        self.assertEqual(result["candidate"]["metrics"]["step"], 0)

    def test_refinement_keeps_original_success_if_all_probes_fail(self):
        class NarrowRenderer(Renderer):
            def __call__(self, latent):
                self.latents.append(float(latent.item()))
                value = 130 / 255 if float(latent.item()) == 1 else 128 / 255
                return torch.full((1, 3, 3, 3), value)
        renderer = NarrowRenderer()
        original, _ = v2.observe(renderer, torch.ones(1), self.detector, self.lpips, self.reference, 2)
        self.assertTrue(original["metrics"]["success"])
        candidates = [original]
        log = v2.refine(renderer, self.z0, original, self.detector, self.lpips, self.reference, candidates)
        self.assertEqual(len(log), 5)
        self.assertTrue(all(not c["success"] for c in log))
        selected, _ = v2.choose_candidate(candidates)
        self.assertIs(selected, original)

    def test_nonfinite_and_disconnected_renderers_fail_closed(self):
        for bad in (lambda z: torch.full((1, 3, 3, 3), float("nan")),
                    lambda z: torch.full((1, 3, 3, 3), 128 / 255)):
            with self.subTest(bad=bad), self.assertRaises((RuntimeError, FloatingPointError)):
                self.run_toy(bad)

    def test_models_must_be_frozen_and_eval(self):
        with self.assertRaises(ValueError):
            self.run_toy(detector=Detector())
        detector = Detector().eval()
        detector.parameter = torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            self.run_toy(detector=detector)

    def test_budget_guarantees_psnr_floor_and_identical_json_semantics(self):
        identical = v2.quality(self.reference, self.reference, 0)
        self.assertIsNone(identical["psnr"])
        self.assertTrue(identical["psnr_infinite"])
        saturated = v2.quality(self.reference + np.uint8(8), self.reference, .02)
        self.assertGreaterEqual(saturated["psnr"], 30)
        self.assertTrue(saturated["low_distortion"])

    def test_selection_ties_and_quality_take_priority_over_high_margin(self):
        def item(lpips, margin, valid=True, success=False):
            return {"metrics": {"lpips": lpips, "real_margin": margin,
                                "low_distortion": valid, "success": success}}
        first = item(.01, .2, success=True)
        same = item(.01, 2, success=True)
        invalid = item(.04, 10, valid=False)
        self.assertIs(v2.choose_candidate([first, same, invalid])[0], first)
        low = item(.02, -.2)
        self.assertIs(v2.choose_candidate([invalid, low])[0], low)

    def test_prefix_is_opt_in_and_invalid_protocol_values_fail(self):
        args = v2.parser().parse_args(["--images_root", "input", "--out_root", "output"])
        self.assertIsNone(args.prefix_name)
        with self.assertRaises(ValueError):
            self.run_toy(maximum=10, prefix=True)
        self.assertEqual(v2.image_seed(42, "a.png"), v2.image_seed(42, "a.png"))
        self.assertNotEqual(v2.image_seed(42, "a.png"), v2.image_seed(42, "b.png"))


if __name__ == "__main__":
    unittest.main()
