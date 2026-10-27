"""Strict budget/export contracts and toy PGD; no real checkpoints or images."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "StealthDiffusion"))
import m3_lao_budget as budget
import m3_lao_pgd as pgd


class BudgetProjectionTests(unittest.TestCase):
    def test_signed_projection_handles_both_uint8_extremes(self):
        reference = np.array([[[0, 255, 128], [255, 0, 127]]], dtype=np.uint8)
        candidate = 255 - reference
        output = budget.project_uint8(candidate, reference, 8)
        expected = np.array([[[8, 247, 127], [247, 8, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(budget.max_difference_pixels(output, reference), 8)

    def test_projection_zero_full_budget_and_idempotence(self):
        rng = np.random.default_rng(7)
        reference = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)
        candidate = rng.integers(0, 256, reference.shape, dtype=np.uint8)
        np.testing.assert_array_equal(budget.project_uint8(candidate, reference, 0), reference)
        np.testing.assert_array_equal(budget.project_uint8(candidate, reference, 255), candidate)
        for limit in (1, 8, 30, 127):
            result = budget.project_uint8(candidate, reference, limit)
            self.assertLessEqual(budget.max_difference_pixels(result, reference), limit)
            np.testing.assert_array_equal(budget.project_uint8(result, reference, limit), result)

    def test_invalid_budget_shape_and_dtype_are_rejected(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for limit in (-1, 256, 8.0, True, float("nan")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                budget.project_uint8(image, image, limit)
        with self.assertRaises(TypeError):
            budget.project_uint8(image.astype(np.float32), image, 8)
        with self.assertRaises(ValueError):
            budget.project_uint8(image[:1], image, 8)

    def test_stats_distinguish_pixel_and_channel_clipping(self):
        reference = np.full((1, 2, 3), 128, dtype=np.uint8)
        candidate = reference.copy()
        candidate[0, 0, 0] = 255
        output = budget.project_uint8(candidate, reference, 8)
        stats = budget.projection_stats(candidate, reference, output, 8)
        self.assertEqual(stats["pre_max_difference_pixels"], 127)
        self.assertEqual(stats["post_max_difference_pixels"], 8)
        self.assertEqual(stats["clipped_pixel_fraction"], 0.5)
        self.assertEqual(stats["clipped_channel_fraction"], 1 / 6)

    def test_png_roundtrip_is_exact_and_overwrite_or_violation_fails(self):
        reference = np.full((4, 4, 3), 250, dtype=np.uint8)
        output = budget.project_uint8(np.zeros_like(reference), reference, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.png"
            info = budget.save_png_verified(path, output, reference, 8)
            np.testing.assert_array_equal(budget.load_rgb(path), output)
            self.assertEqual(info["max_difference_pixels"], 8)
            self.assertEqual(info["sha256"], budget.sha256(path))
            with self.assertRaises(FileExistsError):
                budget.save_png_verified(path, output, reference, 8)
            with self.assertRaises(ValueError):
                budget.save_png_verified(Path(tmp) / "invalid.png", np.zeros_like(reference), reference, 8)


class ExportTests(unittest.TestCase):
    def make_source(self, tmp):
        root = Path(tmp)
        inputs, raw, baseline = root / "inputs", root / "raw", root / "baseline"
        for folder in (inputs, raw, baseline):
            folder.mkdir()
        config = {"res": 4, "n_images": 1, "iterations": 10, "images_root": str(inputs),
                  "encode_source": "clean", "control_source": "clean", "decode_path": "final",
                  "control_scale": 0.5, "lao_residual_anchor": True}
        (raw / "run_config.json").write_text(json.dumps(config))
        (baseline / "run_config.json").write_text(json.dumps({**config, "iterations": 0}))
        record = {"image_id": "0000", "source": "input.png", "seconds": 1.25}
        (raw / "records.jsonl").write_text(json.dumps(record) + "\n")
        (baseline / "records.jsonl").write_text(json.dumps({**record, "seconds": 0.5}) + "\n")
        reference = np.full((4, 4, 3), 128, dtype=np.uint8)
        reset = np.full_like(reference, 140)
        budget.save_png_verified(inputs / "input.png", reference)
        budget.save_png_verified(raw / "0000_originImage.png", reference)
        budget.save_png_verified(baseline / "0000_originImage.png", reference)
        budget.save_png_verified(raw / "0000_adv_image.png", np.full_like(reference, 255))
        budget.save_png_verified(raw / "0000_reset_image.png", reset)
        budget.save_png_verified(baseline / "0000_adv_image.png", reset)
        return inputs, raw, baseline, reference

    def test_export_checks_independent_reset_and_preserves_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs, raw, baseline, reference = self.make_source(tmp)
            original_hash = budget.sha256(raw / "0000_adv_image.png")
            out = Path(tmp) / "derived"
            with contextlib.redirect_stdout(io.StringIO()):
                budget.export_budget(raw, baseline, inputs, out, 8)
            self.assertEqual({p.name for p in out.iterdir() if p.is_dir()}, {"LAO8", "RECON8"})
            self.assertEqual(budget.sha256(raw / "0000_adv_image.png"), original_hash)
            self.assertEqual(budget.sha256(out / "LAO8/0000_reset_image.png"),
                             budget.sha256(out / "RECON8/0000_adv_image.png"))
            np.testing.assert_array_equal(budget.load_rgb(out / "LAO8/0000_originImage.png"), reference)
            for name, seconds in (("LAO8", 1.25), ("RECON8", 0.5)):
                record = json.loads((out / name / "records.jsonl").read_text())
                self.assertEqual(record["max_difference_pixels"], 8)
                self.assertEqual(record["seconds"], seconds)
                self.assertIsNone(record["pgd"])
            with self.assertRaises(FileExistsError):
                budget.export_budget(raw, baseline, inputs, out, 8)

    def test_independent_baseline_mismatch_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs, raw, baseline, reference = self.make_source(tmp)
            (baseline / "0000_adv_image.png").unlink()
            budget.save_png_verified(baseline / "0000_adv_image.png", reference)
            out = Path(tmp) / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "independently run baseline"):
                budget.export_budget(raw, baseline, inputs, out, 8)
            self.assertFalse(out.exists())

    def test_wrong_source_pixels_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs, raw, baseline, reference = self.make_source(tmp)
            (inputs / "input.png").unlink()
            budget.save_png_verified(inputs / "input.png", np.zeros_like(reference))
            out = Path(tmp) / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "canonical source differs"):
                budget.export_budget(raw, baseline, inputs, out, 8)
            self.assertFalse(out.exists())

    def test_existing_experiment_root_is_allowed_without_touching_other_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs, raw, baseline, _ = self.make_source(tmp)
            out = Path(tmp) / "experiment"
            out.mkdir()
            preserved = out / "plan.json"
            preserved.write_text('{"frozen": true}\n')
            before = preserved.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                budget.export_budget(raw, baseline, inputs, out, 8)
            self.assertEqual(preserved.read_bytes(), before)
            self.assertTrue((out / "LAO8").is_dir())
            self.assertTrue((out / "RECON8").is_dir())


class ToyDetector(torch.nn.Module):
    def forward(self, normalized):
        score = normalized.mean(dim=(1, 2, 3)) * 3 - 1
        return torch.stack([torch.zeros_like(score), score], dim=1)


class ToyLPIPS(torch.nn.Module):
    def forward(self, first, second):
        return (first - second).square().mean(dim=(1, 2, 3), keepdim=True)


class PGDContractTests(unittest.TestCase):
    def setUp(self):
        self.detector = ToyDetector().eval().requires_grad_(False)
        self.lpips = ToyLPIPS().eval().requires_grad_(False)
        self.reference = np.full((3, 3, 3), 128, dtype=np.uint8)

    def test_projected_forward_matches_png_bytes_and_has_identity_gradient(self):
        image = torch.full((1, 3, 3, 3), 0.99, requires_grad=True)
        saved = pgd.projected_png_ste(image, self.reference, 8)
        expected = budget.project_uint8(pgd.candidate_uint8(image), self.reference, 8)
        torch.testing.assert_close(saved, pgd.image_tensor(expected, "cpu"), rtol=0, atol=0)
        saved.sum().backward()
        torch.testing.assert_close(image.grad, torch.ones_like(image), rtol=0, atol=0)

    def test_pgd10_is_final_iterate_and_obeys_strict_budget(self):
        output, info = pgd.pgd10(self.detector, self.lpips, self.reference)
        self.assertEqual(info["actual_gradient_steps"], 10)
        self.assertEqual(info["selected_step"], 10)
        self.assertEqual(info["selected_restart"], 0)
        self.assertEqual(len(info["history"]), 10)
        self.assertLessEqual(budget.max_difference_pixels(output, self.reference), 8)
        self.assertEqual(info["final"], pgd.score_pixels(self.detector, output, "cpu"))

    def test_pgd40_is_strong_deterministic_reference_with_120_updates(self):
        output, info = pgd.pgd40(self.detector, self.reference, "fixture-input-sha")
        repeat, again = pgd.pgd40(self.detector, self.reference, "fixture-input-sha")
        self.assertEqual(info["actual_gradient_steps"], 120)
        self.assertEqual(len(info["history"]), 123)
        self.assertGreaterEqual(info["final"]["real_margin"], info["clean"]["real_margin"])
        self.assertLessEqual(budget.max_difference_pixels(output, self.reference), 8)
        np.testing.assert_array_equal(output, repeat)
        self.assertEqual(info["restart_seeds"], again["restart_seeds"])
        self.assertNotEqual(pgd.stable_restart_seed(20261018, "a", 1), pgd.stable_restart_seed(20261018, "b", 1))

    def test_zero_budget_remains_identical_despite_updates(self):
        output, _ = pgd.pgd10(self.detector, self.lpips, self.reference, budget_pixels=0)
        np.testing.assert_array_equal(output, self.reference)

    def test_nonfinite_image_or_detector_output_fails_closed(self):
        with self.assertRaises(FloatingPointError):
            pgd.projected_png_ste(torch.full((1, 3, 3, 3), float("nan")), self.reference, 8)
        class BadDetector(ToyDetector):
            def forward(self, normalized):
                return super().forward(normalized) * float("nan")
        with self.assertRaises(FloatingPointError):
            pgd.pgd10(BadDetector().eval(), self.lpips, self.reference)


if __name__ == "__main__":
    unittest.main()
