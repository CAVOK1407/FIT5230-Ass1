"""LAO research helpers: no checkpoint loading or real-image attack execution."""
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch


SUPPORT_PATH = Path(__file__).resolve().parents[2] / "StealthDiffusion" / "m3_lao_support.py"
spec = importlib.util.spec_from_file_location("m3_lao_support", SUPPORT_PATH)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class SavedImageQuantizationTests(unittest.TestCase):
    def test_forward_matches_actual_png_reload_including_bin_boundaries(self):
        edges = np.arange(256, dtype=np.float32) / np.float32(255)
        values = np.stack([np.nextafter(edges, -np.inf), edges,
                           np.nextafter(edges, np.inf)], axis=0)
        batch = torch.from_numpy(np.stack([values, values[:, ::-1].copy()], axis=0))[:, :, None, :]
        actual = support.saved_image_ste(batch).cpu().numpy()
        for index in range(batch.shape[0]):
            # This is the production to_uint8 expression, then a real PNG roundtrip.
            pixels = np.uint8(batch[index].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="PNG")
            stream.seek(0)
            reloaded = np.asarray(Image.open(stream)).astype(np.float32) / 255.0
            np.testing.assert_array_equal(actual[index].transpose(1, 2, 0), reloaded)

    def test_backward_passes_nonzero_input_gradient(self):
        image = torch.tensor([[[[0.123, 0.501, 0.999]]]], requires_grad=True)
        weights = torch.tensor([[[[2.0, -3.0, 0.25]]]])
        (support.saved_image_ste(image) * weights).sum().backward()
        torch.testing.assert_close(image.grad, weights, rtol=0, atol=0)

    def test_clipping_and_truncation_are_not_rounding(self):
        image = torch.tensor([[[[-0.1, 1.0, 1.1, 0.5]]]])
        expected = torch.tensor([[[[0.0, 1.0, 1.0, 127 / 255]]]])
        torch.testing.assert_close(support.quantize_like_save(image), expected, rtol=0, atol=0)

    def test_nonfinite_image_fails_before_uint8_cast(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                support.saved_image_ste(torch.tensor([value]))


class ControlScaleTests(unittest.TestCase):
    def test_default_preserves_feature_objects(self):
        features = [torch.randn(1, 3, 2, 2), torch.randn(1, 1, 1, 1)]
        self.assertIs(support.scale_control_features(features, 1.0), features)

    def test_zero_and_partial_scale_do_not_mutate_features(self):
        original = torch.tensor([1.0, -2.0], requires_grad=True)
        zero = support.scale_control_features([original], 0.0)[0]
        torch.testing.assert_close(zero, torch.zeros_like(original))
        scaled = support.scale_control_features([original], 0.5)[0]
        scaled.sum().backward()
        torch.testing.assert_close(original, torch.tensor([1.0, -2.0]))
        torch.testing.assert_close(original.grad, torch.tensor([0.5, 0.5]))

    def test_invalid_scale_rejected(self):
        for value in (-0.001, 1.001, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                support.scale_control_features([], value)


class LatentAttributionTests(unittest.TestCase):
    def test_zero_decay_does_not_move_latent_when_objective_gradient_is_zero(self):
        latent = torch.tensor([1.0, -2.0], requires_grad=True)
        original = latent.detach().clone()
        optimizer = support.make_latent_optimizer(latent, lr=0.01, weight_decay=0)
        for _ in range(3):
            optimizer.zero_grad()
            (latent * 0).sum().backward()
            optimizer.step()
        torch.testing.assert_close(latent, original, rtol=0, atol=0)

    def test_legacy_default_optimizer_is_numerically_unchanged(self):
        first = torch.tensor([1.0, -2.0], requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        actual = support.make_latent_optimizer(first, lr=0.01)
        legacy = torch.optim.AdamW([second], lr=0.01)
        for _ in range(3):
            for latent, optimizer in ((first, actual), (second, legacy)):
                optimizer.zero_grad()
                (latent.square().sum()).backward()
                optimizer.step()
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_invalid_decay_rejected(self):
        for value in (-0.01, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                support.make_latent_optimizer(torch.tensor([1.0], requires_grad=True), 0.01, value)

    def test_missing_or_nonfinite_gradient_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "gradient.*missing"):
            support.require_finite(None, "latent gradient")
        with self.assertRaises(FloatingPointError):
            support.tensor_norms(torch.tensor([float("nan")]), "latent gradient")

    def test_pixel_change_fraction_counts_spatial_pixels_not_rgb_channels(self):
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = first.copy()
        second[0, 0, 0] = 1
        second[1, 1, :] = 2
        self.assertEqual(support.changed_pixel_fraction(first, second), 0.5)
        with self.assertRaises(ValueError):
            support.changed_pixel_fraction(first, second[:, :1])


class ResidualAnchorTests(unittest.TestCase):
    def test_exact_zero_residual_restores_full_reference_without_clipping(self):
        full = torch.tensor([[[[-1.25, 0.127, 1.4]]]])
        gated = torch.tensor([[[[-2.0, 0.5, 0.75]]]])
        actual = support.residual_anchor(gated.clone(), full, gated)
        torch.testing.assert_close(actual, full, rtol=0, atol=0)

    def test_current_decoder_gradient_is_preserved_and_references_are_detached(self):
        latent = torch.tensor([0.1, -0.3], requires_grad=True)
        full = torch.tensor([0.4, 0.6], requires_grad=True)
        reference = torch.tensor([0.1, -0.1], requires_grad=True)
        weights = torch.tensor([2.0, -3.0])
        actual = support.residual_anchor(latent * 0.5, full, reference)
        (actual * weights).sum().backward()
        torch.testing.assert_close(latent.grad, weights * 0.5, rtol=0, atol=0)
        self.assertIsNone(full.grad)
        self.assertIsNone(reference.grad)

    def test_residual_precedes_image_clamp_and_saved_image_ste(self):
        current = torch.tensor([[[[0.8, -0.4]]]], requires_grad=True)
        reference = torch.tensor([[[[0.0, 0.0]]]])
        full = torch.tensor([[[[0.0, 0.0]]]])
        raw = support.residual_anchor(current, full, reference)
        saved = support.saved_image_ste((raw / 2 + 0.5).clamp(0, 1))
        saved.sum().backward()
        torch.testing.assert_close(current.grad, torch.full_like(current, 0.5), rtol=0, atol=0)

    def test_nonfinite_input_and_overflow_are_rejected(self):
        good = torch.tensor([1.0])
        bad = torch.tensor([float("nan")])
        for inputs in ((bad, good, good), (good, bad, good), (good, good, bad)):
            with self.subTest(inputs=inputs), self.assertRaises(FloatingPointError):
                support.residual_anchor(*inputs)
        high = torch.tensor([torch.finfo(torch.float32).max])
        with self.assertRaises(FloatingPointError):
            support.residual_anchor(high, high, -high)

    def test_broadcasting_and_mixed_precision_are_rejected(self):
        current = torch.zeros(1, 3, 2, 2)
        with self.assertRaises(ValueError):
            support.residual_anchor(current, torch.zeros(1, 3, 1, 1), current)
        with self.assertRaises(ValueError):
            support.residual_anchor(current, current.double(), current)

    def test_only_clean_final_path_is_allowed_when_enabled(self):
        support.validate_residual_anchor(True, "clean", "clean", "final")
        support.validate_residual_anchor(False, "pgd1", "pgd", "plain")
        for config in (("pgd1", "clean", "final"), ("clean", "pgd", "final"),
                       ("clean", "encoded", "final"), ("clean", "clean", "plain")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                support.validate_residual_anchor(True, *config)


if __name__ == "__main__":
    unittest.main()
