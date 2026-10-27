"""Contract tests for the matched selection comparator, without real models."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m3_lao_pgd import pgd10
from m3_lao_pgd_v2 import candidate, choose_fallback, refinement, run_pair


class ToyDetector(torch.nn.Module):
    def forward(self, normalized):
        mean = normalized.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = normalized.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        raw = normalized * std + mean
        margin = (raw.mean((1, 2, 3)) - .48) * 200
        return torch.stack((-margin / 2, margin / 2), dim=1)


class ToyLPIPS(torch.nn.Module):
    def forward(self, a, b):
        return (a - b).square().mean((1, 2, 3), keepdim=True)


class MatchedPGDTest(unittest.TestCase):
    def setUp(self):
        self.detector = ToyDetector().eval()
        self.lpips = ToyLPIPS().eval()
        self.reference = np.full((8, 8, 3), 120, dtype=np.uint8)

    def test_original_final_iterate_is_preserved(self):
        original, _ = pgd10(self.detector, self.lpips, self.reference)
        output = run_pair(self.detector, self.lpips, self.reference)
        np.testing.assert_array_equal(original, output['PGD10'][0])

    def test_matched_success_uses_prefix_and_keeps_low_distortion(self):
        output = run_pair(self.detector, self.lpips, self.reference)
        matched = output['PGD10_MATCHED'][1]
        self.assertTrue(matched['selected']['buffered_success'])
        self.assertLessEqual(matched['selected']['lpips'], output['PGD10'][1]['selected']['lpips'])
        self.assertEqual(len(matched['refinement']), 5)
        self.assertLess(matched['actual_gradient_steps'], 10)
        self.assertLessEqual(matched['selected']['step'], matched['selection_reads_only_prefix_through'])

    def test_integer_budget_and_determinism(self):
        first = run_pair(self.detector, self.lpips, self.reference)
        second = run_pair(self.detector, self.lpips, self.reference)
        for name in first:
            pixels = first[name][0]
            np.testing.assert_array_equal(pixels, second[name][0])
            self.assertLessEqual(int(np.abs(pixels.astype(np.int16) - self.reference).max()), 8)

    def test_no_success_does_not_select_an_invalid_quality_candidate(self):
        a = candidate(self.reference, self.reference, -10, .001, 0)
        b = candidate(self.reference, self.reference, -1, .04, 1)
        self.assertIs(choose_fallback([a, b]), a)

    def test_ties_preserve_earlier_candidate(self):
        a = candidate(self.reference, self.reference, -1, .001, 0)
        b = candidate(self.reference, self.reference, -1, .001, 1)
        self.assertIs(choose_fallback([a, b]), a)

    def test_already_successful_clean_image_needs_no_matched_updates(self):
        reference = np.full((8, 8, 3), 160, dtype=np.uint8)
        output = run_pair(self.detector, self.lpips, reference)
        matched = output['PGD10_MATCHED'][1]
        self.assertEqual(matched['actual_gradient_steps'], 0)
        self.assertEqual(matched['refinement'], [])
        np.testing.assert_array_equal(output['PGD10_MATCHED'][0], reference)


if __name__ == '__main__':
    unittest.main()
