from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from peer_eval.predecodeguard.contract import SHAPE, sha256
from peer_eval.predecodeguard.evaluate import evaluate_batch


class FakeGuard:
    threshold = 0.6528052091598511

    def score(self, array):
        return 0.8

    def flagged(self, score):
        return score >= self.threshold


class SavedFileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "latents").mkdir()
        self.returned = self.root / "returned"
        self.returned.mkdir()
        samples = []
        for i in range(10):
            name = f"toy_{i:02d}.npy"
            source = self.root / "latents" / name
            np.save(source, np.zeros(SHAPE, np.float16))
            np.save(self.returned / name, np.zeros(SHAPE, np.float16))
            samples.append(dict(filename=name, baseline_score=0.8, baseline_flagged=True, sha256=sha256(source)))
        self.guard = FakeGuard()
        self.guard.manifest = {"samples":samples, "perturbation":{"epsilon":0.1}}
        self.names = [r["filename"] for r in samples]

    def evaluate(self, **kw):
        return evaluate_batch(self.root, self.returned, self.guard, method="lpgd", attempt_names=self.names, **kw)

    def test_identity_and_complete_denominator(self):
        rows, audit = self.evaluate()
        self.assertTrue(audit["complete"])
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["valid"] and r["identical"] and r["linf"] == 0 for r in rows))

    def test_missing_and_malformed_not_dropped(self):
        (self.returned / self.names[0]).unlink()
        (self.returned / self.names[1]).write_bytes(b"not an npy")
        rows, audit = self.evaluate()
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(r["attempted"] for r in rows), 10)
        self.assertFalse(rows[0]["valid"])
        self.assertFalse(rows[1]["valid"])
        self.assertFalse(audit["complete"])

    def test_extra_file_invalidates_batch(self):
        (self.returned / "metadata.json").write_text("{}")
        rows, audit = self.evaluate()
        self.assertEqual(audit["extra"], ["metadata.json"])
        self.assertFalse(any(r["valid"] for r in rows))

    def test_fp16_rounding_can_invalidate_budget(self):
        a = np.full(SHAPE, np.float16(0.10004))
        np.save(self.returned / self.names[0], a)
        rows, _ = self.evaluate()
        self.assertFalse(rows[0]["valid"])
        self.assertIn("budget exceeded", rows[0]["error"])

    def test_generation_error_survives_reporting(self):
        rows, _ = self.evaluate(records={self.names[0]:{"error":"synthetic failed run"}})
        self.assertIn("synthetic failed run", rows[0]["error"])
        self.assertFalse(rows[0]["valid"])

    def test_unreadable_file_records_error_not_abort(self):
        original = sha256
        def failed(path):
            if Path(path).name == self.names[0]:
                raise PermissionError("synthetic permission error")
            return original(path)
        with patch("peer_eval.predecodeguard.evaluate.sha256", side_effect=failed):
            rows, _ = self.evaluate()
        self.assertEqual(len(rows), 10)
        self.assertFalse(rows[0]["valid"])
        self.assertIn("permission error", rows[0]["error"])


if __name__ == "__main__":
    unittest.main()
