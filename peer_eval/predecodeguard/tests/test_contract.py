import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from peer_eval.predecodeguard.adapter import FrozenGuard
from peer_eval.predecodeguard.contract import (
    HERE, METHODS, SHAPE, check_protocol, load_latent, package_manifest,
    proposal_digest, require_agreement, sha256,
)
from peer_eval.predecodeguard.run import check_output_location, main


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = json.loads((HERE / "protocol.pending.json").read_text())
        self.manifest = {"threshold":self.protocol["threshold"], "perturbation":{"epsilon":0.1},
                         "samples":[{"filename":n} for n in self.protocol["sample_filenames"]]}

    def approved(self):
        p = copy.deepcopy(self.protocol)
        evidence = self.root / "synthetic_confirmation.txt"
        evidence.write_text("Synthetic unit-test fixture only; NOT a peer agreement.")
        p["status"] = "confirmed"
        p["agreement"] = {"confirmed_by":"synthetic fixture", "confirmed_at":"2026-10-18T00:00:00+08:00",
                          "reference":"unit-test-only", "evidence_path":str(evidence),
                          "evidence_sha256":sha256(evidence), "proposal_sha256":proposal_digest(p)}
        return p

    def test_pending_blocks_every_perturbation(self):
        for method in METHODS:
            with self.subTest(method=method), self.assertRaises(PermissionError):
                require_agreement(self.protocol, method, execute_approved=True, protocol_path=self.root / "p.json")

    def test_confirmation_and_execution_flag_both_required(self):
        p = self.approved()
        with self.assertRaises(PermissionError):
            require_agreement(p, "lpgd", execute_approved=False, protocol_path=self.root / "p.json")
        require_agreement(p, "lpgd", execute_approved=True, protocol_path=self.root / "p.json")

    def test_changed_proposal_and_evidence_rejected(self):
        p = self.approved()
        p["methods"]["lpgd"]["steps"] += 1
        with self.assertRaises(PermissionError):
            require_agreement(p, "lpgd", execute_approved=True, protocol_path=self.root / "p.json")
        p = self.approved()
        Path(p["agreement"]["evidence_path"]).write_text("modified fixture")
        with self.assertRaises(PermissionError):
            require_agreement(p, "lpgd", execute_approved=True, protocol_path=self.root / "p.json")

    def test_status_only_not_approval(self):
        p = copy.deepcopy(self.protocol)
        p["status"] = "confirmed"
        with self.assertRaises(PermissionError):
            require_agreement(p, "random", execute_approved=True, protocol_path=self.root / "p.json")

    def test_budget_threshold_and_allocation_frozen(self):
        check_protocol(self.protocol, self.manifest, self.protocol["package_manifest_sha256"])
        for key, value in [("epsilon", 0.10000001), ("epsilon", float("nan")), ("tolerance", 1e-6)]:
            p = copy.deepcopy(self.protocol)
            p["budget"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check_protocol(p, self.manifest, p["package_manifest_sha256"])
        p = copy.deepcopy(self.protocol)
        p["threshold"] = 0.5
        with self.assertRaises(ValueError):
            check_protocol(p, self.manifest, p["package_manifest_sha256"])
        p = copy.deepcopy(self.protocol)
        p["methods"]["lstop"]["control_filenames"] = ["unknown.npy"]
        with self.assertRaises(ValueError):
            check_protocol(p, self.manifest, p["package_manifest_sha256"])

    def test_cli_gate_precedes_model_and_output_creation(self):
        for method in METHODS:
            out = self.root / method
            with patch("peer_eval.predecodeguard.run.package_manifest", return_value=self.manifest), \
                 patch("peer_eval.predecodeguard.run.sha256", return_value=self.protocol["package_manifest_sha256"]), \
                 patch("peer_eval.predecodeguard.run.FrozenGuard") as guard:
                with self.assertRaises(PermissionError):
                    main(["experiment", "--package", str(self.root), "--out", str(out),
                          "--method", method, "--execute-approved"])
                guard.assert_not_called()
                self.assertFalse(out.exists())

    def test_early_stop_margin_is_validated_and_bound_to_agreement(self):
        p = copy.deepcopy(self.protocol)
        p["methods"]["lstop"]["early_stop_margin"] = 0.001
        check_protocol(p, self.manifest, p["package_manifest_sha256"])
        for value in (-0.001, True, float("nan"), float("inf"), p["threshold"]):
            invalid = copy.deepcopy(p)
            invalid["methods"]["lstop"]["early_stop_margin"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_protocol(invalid, self.manifest, invalid["package_manifest_sha256"])
        for method in ("random", "lpgd"):
            invalid = copy.deepcopy(p)
            invalid["methods"][method]["early_stop_margin"] = 0.001
            with self.subTest(method=method), self.assertRaises(ValueError):
                check_protocol(invalid, self.manifest, invalid["package_manifest_sha256"])
        approved = self.approved()
        approved["methods"]["lstop"]["early_stop_margin"] = 0.001
        with self.assertRaises(PermissionError):
            require_agreement(approved, "lstop", execute_approved=True, protocol_path=self.root / "p.json")

    def test_output_cannot_overlap_source(self):
        source = self.root / "source"
        for out in [source, source / "new", self.root]:
            with self.assertRaises(ValueError):
                check_output_location(out, source)
        check_output_location(self.root / "output", source)

    def test_threshold_equality_is_flagged(self):
        guard = FrozenGuard.__new__(FrozenGuard)
        guard.threshold = 0.6528052091598511
        self.assertTrue(guard.flagged(guard.threshold))
        self.assertFalse(guard.flagged(guard.threshold - 1e-8))

    def test_array_dtype_shape_and_finiteness(self):
        path = self.root / "sample.npy"
        for array in [np.zeros(SHAPE, np.float32), np.zeros((1,16,2,2), np.float16), np.full(SHAPE, np.nan, np.float16)]:
            np.save(path, array)
            with self.assertRaises(ValueError):
                load_latent(path)
        np.save(path, np.zeros(SHAPE, np.float16))
        self.assertEqual(load_latent(path).dtype, np.float16)

    def test_changed_pinned_package_file_rejected_before_import(self):
        source = self.root / "source"
        source.mkdir()
        script = source / "model.py"
        script.write_text("raise RuntimeError('this fixture must never be executed')")
        lock = self.root / "lock.json"
        lock.write_text(json.dumps({"files":{"model.py":sha256(script)}}))
        script.write_text("altered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            package_manifest(source, lock)


if __name__ == "__main__":
    unittest.main()
