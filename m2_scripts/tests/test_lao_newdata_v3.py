"""Lightweight provenance-gate checks; never import models or generate images."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "StealthDiffusion/m3_lao_newdata_v3.py"
SPEC = importlib.util.spec_from_file_location("newdata_v3_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateFreezeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.evidence = self.root / "development.json"
        self.evidence.write_text('{"split": "development"}\n')
        self.source = self.root / "candidate.py"
        self.source.write_text("METHOD = 'frozen'\n")
        self.freeze = self.root / "candidate_freeze.json"
        self.document = {
            "status": "frozen", "created_at": "2026-01-01T00:00:00+00:00",
            "selected_configuration": {"test": True},
            "development_evidence": {"path": str(self.evidence), "sha256": MODULE.sha256(self.evidence)},
            "source_sha256": {str(self.source): MODULE.sha256(self.source)},
        }
        self.save()

    def save(self):
        self.freeze.write_text(json.dumps(self.document))

    def test_valid_freeze_binds_file_and_all_evidence(self):
        frozen = MODULE.validate_candidate_freeze(self.freeze)
        self.assertEqual(frozen["sha256"], MODULE.sha256(self.freeze))
        self.assertEqual(frozen["selected_configuration"], {"test": True})
        self.assertEqual(MODULE.validate_candidate_freeze(self.freeze, frozen), frozen)

    def test_file_metadata_change_is_rejected_after_preparation(self):
        frozen = MODULE.validate_candidate_freeze(self.freeze)
        self.document["selected_configuration"] = {"test": False}
        self.save()
        with self.assertRaisesRegex(ValueError, "differs from the frozen generation plan"):
            MODULE.validate_candidate_freeze(self.freeze, frozen)

    def test_changed_development_evidence_is_rejected(self):
        self.evidence.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Development evidence SHA-256 mismatch"):
            MODULE.validate_candidate_freeze(self.freeze)

    def test_changed_candidate_source_is_rejected(self):
        self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Frozen candidate source SHA-256 mismatch"):
            MODULE.validate_candidate_freeze(self.freeze)

    def test_relative_or_missing_evidence_is_rejected(self):
        for value in ("relative.json", str(self.root / "missing.json")):
            with self.subTest(value=value):
                self.document["development_evidence"]["path"] = value
                self.save()
                with self.assertRaises(ValueError):
                    MODULE.validate_candidate_freeze(self.freeze)

    def test_unfrozen_invalid_time_or_missing_source_map_is_rejected(self):
        for key, value in (("status", "draft"), ("created_at", "2026-01-01T00:00:00"),
                           ("created_at", "2999-01-01T00:00:00+00:00"),
                           ("selected_configuration", []), ("source_sha256", {})):
            with self.subTest(key=key, value=value):
                original = self.document[key]
                self.document[key] = value
                self.save()
                with self.assertRaises(ValueError):
                    MODULE.validate_candidate_freeze(self.freeze)
                self.document[key] = original

    def test_both_entrypoints_gate_before_any_model_audit_or_output(self):
        missing = self.root / "missing_freeze.json"
        for operation in (MODULE.prepare, MODULE.generate):
            with self.subTest(operation=operation.__name__):
                out = self.root / operation.__name__ / "data"
                with patch.object(MODULE, "model_provenance") as model_audit:
                    with self.assertRaisesRegex(ValueError, "candidate_freeze"):
                        operation(out, self.root / "model", "mps", missing)
                    model_audit.assert_not_called()
                self.assertFalse(out.exists())

    def test_cli_requires_candidate_freeze_for_prepare_and_generate(self):
        for mode in ("--prepare_only", "--generate"):
            with self.subTest(mode=mode), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as caught:
                    MODULE.main(["--out", str(self.root / "data"), "--model_path", "/abs/model", mode])
                self.assertEqual(caught.exception.code, 2)

    def test_first_batch_manifest_change_fails_before_reading_any_images(self):
        batch = self.root / "first_batch"
        batch.mkdir()
        (batch / "data_manifest.json").write_text('{}\n')
        with patch.object(MODULE, "FIRST_BATCH_ROOT", batch):
            with self.assertRaisesRegex(ValueError, "Prior validation manifest SHA-256 mismatch"):
                MODULE.audit_first_batch()

    def test_second_batch_manifest_change_fails_before_reading_any_images(self):
        batch = self.root / "second_batch"
        batch.mkdir()
        (batch / "data_manifest.json").write_text('{}\n')
        with patch.object(MODULE, "SECOND_BATCH_ROOT", batch):
            with self.assertRaisesRegex(ValueError, "Prior validation manifest SHA-256 mismatch"):
                MODULE.audit_second_batch()

    def test_root_option_passes_data_child_without_creating_outputs(self):
        with patch.object(MODULE, "prepare") as prepare, patch.object(MODULE, "generate") as generate:
            MODULE.main(["--root", str(self.root / "validation"), "--model_path", "/abs/model",
                         "--candidate_freeze", str(self.freeze), "--prepare_only"])
            self.assertEqual(prepare.call_args.args[0], (self.root / "validation/data").resolve())
            generate.assert_not_called()
        self.assertFalse((self.root / "validation").exists())

    def second_batch_fixture(self):
        """Metadata-only fixture; actual canonical validation is not bypassed in production."""
        data = self.root / "second/data"
        data.mkdir(parents=True)
        reference = MODULE.validate_candidate_freeze(self.freeze)
        plan = {"created_at": "2026-01-02T00:00:00+00:00", "old_inputs": {}, "candidate_freeze": reference}
        MODULE.write_json(data / "generation_plan.json", plan)
        manifest = {"generation_plan": plan, "old_inputs": {}, "candidate_freeze": reference, "generated": []}
        MODULE.write_json(data.parent / "data_manifest.json", manifest)
        manifest_hash = MODULE.sha256(data.parent / "data_manifest.json")
        MODULE.write_json(data / "generation_state.json", {"status": "verified", "completed_images": 50,
                          "plan_sha256": MODULE.sha256(data / "generation_plan.json"),
                          "data_manifest_sha256": manifest_hash})
        (data / "generation_records.jsonl").write_text("")
        evidence = {"manifest_path": str(data.parent / "data_manifest.json"), "plan_path": str(data / "generation_plan.json"),
                    "plan_sha256": MODULE.sha256(data / "generation_plan.json")}
        return data, manifest_hash, evidence

    def check_second(self, manifest_hash, evidence):
        with patch.object(MODULE, "audit_prior_batch", return_value=([], set(), set(), evidence)), \
             patch.object(MODULE, "SECOND_BATCH_MANIFEST_SHA256", manifest_hash):
            return MODULE.audit_second_batch()

    def test_second_batch_binds_frozen_candidate_and_generation_logs(self):
        data, manifest_hash, evidence = self.second_batch_fixture()
        result = self.check_second(manifest_hash, evidence)[3]
        self.assertEqual(result["candidate_freeze"]["sha256"], MODULE.sha256(self.freeze))
        self.assertEqual(result["generation_records_sha256"], MODULE.sha256(data / "generation_records.jsonl"))

    def test_second_batch_rejects_modified_embedded_plan(self):
        data, manifest_hash, evidence = self.second_batch_fixture()
        manifest = MODULE.read_json(data.parent / "data_manifest.json")
        manifest["generation_plan"]["created_at"] = "2026-01-03T00:00:00+00:00"
        MODULE.write_json(data.parent / "data_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "full frozen data plan"):
            self.check_second(manifest_hash, evidence)

    def test_second_batch_rejects_data_created_before_freeze(self):
        data, manifest_hash, evidence = self.second_batch_fixture()
        plan = MODULE.read_json(data / "generation_plan.json")
        plan["created_at"] = "2025-12-31T00:00:00+00:00"
        MODULE.write_json(data / "generation_plan.json", plan)
        manifest = MODULE.read_json(data.parent / "data_manifest.json")
        manifest["generation_plan"] = plan
        MODULE.write_json(data.parent / "data_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "preceded its candidate freeze"):
            self.check_second(manifest_hash, evidence)

    def test_second_batch_rejects_unverified_state(self):
        data, manifest_hash, evidence = self.second_batch_fixture()
        state = MODULE.read_json(data / "generation_state.json")
        state["status"] = "generating"
        MODULE.write_json(data / "generation_state.json", state)
        with self.assertRaisesRegex(ValueError, "expected data provenance"):
            self.check_second(manifest_hash, evidence)

    def test_second_batch_rejects_changed_generation_records(self):
        data, manifest_hash, evidence = self.second_batch_fixture()
        (data / "generation_records.jsonl").write_text('{"seed": 999}\n')
        with self.assertRaisesRegex(ValueError, "generation log differs"):
            self.check_second(manifest_hash, evidence)


if __name__ == "__main__":
    unittest.main()
