"""Reporting checks use synthetic dictionaries only; no model is executed."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from peer_eval.predecodeguard.report import summarize, write_reports


def sample(name="sample.pt", **changes):
    row = dict(filename=name, method="planned_method", attempted=True,
               baseline_score=0.9, baseline_flagged=True, score=0.3, flagged=False,
               score_displacement=-0.6, linf=0.05, l2=0.1, valid=True,
               status="ok", error="", sha256="a" * 64, original_sha256="b" * 64,
               identical=False, seconds=1.0, steps=2)
    row.update(changes)
    return row


def review(row, **changes):
    value = dict(filename=row["filename"], sha256=row["sha256"], decode_status="ok",
                 usability="usable", content_label="retained", reviewer="reviewer-1",
                 decoder_reference="fixed-decoder-v1")
    value.update(changes)
    return value


class ReportingTests(unittest.TestCase):
    def test_failures_remain_in_denominator_controls_excluded(self):
        rows = [sample(), sample("missing.pt", valid=False, status="missing", score=None, flagged=None, sha256=None),
                sample("control.pt", method="roundtrip", attempted=True),
                sample("negative.pt", baseline_flagged=False)]
        result = summarize(rows)
        self.assertEqual(result["attempted"], 3)
        self.assertEqual(result["eligible_attempts"], 2)
        self.assertEqual(result["detector_flips"], 1)
        self.assertEqual(result["detector_flip_rate"], 0.5)
        self.assertEqual(result["detector_flip_rate_all_attempts"], 1 / 3)
        self.assertEqual(result["detector_flip_rate_all_rows_including_controls"], 1 / 4)
        self.assertEqual(result["status_counts"]["missing"], 1)
        self.assertEqual(result["review_pending"], 1)
        self.assertIsNone(result["confirmed_count"])
        self.assertIsNone(result["confirmed_rate"])
        self.assertIsNone(result["confirmed_rate_all_rows_including_controls"])

    def test_partial_review_does_not_publish_final_rate(self):
        rows = [sample("one.pt"), sample("two.pt")]
        result = summarize(rows, [review(rows[0])])
        self.assertEqual(result["confirmed_count"], 1)
        self.assertEqual(result["review_pending"], 1)
        self.assertIsNone(result["confirmed_rate"])

    def test_full_adjudication_uses_all_eligible_attempts(self):
        rows = [sample("one.pt"), sample("two.pt"), sample("failed.pt", valid=False, status="missing", sha256=None)]
        result = summarize(rows, [review(rows[0]), review(rows[1], content_label="changed")])
        self.assertEqual(result["confirmed_count"], 1)
        self.assertEqual(result["confirmed_rate"], 1 / 3)
        self.assertEqual(result["review_pending"], 0)
        self.assertTrue(result["review_complete"])

    def test_review_must_match_filename_and_hash(self):
        row = sample()
        result = summarize([row], [review(row, sha256="c" * 64)])
        self.assertEqual(result["review_pending"], 1)
        self.assertEqual(result["unmatched_review_rows"], 1)
        self.assertIsNone(result["confirmed_count"])

    def test_optional_original_hash_must_match_source(self):
        row = sample()
        result = summarize([row], [review(row, original_sha256=row["original_sha256"])])
        self.assertEqual(result["confirmed_count"], 1)
        with self.assertRaisesRegex(ValueError, "original_sha256"):
            summarize([row], [review(row, original_sha256="c" * 64)])

    def test_uncertain_and_unattributed_reviews_remain_pending(self):
        row = sample()
        for changes in ({"content_label": "uncertain"}, {"reviewer": ""}, {"decoder_reference": ""}):
            with self.subTest(changes=changes):
                result = summarize([row], [review(row, **changes)])
                self.assertEqual(result["review_pending"], 1)
                self.assertEqual(result["confirmed_count"], 0)
                self.assertIsNone(result["confirmed_rate"])

    def test_decisive_negative_does_not_require_other_review_fields(self):
        row = sample()
        result = summarize([row], [review(row, decode_status="failed", usability="", content_label="")])
        self.assertEqual(result["confirmed_count"], 0)
        self.assertEqual(result["confirmed_rate"], 0)
        self.assertEqual(result["review_pending"], 0)

    def test_no_flip_is_definitive_without_content_review(self):
        result = summarize([sample(flagged=True, score=0.8)])
        self.assertEqual(result["review_required"], 0)
        self.assertEqual(result["confirmed_count"], 0)
        self.assertEqual(result["confirmed_rate"], 0)

    def test_noop_is_not_attack_success(self):
        result = summarize([sample(method="R0", identical=True)])
        self.assertEqual(result["attempted"], 0)
        self.assertEqual(result["eligible_attempts"], 0)
        self.assertIsNone(result["detector_flip_rate"])
        self.assertIsNone(result["confirmed_count"])
        self.assertIsNone(result["confirmed_rate"])

    def test_method_groups_do_not_inflate_source_samples(self):
        rows = [sample(method="method_a"), sample(method="method_b", sha256="c" * 64)]
        result = summarize(rows)
        self.assertEqual(result["source_samples"], 1)
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["by_method"]["method_a"]["attempted"], 1)

    def test_conflicting_duplicate_or_invalid_review_rejected(self):
        row = sample()
        with self.assertRaises(ValueError):
            summarize([row], [review(row), review(row, content_label="changed")])
        with self.assertRaises(ValueError):
            summarize([row], [review(row, usability="yes")])

    def test_artifacts_include_failed_rows_and_no_review_scores(self):
        rows = [sample(), sample("missing.pt", valid=False, status="missing", sha256=None)]
        result = summarize(rows)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_reports(out, rows, result)
            self.assertEqual({path.name for path in out.iterdir()}, {"per_sample.csv", "summary.json", "summary.md", "review_template.csv"})
            with (out / "per_sample.csv").open() as handle:
                stored = list(csv.DictReader(handle))
            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[1]["status"], "missing")
            with (out / "review_template.csv").open() as handle:
                reader = csv.DictReader(handle)
                self.assertNotIn("score", reader.fieldnames)
                self.assertNotIn("method", reader.fieldnames)
                self.assertIn("original_sha256", reader.fieldnames)
                template = list(reader)
            self.assertEqual(template[0]["content_label"], "")
            self.assertIsNone(json.loads((out / "summary.json").read_text())["confirmed_rate"])

    def test_empty_summary_is_valid(self):
        result = summarize([])
        self.assertEqual(result["attempted"], 0)
        self.assertIsNone(result["confirmed_rate"])

    def test_blank_template_rows_are_not_review_evidence(self):
        rows = [sample(), sample("missing.pt", valid=False, sha256=None)]
        template = [{"filename": row["filename"], "sha256": row["sha256"], "content_label": ""} for row in rows]
        result = summarize(rows, template)
        self.assertEqual(result["review_submitted"], 0)
        self.assertIsNone(result["confirmed_count"])

    def test_non_boolean_attempted_cannot_silently_remove_denominator(self):
        for attempted in ("True", "False", 1, 0, None):
            with self.subTest(attempted=attempted), self.assertRaises(ValueError):
                summarize([sample(attempted=attempted)])

    def test_all_output_audit_is_unique_and_separate_from_candidates(self):
        rows = [sample(method="copy", attempted=False), sample(method="roundtrip", attempted=False)]
        result = summarize(rows, [review(rows[0])])
        self.assertEqual(result["review_required"], 0)
        self.assertEqual(result["all_output_unique_valid_files"], 1)
        self.assertEqual(result["all_output_review_submitted"], 1)
        self.assertFalse(result["candidate_review_complete"])


if __name__ == "__main__":
    unittest.main()
