"""Saved-file/statistics contract tests; no torch or model execution."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from PIL import Image


MODULE = Path(__file__).resolve().parents[2] / "StealthDiffusion" / "m3_lao_v3_report.py"
spec = importlib.util.spec_from_file_location("report_v3_under_test", MODULE)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    n = 4
    mapping = {}
    for name in reporter.METHODS:
        folder = root / name
        folder.mkdir()
        mapping[name] = str(folder)
        (folder / "run_config.json").write_text(json.dumps({"variant": name, "budget_pixels": 8}))
        records = []
        for index in range(n):
            key = f"{index:04d}"
            original = Image.new("RGB", (224, 224), (120 + index, 120, 120))
            output = Image.new("RGB", (224, 224), (121 + index, 120, 120))
            original.save(folder / f"{key}_originImage.png")
            output.save(folder / f"{key}_adv_image.png")
            if name in ("L", "H", "L_EOT", "H_EOT"):
                output.save(folder / f"{key}_reset_image.png")
            if name in ("H", "H_EOT"):
                output.save(folder / f"{key}_lao_image.png")
            records.append({"image_id": key, "source": f"sample{index}.png", "seconds": 1.,
                            "max_difference_pixels": 1, "lpips": .01,
                            "actual_gradient_steps": 3, "latent_gradient_steps": [{"step": 1}],
                            "pixel_gradient_steps": [{"step": 1}, {"step": 2}],
                            "compute": {"detector_forwards": 7, "detector_view_examples": 9,
                                        "lpips_forwards": 7, "renderer_forwards": 2},
                            "timing_is_shared": False})
        (folder / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    evaluation = root / "eval"
    evaluation.mkdir()
    rows, real, summaries = [], [], []
    psnr = reporter.pixel_metrics(Image.new("RGB", (224, 224), (120, 120, 120)),
                                  Image.new("RGB", (224, 224), (121, 120, 120)))[1]
    for det in reporter.DETECTORS:
        for cond in reporter.CONDITIONS:
            for index in range(n):
                pr = int(not (index == 0 and cond == "jpeg75"))
                real.append({"detector": det, "condition": cond, "image": f"real{index}.png",
                             "pred": pr, "real_margin": 1 if pr else -1})
            for name in reporter.METHODS:
                cell = []
                for index in range(n):
                    clean_pred = int(index == 3 or (index == 2 and cond == "jpeg85"))
                    adv_pred = int(index in ((0, 1, 2, 3) if name == "H_EOT" else (0, 2, 3)))
                    row = {"run": name, "detector": det, "condition": cond, "image_id": f"{index:04d}",
                           "clean_pred": clean_pred, "clean_margin": 1 if clean_pred else -1,
                           "adv_pred": adv_pred, "adv_margin": 1 if adv_pred else -1,
                           "psnr": psnr if cond == "none" else "", "linf": 1/255 if cond == "none" else "",
                           "lpips": .01 if cond == "none" else "", "seconds": 1.}
                    rows.append(row)
                    cell.append(row)
                caught = [r for r in cell if r["clean_pred"] == 0]
                summaries.append({"run": name, "detector": det, "condition": cond, "n": n,
                                  "n_clean_detected": len(caught),
                                  "cond_asr": sum(r["adv_pred"] for r in caught) / len(caught),
                                  "uncond_escape": sum(r["adv_pred"] for r in cell) / n,
                                  "real_fpr": .25 if cond == "jpeg75" else 0., "lpips": .01})
    write_csv(evaluation / "per_image.csv", rows)
    write_csv(evaluation / "summary.csv", summaries)
    write_csv(evaluation / "real_predictions.csv", real)
    return evaluation, mapping, n


class StatisticsTest(unittest.TestCase):
    def test_known_discordant_pair_and_exact_binomial(self):
        result = reporter.paired([True] * 15 + [False] * 26,
                                 [False] * 41, list(range(41)))
        self.assertEqual(result["left_only"], 15)
        self.assertEqual(result["exact_mcnemar_p"], 2 / 2 ** 15)
        self.assertAlmostEqual(result["paired_difference"], 15/41)
        self.assertAlmostEqual(result["ci95_lower"], 9/41)
        self.assertAlmostEqual(result["ci95_upper"], 21/41)
        self.assertGreater(result["conservative_ci95_lower"], 0)

    def test_zero_discordance_does_not_imply_noninferiority(self):
        result = reporter.paired([True]*50, [True]*50, list(range(50)))
        self.assertEqual(result["ci95_lower"], 0)
        self.assertLess(result["conservative_ci95_lower"], -.05)
        self.assertEqual(result["exact_mcnemar_p"], 1.)

    def test_empty_cohort_has_no_inferred_success(self):
        result = reporter.paired([], [], [])
        self.assertIsNone(result["paired_difference"])
        self.assertIsNone(result["exact_mcnemar_p"])

    def test_cp_matches_closed_form_boundaries(self):
        lo, hi = reporter.clopper_pearson(0, 50)
        self.assertEqual(lo, 0)
        self.assertAlmostEqual(hi, 1 - (.025/2) ** (1/50))
        lo, hi = reporter.clopper_pearson(50, 50)
        self.assertEqual(hi, 1)
        self.assertAlmostEqual(lo, (.025/2) ** (1/50))


class SavedPNGReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evaluation, self.mapping, self.n = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return reporter.build_report(self.evaluation, self.mapping, self.n)

    def change_rows(self, filename, fn):
        path = self.evaluation / filename
        rows = reporter.read_csv(path)
        fn(rows)
        write_csv(path, rows)

    def test_complete_cohort_counts_source_budget_reset_stage_and_compute(self):
        result = self.build()
        self.assertEqual(result["primary"]["n_eligible"], 2)
        self.assertEqual(result["primary"]["left_success"], 2)
        self.assertEqual(result["primary"]["right_success"], 1)
        self.assertEqual(result["primary"]["left_only_ids"], ["0001"])
        self.assertEqual(result["integrity"]["summary_cells"], 112)
        self.assertEqual(result["integrity"]["pixel_checks"], 28)
        self.assertEqual(result["integrity"]["independent_reset_checks"], 16)
        self.assertEqual(result["integrity"]["hybrid_latent_stage_checks"], 8)
        self.assertEqual(result["methods"]["P"]["compute"]["latent_gradient_steps"]["total"], 4)
        self.assertEqual(result["methods"]["P"]["compute"]["pixel_gradient_steps"]["total"], 8)
        fpr = next(r for r in result["real_fpr"] if r["detector"] == "E" and r["condition"] == "jpeg75")
        self.assertEqual(fpr["real_fpr"], .25)
        self.assertEqual(result["n_generated"], 4)

    def test_true_saved_budget_zero_tolerance(self):
        path = Path(self.mapping["P"]) / "0000_adv_image.png"
        altered = Image.open(path).copy()
        altered.putpixel((0, 0), (129, 120, 120))
        altered.save(path)
        with self.assertRaisesRegex(ValueError, "integer budget violation"):
            self.build()

    def test_independent_reset_mismatch_is_rejected(self):
        path = Path(self.mapping["L_EOT"]) / "0000_reset_image.png"
        altered = Image.open(path).copy()
        altered.putpixel((0, 0), (122, 120, 120))
        altered.save(path)
        with self.assertRaisesRegex(ValueError, "Reset differs"):
            self.build()

    def test_hybrid_stage_mismatch_is_rejected(self):
        path = Path(self.mapping["H"]) / "0000_lao_image.png"
        altered = Image.open(path).copy()
        altered.putpixel((0, 0), (122, 120, 120))
        altered.save(path)
        with self.assertRaisesRegex(ValueError, "Hybrid latent stage differs"):
            self.build()

    def test_mismatched_sources_rejected(self):
        path = Path(self.mapping["P"]) / "0000_originImage.png"
        altered = Image.open(path).copy()
        altered.putpixel((0, 0), (122, 120, 120))
        altered.save(path)
        with self.assertRaisesRegex(ValueError, "Original PNG bytes differ"):
            self.build()

    def test_clean_eligibility_must_match_between_methods(self):
        def change(rows):
            row = next(r for r in rows if r["run"] == "P" and r["image_id"] == "0000")
            row.update(clean_pred=1, clean_margin=1)
        self.change_rows("per_image.csv", change)
        with self.assertRaisesRegex(ValueError, "Clean prediction differs"):
            self.build()

    def test_missing_transform_row_fails(self):
        self.change_rows("per_image.csv", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "Incomplete CSV coverage"):
            self.build()

    def test_lpips_not_dropped_under_compression(self):
        def change(rows):
            for row in rows:
                if row["run"] == "H_EOT" and row["image_id"] == "0001" and row["condition"] == "none":
                    row["lpips"] = .04
        self.change_rows("per_image.csv", change)
        path = Path(self.mapping["H_EOT"]) / "records.jsonl"
        records = [json.loads(r) for r in path.read_text().splitlines()]
        records[1]["lpips"] = .04
        path.write_text("\n".join(json.dumps(r) for r in records))
        self.change_rows("summary.csv", lambda rows: [row.update(lpips=.0175) for row in rows if row["run"] == "H_EOT"])
        result = self.build()
        self.assertEqual(result["primary"]["left_success"], 1)
        self.assertEqual(result["primary"]["n_eligible"], 2)
        self.assertFalse(result["methods"]["H_EOT"]["quality"]["mean_lpips_guard_pass"])
        self.assertFalse(result["primary"]["superiority_criterion_met"])
        self.assertEqual(result["methods"]["H_EOT"]["quality"]["n"], 4)

    def test_saved_psnr_is_recomputed(self):
        self.change_rows("per_image.csv", lambda rows: rows[0].update(psnr=99))
        with self.assertRaisesRegex(ValueError, "actual PNG PSNR"):
            self.build()

    def test_record_mismatch_and_summary_mismatch_are_rejected(self):
        self.change_rows("summary.csv", lambda rows: rows[0].update(real_fpr=.5))
        with self.assertRaisesRegex(ValueError, "summary FPR"):
            self.build()

    def test_record_margin_accepts_known_batch_error_without_changing_predictions(self):
        def change(rows):
            row = next(r for r in rows if r["run"] == "P_EOT" and r["detector"] == "E"
                       and r["condition"] == "none" and r["image_id"] == "0000")
            row["adv_margin"] = 18.022096633911133
        self.change_rows("per_image.csv", change)
        path = Path(self.mapping["P_EOT"]) / "records.jsonl"
        records = [json.loads(r) for r in path.read_text().splitlines()]
        records[0]["final"] = {"E": {"pred": 1, "real_margin": 18.022123336791992}}
        path.write_text("\n".join(json.dumps(r) for r in records))
        result = self.build()
        self.assertEqual(result["primary"]["left_only_ids"], ["0001"])
        records[0]["final"]["E"]["pred"] = 0
        path.write_text("\n".join(json.dumps(r) for r in records))
        with self.assertRaisesRegex(ValueError, "class changed"):
            self.build()

    def test_record_margin_tolerance_still_rejects_material_discrepancy(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            reporter.close(18.022123336791992, 18.023, "record margin", 2e-5, rel_tol=1e-5)
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            reporter.close(.10003, .1, "near boundary", 2e-5, rel_tol=1e-5)

    def test_overwrite_refused_and_strict_json_outputs(self):
        report = self.build()
        path = self.root / "report"
        reporter.write_report(report, path)
        self.assertTrue((path / "report.md").is_file())
        self.assertTrue((path / "per_image_joint_outcomes.csv").is_file())
        parsed = json.loads((path / "report.json").read_text())
        self.assertEqual(parsed["primary"]["n_eligible"], 2)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            reporter.write_report(report, path)


if __name__ == "__main__":
    unittest.main()
