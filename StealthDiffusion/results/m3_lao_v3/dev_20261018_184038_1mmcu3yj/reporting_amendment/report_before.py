#!/usr/bin/env python3
"""Audit the six frozen LAO/PGD/JPEG strategies without loading any model.

LPIPS and predictions come from m2_evaluate's saved-PNG evaluation. Pixel
budgets and PSNR are independently recomputed. This reporter does not certify
data novelty or model/source freezing; those require the runner's separate audit.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageChops


METHODS = ("Z", "P", "L", "H", "P_EOT", "L_EOT", "H_EOT")
CONDITIONS = ("none", "jpeg75", "jpeg85", "jpeg95")
TRAIN_CONDITIONS = ("none", "jpeg80", "jpeg90")
DETECTORS = ("E", "R", "D", "S")
BUDGET = 8
LPIPS_MAX = .03
PSNR_MIN = 30.
MEAN_LPIPS_MAX = .015
BOOTSTRAP_SEED = 20261018
BOOTSTRAP_RESAMPLES = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        result = list(csv.DictReader(handle))
    require(result, f"Empty CSV: {path}")
    return result


def number(value, label, optional=False, infinity=False):
    if value is None or str(value).strip().lower() in ("", "none", "null", "nan"):
        require(optional, f"Missing numeric value: {label}")
        return None
    result = float(value)
    require(not math.isnan(result) and (infinity or math.isfinite(result)), f"Invalid number: {label}")
    return result


def predicted(value, label):
    result = number(value, label)
    require(result in (0, 1), f"Prediction must be 0=AI or 1=real: {label}")
    return int(result)


def close(left, right, label, abs_tol=1e-6):
    require(math.isclose(left, right, rel_tol=1e-6, abs_tol=abs_tol),
            f"Inconsistent {label}: {left} vs {right}")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def image(path):
    with Image.open(path) as opened:
        require(opened.format == "PNG" and opened.mode == "RGB" and opened.size == (224, 224),
                f"Expected canonical RGB 224x224 PNG: {path}")
        return opened.copy()


def pixels_sha(pixels):
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def pixel_metrics(original, output):
    delta = ImageChops.difference(original, output)
    maximum = max(high for _, high in delta.getextrema())
    mse = sum((index % 256) ** 2 * count for index, count in enumerate(delta.histogram())) / (224 * 224 * 3)
    return int(maximum), math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def percentile(values, probability):
    values = sorted(values)
    point = (len(values) - 1) * probability
    lower, upper = int(math.floor(point)), int(math.ceil(point))
    if lower == upper or values[lower] == values[upper]:
        return values[lower]
    if math.isinf(values[upper]):
        return values[upper]
    return values[lower] + (point - lower) * (values[upper] - values[lower])


def mcnemar(left_only, right_only):
    discordant = left_only + right_only
    if not discordant:
        return 1.
    return min(1., 2 * sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / 2 ** discordant)


def binomial_cdf(k, n, p):
    return min(1., sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1)))


def clopper_pearson(k, n, alpha=.025):
    require(0 <= k <= n and n > 0, "Invalid binomial counts")
    lower, upper, tail = 0., 1., alpha / 2
    if k:
        lo, hi = 0., 1.
        for _ in range(80):
            mid = (lo + hi) / 2
            if 1 - binomial_cdf(k - 1, n, mid) < tail:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k < n:
        lo, hi = 0., 1.
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_cdf(k, n, mid) > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def paired(left, right, ids, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    require(len(left) == len(right) == len(ids), "Unaligned paired outcomes")
    only_left = [i for i, a, b in zip(ids, left, right) if a and not b]
    only_right = [i for i, a, b in zip(ids, left, right) if b and not a]
    n = len(ids)
    result = {"n_eligible": n, "left_success": sum(left), "right_success": sum(right),
              "left_only": len(only_left), "right_only": len(only_right),
              "left_only_ids": only_left, "right_only_ids": only_right,
              "both_success": sum(a and b for a, b in zip(left, right)),
              "both_failure": sum(not a and not b for a, b in zip(left, right)),
              "bootstrap_seed": seed, "bootstrap_resamples": resamples}
    if not n:
        return {**result, "paired_difference": None, "ci95_lower": None, "ci95_upper": None,
                "exact_mcnemar_p": None, "conservative_ci95_lower": None, "conservative_ci95_upper": None}
    delta = [int(a) - int(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    boot = [sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    ci_left, ci_right = clopper_pearson(len(only_left), n), clopper_pearson(len(only_right), n)
    return {**result, "paired_difference": sum(delta) / n, "ci95_lower": percentile(boot, .025),
            "ci95_upper": percentile(boot, .975), "exact_mcnemar_p": mcnemar(len(only_left), len(only_right)),
            "conservative_ci95_lower": ci_left[0] - ci_right[1],
            "conservative_ci95_upper": ci_left[1] - ci_right[0]}


def lookup_number(record, paths):
    for path in paths:
        value = record
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            if isinstance(value, list):
                return len(value)
            return number(value, path)
    return None


def compute_summary(records):
    """Do not substitute missing counters with zero or sum shared trajectories."""
    fields = {
        "seconds": ("seconds", "elapsed_sec"),
        "gradient_steps": ("actual_gradient_steps", "actual_steps"),
        "latent_gradient_steps": ("actual_latent_gradient_steps", "compute.latent_gradient_steps", "latent_gradient_steps"),
        "pixel_gradient_steps": ("actual_pixel_gradient_steps", "compute.pixel_gradient_steps", "pixel_gradient_steps"),
        "detector_forwards": ("detector_forwards", "compute.detector_forwards", "compute.E_forwards", "objective_detector_forwards"),
        "detector_view_examples": ("detector_view_examples", "compute.detector_view_examples", "compute.E_view_examples"),
        "lpips_forwards": ("lpips_forwards", "compute.lpips_forwards", "compute.LPIPS_forwards", "objective_lpips_forwards"),
        "renderer_forwards": ("renderer_forwards", "compute.renderer_forwards", "objective_renderer_forwards"),
        "jpeg_forwards": ("jpeg_forwards", "compute.jpeg_forwards", "compute.JPEG_encodes"),
    }
    result = {}
    for label, paths in fields.items():
        values = [lookup_number(row, paths) for row in records]
        present = [v for v in values if v is not None]
        require(all(v >= 0 for v in present), f"Negative compute counter: {label}")
        result[label] = {"available_n": len(present), "total": sum(present) if present else None,
                         "mean": mean(present) if present else None, "median": median(present) if present else None,
                         "complete": len(present) == len(records)}
    result["shared_timing_records"] = sum(bool(r.get("seconds_is_shared", r.get("timing_is_shared", False))) for r in records)
    result["cross_method_totals_must_not_be_summed"] = result["shared_timing_records"] > 0
    result["per_image_compute"] = [{"image_id": row["image_id"], "compute": row.get("compute"),
                                    "preparation_seconds": row.get("preparation_seconds"),
                                    "timing_note": row.get("timing_note"),
                                    "stages": {name: {key: stage.get(key) for key in ("seconds", "compute", "actual_gradient_steps", "selected")}
                                               for name, stage in row.get("stages", {}).items()}}
                                   for row in records]
    return result


def inspect_methods(method_paths, n):
    require(set(method_paths) == set(METHODS), f"Expected exactly methods {METHODS}")
    references, source_names, ids, info, quality_pixels = {}, {}, None, {}, {}
    checks, reset_checks, stage_checks = [], [], []
    for name in METHODS:
        folder = Path(method_paths[name]).resolve()
        require(folder.is_dir(), f"Missing method directory: {folder}")
        record_path = folder / "records.jsonl"
        rows = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
        records = {}
        for record in rows:
            key = str(record.get("image_id", ""))
            require(key and key not in records, f"Missing/duplicate record ID: {name}/{key}")
            records[key] = record
        actual_ids = sorted(p.name.removesuffix("_originImage.png") for p in folder.glob("*_originImage.png"))
        adv_ids = sorted(p.name.removesuffix("_adv_image.png") for p in folder.glob("*_adv_image.png"))
        require(len(records) == len(actual_ids) == n and set(records) == set(actual_ids) == set(adv_ids),
                f"Incomplete record/PNG coverage: {name}")
        if ids is None:
            ids = actual_ids
        require(ids == actual_ids, f"Method image IDs differ: {name}")
        config_path = folder / "run_config.json"
        require(config_path.is_file(), f"Missing run_config.json: {name}")
        config = read_json(config_path)
        if "budget_pixels" in config:
            require(config["budget_pixels"] == BUDGET, f"Configuration budget differs: {name}")
        if "train_views" in config:
            expected_views = TRAIN_CONDITIONS if name.endswith("_EOT") else ("none",)
            require(tuple(config["train_views"]) == expected_views, f"Unexpected optimization views: {name}")
        if "surrogates" in config:
            require(config["surrogates"] == "E", f"Unexpected optimization detector: {name}")
        info[name] = {"path": str(folder), "records_sha256": sha(record_path),
                      "run_config_sha256": sha(config_path), "config": config,
                      "compute": compute_summary(rows), "records": records}
        for key in ids:
            original_path, output_path = folder / f"{key}_originImage.png", folder / f"{key}_adv_image.png"
            original, output = image(original_path), image(output_path)
            source_hash = sha(original_path)
            source = records[key].get("source")
            require(isinstance(source, str) and source, f"Missing source identity: {name}/{key}")
            if key not in references:
                references[key] = source_hash
                source_names[key] = source
            require(source_hash == references[key], f"Original PNG bytes differ between methods: {name}/{key}")
            require(source == source_names[key], f"Source identity differs between methods: {name}/{key}")
            maximum, psnr = pixel_metrics(original, output)
            require(maximum <= BUDGET, f"Saved PNG integer budget violation: {name}/{key}: {maximum} > {BUDGET}")
            record = records[key]
            if all(field in record for field in ("latent_gradient_steps", "pixel_gradient_steps")):
                gradients = []
                for stage in ("latent", "pixel"):
                    updates = record[f"{stage}_gradient_steps"]
                    require(isinstance(updates, list), f"Gradient history is not a list: {name}/{key}/{stage}")
                    gradients.extend(updates)
                    count_key = f"actual_{stage}_gradient_steps"
                    if count_key in record:
                        require(record[count_key] == len(updates), f"Gradient count differs from history: {name}/{key}/{stage}")
                    for update in updates:
                        for field in ("grad_l2", "grad_linf", "update_l2", "update_linf"):
                            if field in update:
                                require(number(update[field], field) >= 0, f"Invalid gradient norm: {name}/{key}")
                if "actual_gradient_steps" in record:
                    require(record["actual_gradient_steps"] == len(gradients), f"Total gradient count differs: {name}/{key}")
            for field in ("max_difference_pixels",):
                if record.get(field) is not None:
                    close(number(record[field], field), maximum, f"record {field} {name}/{key}", 0)
            for field in ("output_sha256",):
                if record.get(field):
                    require(record[field] == sha(output_path), f"Output SHA mismatch: {name}/{key}")
            if record.get("canonical_source_png_sha256"):
                require(record["canonical_source_png_sha256"] == source_hash, f"Original SHA mismatch: {name}/{key}")
            if record.get("canonical_source_pixels_sha256"):
                require(record["canonical_source_pixels_sha256"] == pixels_sha(original), f"Original pixel SHA mismatch: {name}/{key}")
            selected = record.get("selected", {})
            if selected.get("sha256"):
                require(selected["sha256"] == sha(output_path), f"Selected PNG SHA mismatch: {name}/{key}")
            if selected.get("pixels_sha256"):
                require(selected["pixels_sha256"] == pixels_sha(output), f"Selected pixel SHA mismatch: {name}/{key}")
            if "max_difference_pixels" in selected:
                require(selected["max_difference_pixels"] == maximum, f"Selected pixel budget differs: {name}/{key}")
            if record.get("source_file_sha256") and config.get("images_root"):
                source_file = Path(config["images_root"]) / Path(source).name
                require(source_file.is_file() and sha(source_file) == record["source_file_sha256"],
                        f"Canonical input file SHA mismatch: {name}/{key}")
                require(pixels_sha(image(source_file)) == pixels_sha(original), f"Canonical input pixels differ: {name}/{key}")
            quality_pixels[name, key] = {"max_difference_pixels": maximum, "psnr": psnr}
            checks.append({"run": name, "image_id": key, "source": source,
                           "origin_sha256": source_hash, "output_sha256": sha(output_path),
                           "max_difference_pixels": maximum, "psnr": psnr})
    require(len(set(references.values())) == n, "Duplicate canonical source PNG bytes across image IDs")
    for name in ("L", "H", "L_EOT", "H_EOT"):
        for key in ids:
            reset_path = Path(info[name]["path"]) / f"{key}_reset_image.png"
            baseline_path = Path(info["Z"]["path"]) / f"{key}_adv_image.png"
            require(reset_path.is_file(), f"Missing actual latent reset PNG: {name}/{key}")
            require(sha(reset_path) == sha(baseline_path), f"Reset differs from independent Z PNG: {name}/{key}")
            reset_checks.append({"run": name, "image_id": key, "reset_sha256": sha(reset_path),
                                 "zero_sha256": sha(baseline_path), "identical": True})
    for name, latent in (("H", "L"), ("H_EOT", "L_EOT")):
        for key in ids:
            stage_path = Path(info[name]["path"]) / f"{key}_lao_image.png"
            latent_path = Path(info[latent]["path"]) / f"{key}_adv_image.png"
            require(stage_path.is_file(), f"Missing actual hybrid latent-stage PNG: {name}/{key}")
            require(sha(stage_path) == sha(latent_path), f"Hybrid latent stage differs from independent {latent}: {key}")
            stage_checks.append({"run": name, "image_id": key, "latent_comparator": latent,
                                 "stage_sha256": sha(stage_path), "latent_sha256": sha(latent_path), "identical": True})
    return ids, info, quality_pixels, checks, reset_checks, stage_checks


def build_report(evaluation, method_paths, n):
    require(isinstance(n, int) and n > 0, "Expected n must be positive")
    evaluation = Path(evaluation).resolve()
    ids, method_info, pixel_quality, pixel_checks, reset_checks, stage_checks = inspect_methods(method_paths, n)
    index, clean_index, quality = {}, {}, {}
    rows = read_csv(evaluation / "per_image.csv")
    detectors = tuple(d for d in DETECTORS if any(row["detector"] == d for row in rows))
    require(set(row["detector"] for row in rows) == set(DETECTORS),
            "Unexpected/missing detector labels")
    require(set(row["condition"] for row in rows) == set(CONDITIONS),
            "Formal evaluation requires exactly none,jpeg75,jpeg85,jpeg95; training JPEG80/90 are excluded")
    for row in rows:
        name, det, cond, key = row["run"], row["detector"], row["condition"], row["image_id"]
        require(name in METHODS and key in ids, f"Unexpected CSV method/ID: {name}/{key}")
        item = (name, det, cond, key)
        require(item not in index, f"Duplicate CSV row: {item}")
        cp, ap = predicted(row["clean_pred"], item), predicted(row["adv_pred"], item)
        cm, am = number(row["clean_margin"], item), number(row["adv_margin"], item)
        # Exact ties use class 0 under argmax; never reconstruct predictions from rounded margins.
        require((cm > 0) == bool(cp) and (am > 0) == bool(ap), f"Prediction/margin disagree: {item}")
        clean_key = (det, cond, key)
        if clean_key in clean_index:
            require(cp == clean_index[clean_key][0], f"Clean prediction differs across methods: {clean_key}")
            close(cm, clean_index[clean_key][1], f"clean margin {clean_key}", 2e-5)
        else:
            clean_index[clean_key] = (cp, cm)
        index[item] = {"clean_pred": cp, "adv_pred": ap, "clean_margin": cm, "adv_margin": am}
        if cond == "none":
            lp = number(row["lpips"], f"LPIPS {item}")
            require(lp >= 0, f"Negative LPIPS: {item}")
            ps = number(row["psnr"], f"PSNR {item}", infinity=True)
            li = number(row["linf"], f"Linf {item}")
            actual = pixel_quality[name, key]
            close(ps, actual["psnr"], f"actual PNG PSNR {item}", 1e-4)
            close(li, actual["max_difference_pixels"] / 255., f"actual PNG Linf {item}")
            q = {**actual, "lpips": lp,
                 "qualified": lp <= LPIPS_MAX and actual["psnr"] >= PSNR_MIN,
                 "legacy_qualified": lp <= .15 and actual["psnr"] >= PSNR_MIN}
            if (name, key) in quality:
                close(lp, quality[name, key]["lpips"], f"LPIPS across detectors {name}/{key}")
            quality[name, key] = q
            record_lp = lookup_number(method_info[name]["records"][key],
                                      ("lpips", "selected.saved_png.lpips", "selected.lpips", "selected.metrics.lpips"))
            if record_lp is not None:
                close(lp, record_lp, f"record LPIPS {name}/{key}")
            if det == "E":
                record = method_info[name]["records"][key]
                recorded = record.get("final", {}).get("E")
                if recorded is not None:
                    require(predicted(recorded["pred"], item) == ap, f"Record versus evaluated saved-PNG class changed: {item}")
                    close(number(recorded["real_margin"], item), am, f"record E margin {item}", 2e-5)
    expected = {(name, det, cond, key) for name in METHODS for det in detectors for cond in CONDITIONS for key in ids}
    require(set(index) == expected, f"Incomplete CSV coverage: {len(index)} vs {len(expected)}")
    require(set(quality) == {(name, key) for name in METHODS for key in ids}, "Incomplete original-PNG quality scores")

    real_index, real_ids = {}, None
    for row in read_csv(evaluation / "real_predictions.csv"):
        det, cond, key = row["detector"], row["condition"], row["image"]
        require(det in detectors and cond in CONDITIONS, "Unexpected real prediction condition")
        item = (det, cond, key)
        require(item not in real_index, f"Duplicate real row: {item}")
        pr = predicted(row["pred"], item)
        margin = number(row["real_margin"], item)
        require((margin > 0) == bool(pr), f"Real prediction/margin disagree: {item}")
        real_index[item] = pr
    fpr = []
    for det in detectors:
        for cond in CONDITIONS:
            keys = sorted(key for d, c, key in real_index if (d, c) == (det, cond))
            require(len(keys) == n, f"Expected {n} real controls for {det}/{cond}")
            if real_ids is None:
                real_ids = keys
            require(keys == real_ids, "Real control identities differ across cells")
            false = sum(real_index[det, cond, key] == 0 for key in keys)
            fpr.append({"detector": det, "condition": cond, "n_real": len(keys), "false_positives": false,
                        "real_fpr": false / len(keys)})

    metrics, robustness, paired_rows, outcome_rows = [], [], [], []
    for name in METHODS:
        values = [quality[name, key] for key in ids]
        method_info[name]["quality"] = {
            "n": n, "qualified": sum(q["qualified"] for q in values),
            "legacy_qualified": sum(q["legacy_qualified"] for q in values),
            "lpips_mean": mean(q["lpips"] for q in values), "lpips_median": median(q["lpips"] for q in values),
            "lpips_max": max(q["lpips"] for q in values), "lpips_p95": percentile([q["lpips"] for q in values], .95),
            "psnr_mean": mean(q["psnr"] for q in values), "psnr_median": median(q["psnr"] for q in values),
            "maximum_pixel_difference": max(q["max_difference_pixels"] for q in values),
            "mean_lpips_guard_pass": mean(q["lpips"] for q in values) <= MEAN_LPIPS_MAX}
        for det in detectors:
            for cond in CONDITIONS:
                eligible = [key for key in ids if index[name, det, cond, key]["clean_pred"] == 0]
                success = [key for key in eligible if index[name, det, cond, key]["adv_pred"] == 1]
                good = [key for key in success if quality[name, key]["qualified"]]
                legacy = [key for key in success if quality[name, key]["legacy_qualified"]]
                metrics.append({"run": name, "detector": det, "condition": cond, "n": n,
                                "n_clean_detected": len(eligible), "success": len(success), "cond_asr": ratio(len(success), len(eligible)),
                                "quality_success": len(good), "quality_asr": ratio(len(good), len(eligible)),
                                "legacy_quality_success": len(legacy), "legacy_quality_asr": ratio(len(legacy), len(eligible)),
                                "uncond_escape": sum(index[name, det, cond, key]["adv_pred"] == 1 for key in ids) / n})
            joint_ids = [key for key in ids if all(index[name, det, cond, key]["clean_pred"] == 0 for cond in CONDITIONS)]
            outcomes = {key: all(index[name, det, cond, key]["adv_pred"] == 1 for cond in CONDITIONS) for key in ids}
            robust = [key for key in joint_ids if outcomes[key]]
            good = [key for key in robust if quality[name, key]["qualified"]]
            robustness.append({"run": name, "detector": det, "conditions": list(CONDITIONS), "n": n,
                               "n_joint_clean_detected": len(joint_ids), "joint_success": len(robust),
                               "joint_asr": ratio(len(robust), len(joint_ids)), "quality_joint_success": len(good),
                               "quality_joint_asr": ratio(len(good), len(joint_ids)),
                               "uncond_joint_escape": sum(outcomes.values()) / n,
                               "eligible_ids": joint_ids, "quality_success_ids": good})
            for key in ids:
                outcome_rows.append({"run": name, "detector": det, "image_id": key, "clean_joint_eligible": key in joint_ids,
                                     "all_conditions_real": outcomes[key], "original_quality_qualified": quality[name, key]["qualified"],
                                     "eligible_quality_success": key in good, **quality[name, key]})

    lookup = {(row["run"], row["detector"]): row for row in robustness}
    comparisons = [("H_EOT", "P_EOT", "primary"), ("L_EOT", "P_EOT", "secondary"),
                   ("H_EOT", "L_EOT", "secondary"), ("P_EOT", "P", "descriptive_eot"),
                   ("L_EOT", "L", "descriptive_eot"), ("H_EOT", "H", "descriptive_eot")]
    for left, right, role in comparisons:
        a, b = lookup[left, "E"], lookup[right, "E"]
        require(a["eligible_ids"] == b["eligible_ids"], "Joint clean cohort differs between methods")
        cohort = a["eligible_ids"]
        stats = paired([key in a["quality_success_ids"] for key in cohort],
                       [key in b["quality_success_ids"] for key in cohort], cohort)
        means_pass = all(method_info[name]["quality"]["mean_lpips_guard_pass"] for name in (left, right))
        superior = bool(stats["n_eligible"] and stats["ci95_lower"] > 0 and stats["exact_mcnemar_p"] < .05 and means_pass)
        paired_rows.append({"left": left, "right": right, "role": role, "detector": "E",
                            "endpoint": "quality-qualified success under every fixed evaluation condition",
                            "both_mean_lpips_guards_pass": means_pass, "superiority_criterion_met": superior,
                            **stats})

    summaries = read_csv(evaluation / "summary.csv")
    summary_index = {(r["run"], r["detector"], r["condition"]): r for r in summaries}
    require(len(summary_index) == len(summaries) == len(metrics), "Missing/duplicate summary cells")
    fpr_index = {(r["detector"], r["condition"]): r["real_fpr"] for r in fpr}
    for cell in metrics:
        key = cell["run"], cell["detector"], cell["condition"]
        require(key in summary_index, f"Missing summary cell: {key}")
        reported = summary_index[key]
        for field in ("n", "n_clean_detected", "uncond_escape"):
            close(number(reported[field], field), cell[field], f"summary {key}/{field}")
        asr = number(reported["cond_asr"], "cond_asr", optional=True)
        require((asr is None) == (cell["cond_asr"] is None), f"Summary ASR denominator mismatch: {key}")
        if asr is not None:
            close(asr, cell["cond_asr"], f"summary ASR {key}")
        close(number(reported["real_fpr"], "real_fpr"), fpr_index[key[1:]], f"summary FPR {key}")
        close(number(reported["lpips"], "summary LPIPS"), method_info[key[0]]["quality"]["lpips_mean"], f"summary LPIPS {key}")

    for value in method_info.values():
        del value["records"]
    return {"schema_version": 3, "status": "verified", "created_at": datetime.now().astimezone().isoformat(),
            "evaluation": str(evaluation), "n_generated": n, "n_real": n, "detectors": list(detectors),
            "conditions": list(CONDITIONS), "training_conditions": list(TRAIN_CONDITIONS),
            "budget_pixels": BUDGET, "quality_screen": {"psnr_min": PSNR_MIN, "lpips_max": LPIPS_MAX},
            "mean_lpips_max": MEAN_LPIPS_MAX, "methods": method_info, "metrics": metrics,
            "robustness": robustness, "paired_comparisons": paired_rows, "real_fpr": fpr,
            "primary": paired_rows[0], "per_image_joint_outcomes": outcome_rows,
            "integrity": {"pixel_checks": len(pixel_checks), "pixel_budget_violations": 0,
                          "canonical_source_consistency_checks": n * len(METHODS), "csv_rows": len(index),
                          "summary_cells": len(metrics), "real_csv_rows": len(real_index),
                          "independent_reset_checks": len(reset_checks), "reset_evidence": reset_checks,
                          "hybrid_latent_stage_checks": len(stage_checks), "hybrid_latent_stage_evidence": stage_checks,
                          "pixel_evidence": pixel_checks, "input_sha256": {name: sha(evaluation / name)
                              for name in ("per_image.csv", "real_predictions.csv", "summary.csv")}},
            "limitations": [
                "Predictions and LPIPS are read from independent saved-PNG evaluation; this report never loads a model.",
                "PNG budgets and PSNR are recomputed; LPIPS is cross-checked with records when available, not independently re-executed.",
                "Only the fixed E joint-condition H_EOT versus P_EOT comparison is primary; secondary/descriptive tests are not multiplicity-adjusted.",
                "Mean LPIPS <= .015 is a method-level guard, not a reason to remove any individual sample.",
                "Every transform uses original-PNG quality; the integer budget applies before JPEG, not to compressed pixels.",
                "FPR must be read beside ASR. Nonzero robust success alone does not establish improvement or a useful detector operating point.",
                "Joint robustness concerns only four fixed JPEG/identity conditions, not arbitrary transformations.",
                "This audit checks source consistency between outputs, not unseen-data provenance, source-code freezing, or human semantic fidelity.",
                "Standalone L/P/H comparisons do not establish adaptive causal stage dominance without matched renderer controls and reoptimization.",
            ]}


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(item) for item in value]
    return value


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def markdown(report):
    primary = report["primary"]
    pct = lambda x: "unavailable" if x is None else f"{100 * x:.2f}%"
    text = ["# LAO / PGD / JPEG validation", "", f"All {report['n_generated']} generated samples are retained. "
            "Budget: saved RGB uint8 maximum difference ≤8; quality: PSNR≥30 and LPIPS≤0.03; mean LPIPS guard≤0.015.", "",
            "Primary E cohort: clean detected under every condition (none, JPEG75, JPEG85, JPEG95). "
            "Success requires the same output to be classified real under all four conditions and pass original-PNG quality.", "",
            "| Method | Joint quality success / eligible | Mean LPIPS | Mean guard |", "|---|---:|---:|---|" ]
    for name in METHODS:
        robust = next(r for r in report["robustness"] if r["run"] == name and r["detector"] == "E")
        q = report["methods"][name]["quality"]
        text.append(f"| {name} | {robust['quality_joint_success']}/{robust['n_joint_clean_detected']} | {q['lpips_mean']:.6f} | {q['mean_lpips_guard_pass']} |")
    text += ["", f"Primary H_EOT − P_EOT: {pct(primary['paired_difference'])}; paired bootstrap 95% CI "
             f"[{pct(primary['ci95_lower'])}, {pct(primary['ci95_upper'])}]; exact McNemar p={primary['exact_mcnemar_p']}. "
             f"Added/lost successes: {primary['left_only']}/{primary['right_only']}. "
             f"Predeclared superiority criterion met: {primary['superiority_criterion_met']}.", "",
             "| Detector | Condition | Real false positives / controls | FPR |", "|---|---|---:|---:|"]
    for row in report["real_fpr"]:
        text.append(f"| {row['detector']} | {row['condition']} | {row['false_positives']}/{row['n_real']} | {pct(row['real_fpr'])} |")
    text += ["", "Interpretation limits:", ""] + ["- " + item for item in report["limitations"]]
    return "\n".join(text) + "\n"


def write_report(report, output):
    output = Path(output).resolve()
    require(not output.exists(), f"Refusing to overwrite report: {output}")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(safe_json(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    for filename, key in (("metrics.csv", "metrics"), ("robustness.csv", "robustness"),
                          ("paired_comparisons.csv", "paired_comparisons"), ("real_fpr.csv", "real_fpr"),
                          ("per_image_joint_outcomes.csv", "per_image_joint_outcomes")):
        write_csv(output / filename, report[key])
    (output / "report.md").write_text(markdown(report), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--methods", required=True, help="JSON mapping or mapping file: Z,P,L,H,P_EOT,L_EOT,H_EOT to method directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), f"Refusing to overwrite report: {args.out}")
    mapping = json.loads(args.methods) if args.methods.lstrip().startswith("{") else read_json(args.methods)
    require(isinstance(mapping, dict), "--methods must be a JSON object")
    report = build_report(args.evaluation, mapping, args.n)
    write_report(report, args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out.resolve()), "primary": report["primary"]}))


if __name__ == "__main__":
    main()
