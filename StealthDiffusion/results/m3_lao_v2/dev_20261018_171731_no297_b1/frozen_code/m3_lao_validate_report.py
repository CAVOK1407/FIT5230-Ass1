#!/usr/bin/env python3
"""Read-only audit of the frozen LAO / strict saved-PNG-budget validation.

No model is imported or executed. Pillow is used only to read and compare PNG
pixels. Writes a new report directory and refuses to overwrite existing output.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageChops


METHODS = ("RAW_LAO", "RAW_RECON", "LAO8", "RECON8", "PGD10", "PGD40")
BOUNDED = frozenset(("LAO8", "RECON8", "PGD10", "PGD40"))
DETECTORS = ("E", "R", "D", "S")
CONDITIONS = ("none", "jpeg75", "resize05")
EXPECTED_N = 50
BUDGET_PIXELS = 8
PSNR_MIN = 30.0
LPIPS_MAX = 0.15
BOOTSTRAP_SEED = 20261018
BOOTSTRAP_RESAMPLES = 10000
NONINFERIORITY_MARGIN = 0.05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"Empty CSV: {path}")
    return rows


def numeric(value, label, optional=False, infinity=False):
    if value is None or str(value).strip().lower() in ("", "none", "null", "nan"):
        require(optional, f"Missing numeric value: {label}")
        return None
    result = float(value)
    require(not math.isnan(result) and (infinity or math.isfinite(result)), f"Non-finite value: {label}")
    return result


def pred(value, label):
    result = numeric(value, label)
    require(result in (0, 1), f"Expected 0=AI or 1=real: {label}")
    return int(result)


def close(left, right, label):
    require(math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6),
            f"Inconsistent {label}: {left} vs {right}")


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def avg(values):
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def image_rgb(path):
    with Image.open(path) as source:
        require(source.format == "PNG" and source.mode == "RGB", f"Expected RGB PNG: {path}")
        require(source.size == (224, 224), f"Expected canonical 224x224 image: {path}")
        return source.copy()


def pixel_metrics(original, output):
    difference = ImageChops.difference(original, output)
    max_delta = max(high for _, high in difference.getextrema())
    histogram = difference.histogram()
    mse = sum((index % 256) ** 2 * count for index, count in enumerate(histogram)) / (3 * original.width * original.height)
    psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    return int(max_delta), psnr


def projected_pixels(raw, original):
    lower = original.point([max(0, value - BUDGET_PIXELS) for value in range(256)] * 3)
    upper = original.point([min(255, value + BUDGET_PIXELS) for value in range(256)] * 3)
    return ImageChops.darker(ImageChops.lighter(raw, lower), upper)


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def exact_mcnemar(only_left, only_right):
    """Two-sided exact conditional binomial test on discordant pairs."""
    discordant = only_left + only_right
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(only_left, only_right) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def binomial_cdf(k, n, probability):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.comb(n, index) * probability ** index * (1 - probability) ** (n - index)
                        for index in range(k + 1)))


def clopper_pearson(successes, trials, alpha=0.025):
    """Two-sided exact binomial interval; default confidence is 97.5%."""
    require(0 <= successes <= trials and trials > 0, "Invalid binomial counts")
    tail = alpha / 2
    lower = 0.0
    if successes:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if 1 - binomial_cdf(successes - 1, trials, mid) < tail:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    upper = 1.0
    if successes < trials:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_cdf(successes, trials, mid) > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def conservative_difference_interval(only_left, only_right, trials):
    """Bonferroni combination: simultaneous coverage at least 95%."""
    if not trials:
        return {"conservative_ci95_lower": None, "conservative_ci95_upper": None,
                "left_only_cp975": None, "right_only_cp975": None}
    left_interval = clopper_pearson(only_left, trials)
    right_interval = clopper_pearson(only_right, trials)
    return {"conservative_ci95_lower": left_interval[0] - right_interval[1],
            "conservative_ci95_upper": left_interval[1] - right_interval[0],
            "left_only_cp975": list(left_interval), "right_only_cp975": list(right_interval)}


def paired_statistics(left, right, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    require(len(left) == len(right), "Paired outcomes must have the same length")
    differences = [int(a) - int(b) for a, b in zip(left, right)]
    only_left = sum(a and not b for a, b in zip(left, right))
    only_right = sum(b and not a for a, b in zip(left, right))
    common = sum(a and b for a, b in zip(left, right))
    if not differences:
        return {"n_eligible": 0, "left_only": 0, "right_only": 0, "both_success": 0, "both_failure": 0,
                "paired_difference": None, "ci95_lower": None, "ci95_upper": None, "exact_mcnemar_p": None,
                **conservative_difference_interval(0, 0, 0)}
    generator = random.Random(seed)
    n = len(differences)
    replicates = sorted(sum(differences[generator.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {"n_eligible": n, "left_only": only_left, "right_only": only_right, "both_success": common,
            "both_failure": n - only_left - only_right - common,
            "paired_difference": sum(differences) / n,
            "ci95_lower": percentile(replicates, 0.025), "ci95_upper": percentile(replicates, 0.975),
            "exact_mcnemar_p": exact_mcnemar(only_left, only_right),
            **conservative_difference_interval(only_left, only_right, n)}


def validate_plan(plan):
    for key, expected in (("n_generated", EXPECTED_N), ("n_real", EXPECTED_N), ("budget_pixels", BUDGET_PIXELS)):
        require(plan.get(key) == expected, f"Unexpected frozen plan {key}: expected {expected}")
    require(plan.get("quality_screen") == {"psnr_min": PSNR_MIN, "lpips_max": LPIPS_MAX}, "Unexpected frozen quality screen")
    primary = plan.get("primary", {})
    expected = {"run": "LAO8", "comparator": "PGD40", "detector": "E", "condition": "none",
                "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "noninferiority_margin": NONINFERIORITY_MARGIN}
    for key, value in expected.items():
        require(primary.get(key) == value, f"Unexpected frozen primary setting: {key}")
    require(isinstance(plan.get("methods"), dict) and set(METHODS) <= set(plan["methods"]), "Frozen plan must describe all six methods")
    require(isinstance(plan.get("source_sha256"), dict) and bool(plan["source_sha256"]), "Missing frozen source hashes")


def source_pixels(path, center_crop=False):
    with Image.open(path) as source:
        image = source.convert("RGB")
        if center_crop:
            width, height = image.size
            side = min(width, height)
            left, top = (width - side) // 2, (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
        return image.resize((224, 224), Image.Resampling.LANCZOS)


def verify_data_provenance(root, manifest):
    """Bind novelty claims to the fixed generation plan and actual canonical pixels."""
    data_root = (root / "data").resolve()
    plan_path = data_root / "generation_plan.json"
    generation = read_json(plan_path)
    require(manifest.get("status") == "verified", "Data manifest is not verified")
    require(manifest.get("n_generated") == manifest.get("n_real") == EXPECTED_N, "Data manifest counts differ from 50+50")
    require(Path(manifest.get("data_root", "")).resolve() == data_root, "Data manifest points outside this validation root")
    require(manifest.get("plan_sha256") == digest(plan_path), "Generation plan hash differs from the data manifest")
    require(read_json(data_root / "data_manifest.json") == manifest, "Root and data-folder manifests differ")
    require(generation.get("n_generated") == generation.get("n_real") == EXPECTED_N, "Generation plan counts differ from 50+50")
    require(generation.get("replacement_rule") == "Abort on failure or duplicate; never replace samples", "Unexpected generation replacement rule")
    require(generation.get("method") == "fresh_same_prompt_family_sd15_seeds_without_outcome_selection", "Unexpected generation selection method")
    model = generation["model"]
    require(model.get("id") == "stable-diffusion-v1-5/stable-diffusion-v1-5", "Unexpected held-out generator family")
    novelty = manifest.get("novelty", {})
    for key in ("old_overlap", "new_internal_duplicates", "source_seed_overlap", "replacement_count"):
        require(novelty.get(key) == 0, f"Data novelty check failed: {key}")
    require(novelty.get("detector_scores_used_for_selection") is False, "Outcome-free data selection not declared")
    require(novelty.get("new_unique_rgb224") == 100, "Novelty manifest does not declare 100 unique canonical images")
    state = read_json(data_root / "generation_state.json")
    require(state.get("status") == "verified" and state.get("completed_images") == EXPECTED_N, "Generation did not complete all 50 planned images")
    require(state.get("plan_sha256") == digest(plan_path), "Generation-state plan hash mismatch")
    require(state.get("data_manifest_sha256") == digest(data_root / "data_manifest.json"), "Generation-state manifest hash mismatch")
    old = generation["old_inputs"]
    old_hashes = set(old["old_rgb224_sha256"])
    require(bool(old_hashes), "Old-input exclusion set is empty")
    require(novelty.get("old_canonical_unique_rgb224") == len(old_hashes), "Old-input exclusion count mismatch")
    require(old.get("additional_result_origin_contents") == 0, "Old experiment sources extend beyond the exclusion set")
    actual_old_hashes, actual_old_real_ids, actual_old_generation_ids = set(), set(), set()
    for row in old["canonical_images"]:
        source = Path(row["path"])
        require(digest(source) == row["sha256"], f"An old canonical input changed: {source}")
        pixel_hash = hashlib.sha256(source_pixels(source).tobytes()).hexdigest()
        require(pixel_hash == row["rgb224_sha256"], f"Old canonical pixel hash mismatch: {source}")
        actual_old_hashes.add(pixel_hash)
        if row["kind"] == "gen":
            actual_old_generation_ids.add((row["source"]["model"], int(row["source"]["seed"])))
        elif row["kind"] == "real":
            actual_old_real_ids.add(Path(row["source"]["source"]).name)
        else:
            raise ValueError("Unknown old-input kind")
    require(actual_old_hashes == old_hashes, "Old canonical images do not match the declared exclusion hash set")
    require(actual_old_real_ids == set(old["old_real_source_ids"]), "Old real-source ID exclusion mismatch")
    require(actual_old_generation_ids == {tuple(item) for item in old["old_generation_ids"]}, "Old generation-source ID exclusion mismatch")
    for filename, expected in old["manifest_sha256"].items():
        require(digest(Path(filename)) == expected, f"Old split manifest changed: {filename}")
    all_pixel_hashes, canonical_checks = set(), []
    expected_ids = {f"{index:04d}" for index in range(EXPECTED_N)}
    for kind, plan_key, directory, csv_hash_key in (
        ("generated", "generated_samples", "gen/test", "generated_manifest_sha256"),
        ("real", "real_samples", "real/test", "real_manifest_sha256"),
    ):
        rows = manifest.get(kind, [])
        planned_rows = generation.get(plan_key, [])
        require(len(rows) == len(planned_rows) == EXPECTED_N, f"Incomplete {kind} provenance")
        indexed = {str(row["image_id"]): row for row in rows}
        planned = {str(row["image_id"]): row for row in planned_rows}
        require(set(indexed) == set(planned) == expected_ids, f"Duplicate/missing {kind} provenance IDs")
        csv_path = data_root / directory / "manifest.csv"
        require(digest(csv_path) == manifest.get(csv_hash_key), f"{kind} source CSV hash mismatch")
        csv_rows = read_csv(csv_path)
        csv_index = {row["image_id"]: row for row in csv_rows}
        require(len(csv_rows) == EXPECTED_N and set(csv_index) == expected_ids, f"{kind} source CSV coverage mismatch")
        for image_id in sorted(expected_ids):
            row, sample = indexed[image_id], planned[image_id]
            require(all(str(row.get(key)) == str(value) for key, value in sample.items()), f"{kind}/{image_id} differs from its frozen source choice")
            require(all(str(row.get(key)) == value for key, value in csv_index[image_id].items()), f"{kind}/{image_id} differs from its source CSV")
            path = (data_root / row["relative_path"]).resolve()
            expected_path = (data_root / directory / f"{image_id}.png").resolve()
            require(path.is_relative_to(data_root) and path == expected_path, f"Noncanonical/out-of-root data path: {kind}/{image_id}")
            require(digest(path) == row["sha256"], f"Canonical file hash mismatch: {kind}/{image_id}")
            image = image_rgb(path)
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            require(pixel_hash == row["rgb224_sha256"], f"Canonical RGB224 hash mismatch: {kind}/{image_id}")
            require(pixel_hash not in old_hashes and pixel_hash not in all_pixel_hashes, f"Old overlap or new duplicate: {kind}/{image_id}")
            all_pixel_hashes.add(pixel_hash)
            if kind == "generated":
                require(int(row["seed"]) == 5231000 + int(image_id), f"Unplanned new generation seed: {image_id}")
                require(row.get("model") == model["id"] and row.get("model_revision") == model["revision"], "Generated model provenance mismatch")
                require((row["model"], int(row["seed"])) not in actual_old_generation_ids, "Reused generation source ID")
                require(row.get("steps") == 30 and row.get("guidance") == 7.5, "Changed generation parameters")
                source = (data_root / row["source512"]).resolve()
                require(source.is_relative_to(data_root) and source == (data_root / "gen512" / f"{image_id}.png").resolve(), "Unexpected full-resolution source path")
                require(digest(source) == row["source512_sha256"], f"Full-resolution source hash mismatch: {image_id}")
                with Image.open(source) as full:
                    require(full.format == "PNG" and full.mode == "RGB" and full.size == (512, 512), "Invalid generated source image")
                require(source_pixels(source).tobytes() == image.tobytes(), f"Canonical resize differs from generated source: {image_id}")
            else:
                source = Path(row["source"])
                require(row["source_id"] == source.name and row["source_id"] not in actual_old_real_ids, "Reused/mismatched real source ID")
                require(digest(source) == row["source_sha256"], f"Real source file hash mismatch: {image_id}")
                require(source_pixels(source, center_crop=True).tobytes() == image.tobytes(), f"Real canonical crop differs from frozen source: {image_id}")
            canonical_checks.append({"kind": kind, "image_id": image_id, "relative_path": row["relative_path"],
                                     "sha256": row["sha256"], "rgb224_sha256": pixel_hash})
    generation_records = [json.loads(line) for line in (data_root / "generation_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(generation_records == manifest["generated"], "Generation records differ from the final manifest")
    require(len(all_pixel_hashes) == 100, "Canonical image uniqueness check failed")
    return {"verified": True, "n_generated": 50, "n_real": 50, "new_unique_rgb224": 100,
            "old_overlap": 0, "old_unique_rgb224": len(old_hashes), "plan_sha256": digest(plan_path),
            "canonical_checks": canonical_checks}


def read_records(folder, originals):
    rows = {}
    for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        image_id = str(record["image_id"])
        require(image_id not in rows, f"Duplicate record ID in {folder}: {image_id}")
        require(image_id in originals, f"Unknown image ID in {folder}: {image_id}")
        require(record.get("source") == originals[image_id].name, f"Source mapping mismatch: {folder.name}/{image_id}")
        rows[image_id] = record
    require(set(rows) == set(originals), f"Missing/extra records in {folder}")
    return rows


def read_artifacts(root):
    originals = {path.stem: path for path in sorted((root / "data/gen/test").glob("*.png"))}
    real_paths = sorted((root / "data/real/test").glob("*.png"))
    require(len(originals) == EXPECTED_N and len(real_paths) == EXPECTED_N, "Expected exactly 50 generated and 50 real canonical PNGs")
    original_hashes = {image_id: digest(path) for image_id, path in originals.items()}
    require(len(set(original_hashes.values())) == EXPECTED_N, "Duplicate generated PNG files")
    real_hashes = {path.name: digest(path) for path in real_paths}
    require(len(set(real_hashes.values())) == EXPECTED_N, "Duplicate real PNG files")
    original_images = {image_id: image_rgb(path) for image_id, path in originals.items()}
    require(len({hashlib.sha256(image.tobytes()).hexdigest() for image in original_images.values()}) == EXPECTED_N,
            "Duplicate canonical generated pixel arrays")
    for path in real_paths:
        image_rgb(path)
    configurations, records, artifacts, hashes = {}, {}, [], {}
    output_images = {}
    for name in METHODS:
        folder = root / name
        configurations[name] = read_json(folder / "run_config.json")
        records[name] = read_records(folder, originals)
        for filename in ("run_config.json", "records.jsonl"):
            hashes[str((folder / filename).relative_to(root))] = digest(folder / filename)
        actual_ids = {path.name.removesuffix("_adv_image.png") for path in folder.glob("*_adv_image.png")}
        require(actual_ids == set(originals), f"Output PNG coverage mismatch: {name}")
        for image_id, canonical in originals.items():
            origin_path = folder / f"{image_id}_originImage.png"
            require(digest(origin_path) == original_hashes[image_id], f"Origin PNG is not canonical bytes: {name}/{image_id}")
            output_path = folder / f"{image_id}_adv_image.png"
            output = image_rgb(output_path)
            output_images[(name, image_id)] = output
            maximum, psnr = pixel_metrics(original_images[image_id], output)
            artifacts.append({"run": name, "image_id": image_id, "source": canonical.name,
                              "diagnostic_only": name not in BOUNDED, "max_abs_delta_uint8": maximum,
                              "linf": maximum / 255, "psnr_pixels": psnr,
                              "budget_applicable": name in BOUNDED,
                              "budget_valid": maximum <= BUDGET_PIXELS if name in BOUNDED else None,
                              "origin_sha256": original_hashes[image_id], "output_sha256": digest(output_path)})
    for name, iterations in (("RAW_LAO", 10), ("RAW_RECON", 0)):
        config = configurations[name]
        require(config.get("lao_residual_anchor") is True and config.get("control_scale") == 0.5,
                f"{name} is not the frozen half-gate residual-anchor architecture")
        require(config.get("encode_source") == config.get("control_source") == "clean", f"Pixel-PGD input path present in {name}")
        require(config.get("iterations") == iterations, f"Unexpected iterations in {name}")
        require(config.get("surrogates") == "E", f"Unexpected optimisation target in {name}")
        for image_id, record in records[name].items():
            require(record.get("pgd") is None and not record.get("pgd_info"), f"Pixel PGD contaminated {name}/{image_id}")
            require(len(record.get("history", [])) == iterations, f"Incomplete LAO history: {name}/{image_id}")
    baseline, candidate = configurations["RAW_RECON"], configurations["RAW_LAO"]
    for key in ("seed", "res", "diffusion_steps", "start_step", "guidance", "pretrained_diffusion_path", "encoder_weights",
                "decode_path", "loss", "kappa", "kappa_resolved", "lpips_weight", "lr", "lao_weight_decay", "lao_eot", "di_prob"):
        require(baseline.get(key) == candidate.get(key), f"Raw reconstruction/LAO configuration mismatch: {key}")
    resets = []
    for image_id in originals:
        for name, baseline_name in (("RAW_LAO", "RAW_RECON"), ("LAO8", "RECON8")):
            reset_path = root / name / f"{image_id}_reset_image.png"
            baseline_path = root / baseline_name / f"{image_id}_adv_image.png"
            reset_hash, baseline_hash = digest(reset_path), digest(baseline_path)
            require(reset_hash == baseline_hash, f"Reset output differs from independent baseline: {name}/{image_id}")
            resets.append({"run": name, "baseline": baseline_name, "image_id": image_id,
                           "identical": True, "reset_sha256": reset_hash, "baseline_sha256": baseline_hash})
        for name, raw in (("LAO8", "RAW_LAO"), ("RECON8", "RAW_RECON")):
            expected = projected_pixels(output_images[(raw, image_id)], original_images[image_id])
            require(expected.tobytes() == output_images[(name, image_id)].tobytes(),
                    f"{name}/{image_id} is not the exact export-only integer projection of {raw}")
    return originals, real_hashes, configurations, records, artifacts, resets, hashes


def build_report(root):
    plan_path, manifest_path = root / "plan.json", root / "data_manifest.json"
    plan, manifest = read_json(plan_path), read_json(manifest_path)
    validate_plan(plan)
    provenance = verify_data_provenance(root, manifest)
    originals, real_hashes, configurations, records, artifacts, resets, hashes = read_artifacts(root)
    image_ids = sorted(originals)
    artifact_index = {(row["run"], row["image_id"]): row for row in artifacts}
    paths = {"per_image": root / "eval/per_image.csv", "real_predictions": root / "eval/real_predictions.csv"}
    cells = defaultdict(dict)
    for row in read_csv(paths["per_image"]):
        require(row["run"] in METHODS, f"Unexpected method in evaluation: {row['run']}")
        require(row["detector"] in DETECTORS and row["condition"] in CONDITIONS, "Unexpected detector or condition")
        key = (row["run"], row["detector"], row["condition"])
        image_id = row["image_id"]
        require(image_id in originals and image_id not in cells[key], f"Unknown/duplicate evaluation row: {key}/{image_id}")
        for field in ("clean_pred", "adv_pred"):
            row[field] = pred(row[field], f"{key}/{image_id}/{field}")
        for field in ("clean_margin", "adv_margin"):
            row[field] = numeric(row[field], field)
        for field in ("psnr", "lpips", "ssim", "linf", "seconds"):
            row[field] = numeric(row.get(field), field, optional=True, infinity=field == "psnr")
        cells[key][image_id] = row
    expected_cells = {(name, detector, condition) for name in METHODS for detector in DETECTORS for condition in CONDITIONS}
    require(set(cells) == expected_cells, "Evaluation must cover all six methods, four detectors and three conditions")
    for (name, detector, condition), rows in cells.items():
        require(set(rows) == set(originals), f"Evaluation does not retain all 50 images: {name}/{detector}/{condition}")
        for image_id, row in rows.items():
            reference = cells[("RAW_RECON", detector, condition)][image_id]
            require(row["clean_pred"] == reference["clean_pred"], f"Clean eligibility differs at {name}/{detector}/{condition}/{image_id}")
            close(row["clean_margin"], reference["clean_margin"], "clean margin")
    quality = {}
    for name in METHODS:
        for image_id in image_ids:
            row = cells[(name, "E", "none")][image_id]
            require(row["psnr"] is not None and row["lpips"] is not None, f"Missing primary quality metrics: {name}/{image_id}")
            require(row["lpips"] >= 0, f"Negative LPIPS: {name}/{image_id}")
            close(row["psnr"], artifact_index[(name, image_id)]["psnr_pixels"], "saved-pixel PSNR")
            close(row["linf"], artifact_index[(name, image_id)]["linf"], "saved-pixel L-infinity")
            metric = {field: row[field] for field in ("psnr", "lpips", "ssim", "linf", "seconds")}
            for detector in DETECTORS:
                peer = cells[(name, detector, "none")][image_id]
                require(all(peer[field] == metric[field] for field in metric), f"Quality/time differs across detectors: {name}/{image_id}")
            metric["qualified"] = metric["psnr"] >= PSNR_MIN and metric["lpips"] <= LPIPS_MAX
            quality[(name, image_id)] = metric
    real = defaultdict(dict)
    for row in read_csv(paths["real_predictions"]):
        key = (row["detector"], row["condition"])
        require(key in {(detector, condition) for detector in DETECTORS for condition in CONDITIONS}, "Unexpected real-image evaluation cell")
        filename = Path(row["image"]).name
        require(filename in real_hashes and filename not in real[key], f"Unknown/duplicate real-image row: {key}/{filename}")
        real[key][filename] = pred(row["pred"], "real prediction")
    require(set(real) == {(detector, condition) for detector in DETECTORS for condition in CONDITIONS}, "Missing real-image cells")
    require(all(set(rows) == set(real_hashes) for rows in real.values()), "Each real-image cell must contain all 50 canonical images")

    def valid(name, image_id):
        return name not in BOUNDED or artifact_index[(name, image_id)]["budget_valid"]

    def qualified_success(name, detector, condition, image_id):
        return bool(valid(name, image_id) and quality[(name, image_id)]["qualified"] and
                    cells[(name, detector, condition)][image_id]["adv_pred"] == 1)

    metrics = []
    for name in METHODS:
        for detector in DETECTORS:
            for condition in CONDITIONS:
                rows = cells[(name, detector, condition)]
                eligible = [image_id for image_id in image_ids if rows[image_id]["clean_pred"] == 0]
                classified_success = [image_id for image_id in eligible if rows[image_id]["adv_pred"] == 1]
                valid_success = [image_id for image_id in classified_success if valid(name, image_id)]
                quality_success = [image_id for image_id in eligible if qualified_success(name, detector, condition, image_id)]
                metric = {"run": name, "diagnostic_only": name not in BOUNDED, "detector": detector, "condition": condition,
                          "n": EXPECTED_N, "n_clean_detected": len(eligible), "n_classifier_success": len(classified_success),
                          "n_valid_success": len(valid_success), "n_quality_qualified_success": len(quality_success),
                          "classifier_flip_asr": fraction(len(classified_success), len(eligible)),
                          "cond_asr": fraction(len(valid_success), len(eligible)),
                          "quality_qualified_asr": fraction(len(quality_success), len(eligible)),
                          "uncond_escape": sum(row["adv_pred"] == 1 for row in rows.values()) / EXPECTED_N,
                          "uncond_valid_escape": sum(valid(name, image_id) and rows[image_id]["adv_pred"] == 1 for image_id in image_ids) / EXPECTED_N,
                          "real_n": EXPECTED_N, "real_fpr": sum(value == 0 for value in real[(detector, condition)].values()) / EXPECTED_N,
                          "n_budget_violations": sum(not valid(name, image_id) for image_id in image_ids),
                          "n_quality_qualified": sum(quality[(name, image_id)]["qualified"] for image_id in image_ids)}
                metric.update({field: avg(quality[(name, image_id)][field] for image_id in image_ids)
                               for field in ("psnr", "lpips", "ssim", "linf", "seconds")})
                metrics.append(metric)

    comparisons = []
    for comparator in ("PGD40", "PGD10", "RECON8"):
        for detector in DETECTORS:
            for condition in CONDITIONS:
                eligible = [image_id for image_id in image_ids if cells[("LAO8", detector, condition)][image_id]["clean_pred"] == 0]
                left = [qualified_success("LAO8", detector, condition, image_id) for image_id in eligible]
                right = [qualified_success(comparator, detector, condition, image_id) for image_id in eligible]
                added = [image_id for image_id, a, b in zip(eligible, left, right) if a and not b]
                lost = [image_id for image_id, a, b in zip(eligible, left, right) if b and not a]
                row = {"run": "LAO8", "comparator": comparator, "detector": detector, "condition": condition,
                       "primary": comparator == "PGD40" and detector == "E" and condition == "none",
                       "n": EXPECTED_N, "n_clean_detected": len(eligible),
                       "left_success": sum(left), "right_success": sum(right),
                       "added_success": len(added), "lost_success": len(lost), "net_success": len(added) - len(lost),
                       "difference": fraction(len(added) - len(lost), len(eligible)),
                       "added_ids": added, "lost_ids": lost,
                       "ci95_lower": None, "ci95_upper": None, "exact_mcnemar_p": None,
                       "conservative_ci95_lower": None, "conservative_ci95_upper": None}
                if detector == "E" and condition == "none":
                    row.update(paired_statistics(left, right))
                comparisons.append(row)
    primary = dict(next(row for row in comparisons if row["primary"]))
    budget_violations = [row for row in artifacts if row["budget_applicable"] and not row["budget_valid"]]
    lower, p_value = primary["ci95_lower"], primary["exact_mcnemar_p"]
    primary["superiority_statistical_criterion"] = lower is not None and lower > 0 and p_value < 0.05
    conservative_lower = primary["conservative_ci95_lower"]
    primary["noninferiority_descriptive_criterion"] = conservative_lower is not None and conservative_lower > -NONINFERIORITY_MARGIN
    primary["all_bounded_artifacts_valid"] = not budget_violations
    primary["superiority_supported"] = primary["superiority_statistical_criterion"] and not budget_violations
    primary["noninferiority_descriptively_supported"] = primary["noninferiority_descriptive_criterion"] and not budget_violations
    primary["noninferiority_margin"] = NONINFERIORITY_MARGIN
    primary["noninferiority_interval"] = "Bonferroni difference of two 97.5% Clopper-Pearson intervals; coverage at least 95%"
    primary["bootstrap"] = {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
                            "unit": "paired source image within the common clean-detected population",
                            "interval": "two-sided 95% percentile, linear interpolation", "rng": "Python random.Random"}
    diagnostic_rows = []
    for image_id in image_ids:
        record = records["RAW_LAO"][image_id]
        history = record["history"]
        gradients = [numeric(step["lao_diagnostics"]["gradient"]["l2"], "gradient L2") for step in history]
        updates = [numeric(step["lao_diagnostics"]["update"]["l2"], "latent update L2") for step in history]
        diagnostic_rows.append({"image_id": image_id, "steps": len(history),
                                "finite_gradient_steps": len(gradients), "nonzero_gradient_steps": sum(value > 0 for value in gradients),
                                "nonzero_update_steps": sum(value > 0 for value in updates),
                                "first_gradient_l2": gradients[0], "last_gradient_l2": gradients[-1],
                                "diagnostics": record.get("lao_diagnostics", {})})
    hashes.update({str(path.relative_to(root)): digest(path) for path in [plan_path, manifest_path, *paths.values()]})
    notes = [
        "New-image validation of a frozen residual-rendering architecture. All 100 canonical file/pixel hashes, planned source choices, generation plan hash, internal uniqueness and exclusion against old canonical inputs were verified; no image may be selected by clean or attack scores.",
        "RAW_LAO and RAW_RECON are diagnostic-only and are not admissible competitors in the 8/255 budget comparison.",
        "LAO8 is exactly the integer-budget projection of the frozen RAW_LAO PNG. There is no budget-aware latent re-optimisation in this round; projection is not inside the LAO objective.",
        "PGD10 uses quantised-save-forward margin(kappa=3)+LPIPS; PGD40 is the stronger CE reference with three starts and E-only iterate/restart selection. These are deliberately not identical optimisation objectives or equal compute budgets.",
        "R/D/S and transformed-image results are descriptive. They must not be used for selecting methods, iterations, restarts, thresholds or output files.",
        "Success rates retain every clean-detected source image. A budget violation counts as failure, never exclusion; classifier_flip_asr and uncond_escape additionally expose the raw classifier decisions.",
        "The fixed quality screen is PSNR >= 30 dB and LPIPS <= 0.15 on the unprocessed saved PNG. It is not a semantic or human-review verdict; all 50 images remain in every full-sample table.",
        "Budget is measured before JPEG/resize. For secondary transforms the reported quality screen still describes the original PNG, and real-image FPR is reported under that same transform.",
        "Primary outcome is quality-qualified LAO8 minus PGD40 on E/none, with a common clean-detected denominator, paired bootstrap CI and two-sided exact McNemar test.",
        "Superiority requires the primary bootstrap 95% CI lower bound > 0 and exact p < 0.05, with no invalid bounded artifact. Fixed-margin noninferiority is descriptive only and uses the conservative exact lower bound > -0.05, not the bootstrap bound.",
        "The conservative paired-risk-difference interval is [lo(LAO-only)-hi(PGD-only), hi(LAO-only)-lo(PGD-only)], using two separate 97.5% Clopper-Pearson intervals and Bonferroni coverage of at least 95%. It remains nondegenerate when all outcomes agree; bootstrap [0,0] alone cannot establish 5-point noninferiority. Passing noninferiority alone does not establish usefulness when both methods fail.",
        "Secondary tests and descriptive transfer cells are not multiplicity-adjusted confirmatory claims. Detector cells on the same images are correlated and must not be pooled as independent observations.",
        "The 50 images use the same ten prompt categories with new seeds; these intervals do not establish generalisation to unseen prompts, generator families or detector families.",
        "Zero-step and reset controls establish latent contribution within this residual architecture. Removing PGD does not establish LAO dominance in the original hybrid pipeline.",
        "Reported per-image times follow each run's records and can include diagnostic work. Ten LAO updates and ten PGD updates are not equal-cost computations.",
        "This reporter reads saved files only; it executes no model, training or attack and does not replace the independent source/data provenance checks.",
    ]
    return {"schema_version": 1, "status": "verified" if not budget_violations else "budget_violations_present",
            "n_generated": EXPECTED_N, "n_real": EXPECTED_N, "budget_pixels": BUDGET_PIXELS,
            "quality_screen": {"psnr_min": PSNR_MIN, "lpips_max": LPIPS_MAX},
            "primary": primary, "metrics": metrics, "comparisons": comparisons, "artifacts": artifacts,
            "budget_violations": budget_violations, "reset_checks": resets, "diagnostics": diagnostic_rows,
            "configurations": configurations, "plan": plan, "data_manifest": manifest, "provenance_verification": provenance,
            "input_sha256": hashes, "real_png_sha256": real_hashes, "notes": notes}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(json_safe(value)) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def display(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(display(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def write_report(out, report):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    with (out / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    for name in ("metrics", "comparisons", "artifacts", "reset_checks", "diagnostics", "budget_violations"):
        write_csv(out / f"{name}.csv", report[name])
    primary = report["primary"]
    lines = ["# Frozen LAO: new-image, matched saved-budget validation", "",
             f"Status: **{report['status']}**. Generated images: 50; real images: 50. Saved uint8 L-infinity budget: **8/255, zero tolerance**.", "",
             "## Primary comparison: quality-qualified LAO8 versus PGD40, E / none", "",
             table([primary], ["n_clean_detected", "left_success", "right_success", "difference", "ci95_lower", "ci95_upper", "conservative_ci95_lower", "conservative_ci95_upper", "exact_mcnemar_p"]), "",
             f"Superiority supported: **{primary['superiority_supported']}**. Descriptive 5-point noninferiority criterion supported: **{primary['noninferiority_descriptively_supported']}**.", "",
             "## Interpretation and limits", ""]
    lines.extend("- " + note for note in report["notes"])
    lines.extend(["", "## Budget-constrained methods: all samples retained", "",
                  table([row for row in report["metrics"] if not row["diagnostic_only"]],
                        ["run", "detector", "condition", "n", "n_clean_detected", "n_valid_success", "cond_asr", "n_quality_qualified_success", "quality_qualified_asr", "real_fpr", "n_budget_violations", "psnr", "lpips", "seconds"]),
                  "", "## Raw outputs: diagnostic only", "",
                  table([row for row in report["metrics"] if row["diagnostic_only"] and row["condition"] == "none"],
                        ["run", "detector", "n_clean_detected", "n_classifier_success", "classifier_flip_asr", "psnr", "lpips", "linf", "seconds"]),
                  "", "## Paired quality-qualified comparisons", "",
                  table(report["comparisons"], ["comparator", "detector", "condition", "n_clean_detected", "added_success", "lost_success", "difference", "ci95_lower", "ci95_upper", "conservative_ci95_lower", "conservative_ci95_upper", "exact_mcnemar_p"]),
                  "", "## Integrity and latent diagnostics", "",
                  f"Reset checks: **{len(report['reset_checks'])}/{len(report['reset_checks'])}** identical to independently run reconstruction baselines. "
                  f"Bounded-output violations: **{len(report['budget_violations'])}**. "
                  "Integer export projections were compared pixel for pixel with the raw outputs and canonical originals.", "",
                  table(report["diagnostics"], ["image_id", "steps", "finite_gradient_steps", "nonzero_gradient_steps", "nonzero_update_steps", "first_gradient_l2", "last_gradient_l2"]), ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing output is refused")
    args = parser.parse_args(argv)
    require(not args.out.exists(), f"Output already exists: {args.out}")
    report = build_report(args.root.resolve())
    write_report(args.out.resolve(), report)
    print(json.dumps({"status": report["status"], "out": str(args.out.resolve()), "model_executed": False,
                      "n_generated": report["n_generated"], "budget_violations": len(report["budget_violations"]),
                      "primary_difference": report["primary"]["difference"],
                      "primary_ci95": [report["primary"]["ci95_lower"], report["primary"]["ci95_upper"]]}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f"[LAO validation report] {error}")
