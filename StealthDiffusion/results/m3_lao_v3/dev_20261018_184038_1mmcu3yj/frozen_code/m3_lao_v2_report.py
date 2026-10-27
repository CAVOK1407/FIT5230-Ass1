#!/usr/bin/env python3
"""Read-only second-round audit of low-distortion LAO on new images.

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
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageChops


METHODS = ("OLD_RECON8", "OLD_LAO8", "NEW_RECON8", "NEW_LAO8", "PGD10", "PGD10_MATCHED")
BOUNDED = frozenset(METHODS)
DETECTORS = ("E", "R", "D", "S")
CONDITIONS = ("none", "jpeg75", "resize05")
EXPECTED_N = 50
BUDGET_PIXELS = 8
PSNR_MIN = 30.0
LPIPS_MAX = 0.03
LEGACY_LPIPS_MAX = 0.15
NEW_MEAN_LPIPS_MAX = 0.015
FIRST_MANIFEST_SHA256 = "eb475c014ae6453ba6a35d14cde7c6b25d0421beaba4b833840bb83b82993737"
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
    if lower == upper or sorted_values[lower] == sorted_values[upper]:
        return sorted_values[lower]
    if math.isinf(sorted_values[upper]):
        return sorted_values[upper]
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
    require(manifest.get("generation_plan") == generation, "Embedded generation plan differs from the frozen plan")
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
    require(manifest.get("old_inputs") == old, "Manifest and generation exclusion evidence differ")
    prior_batches = old.get("prior_validation_batches", [])
    require(len(prior_batches) == 1, "Expected exactly one already-observed validation batch")
    prior = prior_batches[0]
    require(prior.get("manifest_sha256") == FIRST_MANIFEST_SHA256, "Unexpected first-round exclusion manifest")
    prior_path = Path(prior.get("manifest_path", ""))
    require(prior_path.is_absolute() and digest(prior_path) == FIRST_MANIFEST_SHA256,
            "First-round exclusion manifest changed")
    prior_manifest = read_json(prior_path)
    require(prior.get("n_generated") == prior.get("n_real") == 50 and prior.get("canonical_files_verified") == 100,
            "Incomplete first-round exclusion evidence")
    require(digest(Path(prior["plan_path"])) == prior["plan_sha256"] == prior_manifest["plan_sha256"],
            "First-round generation-plan binding changed")
    prior_data = Path(prior["data_root"]).resolve()
    require(prior_data == Path(prior_manifest["data_root"]).resolve(), "First-round data-root mismatch")
    prior_hashes = set()
    for row in prior_manifest["generated"] + prior_manifest["real"]:
        path = (prior_data / row["relative_path"]).resolve()
        require(path.is_relative_to(prior_data) and digest(path) == row["sha256"], "First-round canonical input changed")
        pixel_hash = hashlib.sha256(image_rgb(path).tobytes()).hexdigest()
        require(pixel_hash == row["rgb224_sha256"], "First-round canonical RGB hash changed")
        prior_hashes.add(pixel_hash)
    require(len(prior_hashes) == 100, "First-round canonical uniqueness differs")
    old_hashes = set(old["old_rgb224_sha256"])
    require(len(old_hashes) == 346, "Expected 346 old canonical RGB hashes, including first-round validation")
    require(len(set(old["old_real_source_ids"])) == 160, "Expected 160 old real-source IDs")
    require(prior_hashes <= old_hashes, "Already-observed first-round images are not all excluded")
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
                require(int(row["seed"]) == 5232000 + int(image_id), f"Unplanned new generation seed: {image_id}")
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
    freeze_checks = verify_candidate_freeze(root, generation, manifest)
    generation_records = [json.loads(line) for line in (data_root / "generation_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(generation_records == manifest["generated"], "Generation records differ from the final manifest")
    require(len(all_pixel_hashes) == 100, "Canonical image uniqueness check failed")
    return {"verified": True, "n_generated": 50, "n_real": 50, "new_unique_rgb224": 100,
            "old_overlap": 0, "old_unique_rgb224": len(old_hashes), "plan_sha256": digest(plan_path),
            "canonical_checks": canonical_checks, "candidate_freeze": freeze_checks}


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


def timestamp(value, label):
    require(isinstance(value, str), f"Missing timestamp: {label}")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, f"Timestamp needs timezone: {label}")
    return result


def verify_candidate_freeze(root, generation, manifest):
    reference = generation.get("candidate_freeze")
    require(isinstance(reference, dict) and reference == manifest.get("candidate_freeze"),
            "Candidate freeze differs between generation plan and data manifest")
    path = Path(reference.get("path", ""))
    require(path.is_absolute() and path.is_file(), "Candidate freeze must reference an existing absolute path")
    require(digest(path) == reference.get("sha256"), "Candidate freeze file hash mismatch")
    frozen = read_json(path)
    require(frozen.get("status") == "frozen", "Candidate was not frozen before validation")
    for key in ("created_at", "selected_configuration", "development_evidence", "source_sha256"):
        require(reference.get(key) == frozen.get(key), f"Candidate freeze reference differs: {key}")
    require(isinstance(frozen.get("selected_configuration"), dict) and frozen["selected_configuration"],
            "Missing selected candidate configuration")
    created = timestamp(frozen["created_at"], "candidate freeze")
    require(created <= timestamp(generation.get("created_at"), "generation plan"),
            "Candidate was frozen after new data preparation")
    require(created <= timestamp(manifest.get("created_at"), "new data manifest"),
            "Candidate was frozen after new image generation")
    evidence = frozen.get("development_evidence", {})
    evidence_path = Path(evidence.get("path", ""))
    require(evidence_path.is_absolute() and evidence_path.is_file(), "Missing absolute development evidence path")
    require(digest(evidence_path) == evidence.get("sha256"), "Frozen development evidence changed")
    source_hashes = frozen.get("source_sha256")
    require(isinstance(source_hashes, dict) and source_hashes, "Candidate freeze has no source hashes")
    for filename, expected in source_hashes.items():
        source = Path(filename)
        require(source.is_absolute() and digest(source) == expected, f"Candidate source changed: {filename}")
    plan = read_json(root / "plan.json")
    require(frozen.get("diffusion_model") == plan.get("diffusion_model"), "Frozen diffusion model differs from validation plan")
    if "checkpoint_sha256" in plan:
        require(frozen.get("checkpoint_sha256") == plan["checkpoint_sha256"], "Frozen detector checkpoints differ from validation plan")
    if "protocol_sha256" in plan:
        require(frozen.get("protocol_sha256") == plan["protocol_sha256"], "Frozen candidate protocol differs from validation plan")
    if "candidate_freeze" in plan:
        if isinstance(plan["candidate_freeze"], str):
            require(Path(plan["candidate_freeze"]).resolve() == path.resolve() and
                    plan.get("candidate_freeze_sha256") == reference["sha256"],
                    "Experiment and generation candidate freezes differ")
        else:
            require(plan["candidate_freeze"] == reference, "Experiment and generation candidate freezes differ")
    return {"verified": True, "path": str(path), "sha256": digest(path), "created_at": frozen["created_at"],
            "before_generation_plan": True, "development_evidence": evidence,
            "selected_configuration": frozen["selected_configuration"], "source_files_verified": len(source_hashes)}


def validate_plan(root, plan):
    for key, expected in (("n_generated", EXPECTED_N), ("n_real", EXPECTED_N), ("budget_pixels", BUDGET_PIXELS)):
        require(plan.get(key) == expected, f"Unexpected frozen plan {key}: expected {expected}")
    require(plan.get("quality_screen") == {"psnr_min": PSNR_MIN, "lpips_max": LPIPS_MAX}, "Unexpected low-distortion screen")
    require(plan.get("legacy_quality_screen") == {"psnr_min": PSNR_MIN, "lpips_max": LEGACY_LPIPS_MAX}, "Unexpected legacy screen")
    require(plan.get("new_mean_lpips_max") == NEW_MEAN_LPIPS_MAX, "Unexpected new-method mean LPIPS guard")
    primary = plan.get("primary", {})
    for key, value in {"run": "NEW_LAO8", "comparator": "OLD_LAO8", "detector": "E", "condition": "none",
                       "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES}.items():
        require(primary.get(key) == value, f"Unexpected frozen primary setting: {key}")
    paths = plan.get("method_paths", {})
    require(set(paths) == set(METHODS), "Plan must map exactly all six evaluation methods")
    require(all(Path(value).is_absolute() and Path(value).is_dir() for value in paths.values()),
            "Method paths must be existing absolute directories")
    require(len({str(Path(value).resolve()) for value in paths.values()}) == len(METHODS), "Method paths must be distinct")
    require(isinstance(plan.get("source_sha256"), dict) and plan["source_sha256"], "Missing frozen runtime hashes")
    for filename, expected in plan["source_sha256"].items():
        source = Path(filename)
        require(source.is_absolute() and digest(source) == expected, f"Frozen runtime source changed: {filename}")
        relative = source.relative_to(Path(plan["working_directory"])) if "working_directory" in plan else Path(source.name)
        require(digest(root / "frozen_code" / relative) == expected, f"Archived runtime source mismatch: {relative}")
    for filename, expected in plan.get("checkpoint_sha256", {}).items():
        require(Path(filename).is_absolute() and digest(Path(filename)) == expected, f"Frozen checkpoint changed: {filename}")
    diffusion = plan.get("diffusion_model", {})
    model_root = Path(diffusion.get("resolved_local_path", ""))
    require(model_root.is_absolute() and len(diffusion.get("file_sha256", {})) == 8,
            "Expected all eight frozen diffusion-model source files")
    for relative, expected in diffusion["file_sha256"].items():
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts,
                f"Unsafe diffusion-model relative path: {relative}")
        # Hugging Face snapshot files normally symlink into ../../blobs. Keep
        # the lexical path in the pinned snapshot and verify the target bytes.
        path = model_root / relative_path
        require(path.is_relative_to(model_root) and digest(path) == expected,
                f"Frozen diffusion-model source changed: {relative}")
    if "protocol_sha256" in plan:
        require(digest(root / "protocol.md") == plan["protocol_sha256"], "Frozen protocol changed")


def architecture_config(config):
    return config.get("source_raw_config", config)


def read_artifacts(root, plan):
    originals = {path.stem: path for path in sorted((root / "data/gen/test").glob("*.png"))}
    real_paths = sorted((root / "data/real/test").glob("*.png"))
    require(set(originals) == {f"{i:04d}" for i in range(EXPECTED_N)} and len(real_paths) == EXPECTED_N,
            "Expected exactly 50 canonical generated and 50 real PNGs")
    original_hashes = {image_id: digest(path) for image_id, path in originals.items()}
    real_hashes = {path.name: digest(path) for path in real_paths}
    original_images = {image_id: image_rgb(path) for image_id, path in originals.items()}
    folders = {name: Path(plan["method_paths"][name]).resolve() for name in METHODS}
    configurations, records, artifacts, hashes, images = {}, {}, [], {}, {}
    for name, folder in folders.items():
        configurations[name] = read_json(folder / "run_config.json")
        records[name] = read_records(folder, originals)
        for filename in ("run_config.json", "records.jsonl"):
            hashes[str(folder / filename)] = digest(folder / filename)
        require({path.name.removesuffix("_adv_image.png") for path in folder.glob("*_adv_image.png")} == set(originals),
                f"Missing/extra output PNGs: {name}")
        for image_id, canonical in originals.items():
            origin = folder / f"{image_id}_originImage.png"
            require(digest(origin) == original_hashes[image_id], f"Origin is not canonical bytes: {name}/{image_id}")
            output_path = folder / f"{image_id}_adv_image.png"
            output = image_rgb(output_path)
            images[(name, image_id)] = output
            maximum, psnr = pixel_metrics(original_images[image_id], output)
            artifact = {"run": name, "image_id": image_id, "source": canonical.name,
                        "max_abs_delta_uint8": maximum, "linf": maximum / 255, "psnr_pixels": psnr,
                        "budget_valid": maximum <= BUDGET_PIXELS,
                        "origin_sha256": original_hashes[image_id], "output_sha256": digest(output_path)}
            record = records[name][image_id]
            if record.get("output_sha256") is not None:
                require(record["output_sha256"] == artifact["output_sha256"], f"Recorded output SHA mismatch: {name}/{image_id}")
            for key in ("source_file_sha256", "canonical_source_png_sha256"):
                if record.get(key) is not None:
                    require(record[key] == original_hashes[image_id], f"Recorded origin SHA mismatch: {name}/{image_id}")
            artifacts.append(artifact)
    # Configuration comparisons are within each counterfactual pair; new and old
    # objectives intentionally differ. No images may be driven by pixel PGD.
    resets = []
    for image_id in originals:
        new_zero = folders["NEW_RECON8"] / f"{image_id}_adv_image.png"
        old_zero = folders["OLD_RECON8"] / f"{image_id}_adv_image.png"
        new_hash, old_hash = digest(new_zero), digest(old_zero)
        require(new_hash == old_hash, f"New and old zero-step renderers differ: {image_id}")
        resets.append({"run": "NEW_RECON8", "baseline": "OLD_RECON8", "image_id": image_id,
                       "check": "zero_step_renderer_equivalence", "identical": True,
                       "reset_sha256": new_hash, "baseline_sha256": old_hash})
    for candidate, baseline in (("OLD_LAO8", "OLD_RECON8"), ("NEW_LAO8", "NEW_RECON8")):
        left, right = architecture_config(configurations[candidate]), architecture_config(configurations[baseline])
        require(left.get("lao_residual_anchor") is True and right.get("lao_residual_anchor") is True,
                f"Residual anchor missing: {candidate}/{baseline}")
        for key in ("encode_source", "control_source"):
            require(left.get(key) == right.get(key) == "clean", f"Non-clean {key}: {candidate}/{baseline}")
        require(left.get("surrogates") == right.get("surrogates") == "E", "Unexpected LAO optimisation target")
        require(left.get("control_scale") == right.get("control_scale") == 0.5, "Unexpected residual decoder gate")
        require(right.get("iterations") == 0, f"Reconstruction baseline was optimised: {baseline}")
        require(isinstance(left.get("iterations"), int) and left["iterations"] > 0, f"No latent optimisation: {candidate}")
        if candidate == "OLD_LAO8":
            require(left["iterations"] == 10, "Old frozen LAO must retain ten updates")
            require(float(left.get("kappa", -1)) == 3 and left.get("lpips_weight") == 1 and left.get("lr") == 0.01,
                    "Old frozen optimisation settings changed")
        else:
            freeze = read_json(root / "data/generation_plan.json")["candidate_freeze"]["selected_configuration"]
            require(left["iterations"] == freeze.get("iterations") and left.get("lr") == freeze.get("lr"),
                    "New method differs from the selected frozen candidate")
            require(left.get("lao_quantization") == "integer_projection_ste" and left.get("budget_pixels") == BUDGET_PIXELS,
                    "New method is not saved-budget-aware")
            require(float(left.get("kappa", -1)) == 3 and left.get("lpips_weight") == 1,
                    "New fixed optimisation objective changed")
            require(left.get("success_rule") == {"E_margin_min": 0.1, "lpips_max": LPIPS_MAX, "psnr_min": PSNR_MIN},
                    "New fixed selection rule changed")
            require(left.get("refinement_evaluations") == 5 and right.get("independent_zero_step") is True,
                    "Missing fixed five-probe refinement or independent zero-step control")
        for key in ("seed", "res", "diffusion_steps", "start_step", "guidance", "pretrained_diffusion_path", "encoder_weights",
                    "decode_path", "loss", "kappa", "kappa_resolved", "lpips_weight", "lr", "attack_loss_weight",
                    "lao_weight_decay", "lao_eot", "di_prob"):
            require(left.get(key) == right.get(key), f"Counterfactual configuration mismatch: {candidate}/{key}")
        for image_id in originals:
            for name in (candidate, baseline):
                record = records[name][image_id]
                require(record.get("pgd") is None and not record.get("pgd_info"), f"Pixel PGD contaminated {name}/{image_id}")
            reset_path = folders[candidate] / f"{image_id}_reset_image.png"
            baseline_path = folders[baseline] / f"{image_id}_adv_image.png"
            reset_hash, baseline_hash = digest(reset_path), digest(baseline_path)
            require(reset_hash == baseline_hash, f"Reset differs from independent baseline: {candidate}/{image_id}")
            resets.append({"run": candidate, "baseline": baseline, "image_id": image_id, "identical": True,
                           "reset_sha256": reset_hash, "baseline_sha256": baseline_hash})
    # Legacy exported outputs must still be exact projections of their frozen
    # raw files; the new method optimises the saved-budget image directly.
    for name in ("OLD_LAO8", "OLD_RECON8"):
        config = configurations[name]
        require(config.get("derived_from_frozen_output") is True, f"Legacy output is not a frozen export: {name}")
        raw_folder = Path(config["source_run"] if name == "OLD_LAO8" else config["baseline_run"])
        hash_prefix = "source" if name == "OLD_LAO8" else "baseline"
        require(raw_folder.is_absolute(), "Legacy source directory is not absolute")
        require(digest(raw_folder / "run_config.json") == config[hash_prefix + "_config_sha256"], "Legacy raw configuration changed")
        require(digest(raw_folder / "records.jsonl") == config[hash_prefix + "_records_sha256"], "Legacy raw records changed")
        for image_id in originals:
            record = records[name][image_id]
            require(digest(raw_folder / f"{image_id}_originImage.png") == original_hashes[image_id],
                    f"Legacy raw origin is not canonical bytes: {name}/{image_id}")
            raw_path = Path(record["raw_candidate_file"])
            require(raw_path.resolve() == (raw_folder / f"{image_id}_adv_image.png").resolve(), "Legacy raw output path mismatch")
            require(digest(raw_path) == record["raw_candidate_sha256"], "Legacy raw output changed")
            expected = projected_pixels(image_rgb(raw_path), original_images[image_id])
            require(expected.tobytes() == images[(name, image_id)].tobytes(), f"Legacy export projection mismatch: {name}/{image_id}")
            if name == "OLD_LAO8":
                raw_reset = raw_folder / f"{image_id}_reset_image.png"
                raw_baseline = Path(config["baseline_run"]) / f"{image_id}_adv_image.png"
                require(digest(raw_reset) == digest(raw_baseline) == record["raw_reset_sha256"] == record["independent_baseline_sha256"],
                        f"Legacy raw reset differs from independent raw baseline: {image_id}")
                resets.append({"run": "OLD_RAW_LAO", "baseline": "OLD_RAW_RECON", "image_id": image_id,
                               "identical": True, "reset_sha256": digest(raw_reset), "baseline_sha256": digest(raw_baseline)})
    return originals, real_hashes, configurations, records, artifacts, resets, hashes




def build_report(root):
    plan_path, manifest_path = root / "plan.json", root / "data_manifest.json"
    plan, manifest = read_json(plan_path), read_json(manifest_path)
    validate_plan(root, plan)
    provenance = verify_data_provenance(root, manifest)
    originals, real_hashes, configurations, records, artifacts, resets, hashes = read_artifacts(root, plan)
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
            reference = cells[("OLD_RECON8", detector, condition)][image_id]
            require(row["clean_pred"] == reference["clean_pred"], f"Clean eligibility differs at {name}/{detector}/{condition}/{image_id}")
            close(row["clean_margin"], reference["clean_margin"], "clean margin")
    quality = {}
    for name in METHODS:
        for image_id in image_ids:
            row = cells[(name, "E", "none")][image_id]
            require(row["psnr"] is not None and row["lpips"] is not None, f"Missing primary quality metrics: {name}/{image_id}")
            require(row["lpips"] >= 0, f"Negative LPIPS: {name}/{image_id}")
            require(row["seconds"] is not None and row["seconds"] >= 0, f"Missing/negative run time: {name}/{image_id}")
            close(row["psnr"], artifact_index[(name, image_id)]["psnr_pixels"], "saved-pixel PSNR")
            close(row["linf"], artifact_index[(name, image_id)]["linf"], "saved-pixel L-infinity")
            metric = {field: row[field] for field in ("psnr", "lpips", "ssim", "linf", "seconds")}
            for detector in DETECTORS:
                peer = cells[(name, detector, "none")][image_id]
                require(all(peer[field] == metric[field] for field in metric), f"Quality/time differs across detectors: {name}/{image_id}")
            metric["qualified"] = metric["psnr"] >= PSNR_MIN and metric["lpips"] <= LPIPS_MAX
            metric["legacy_qualified"] = metric["psnr"] >= PSNR_MIN and metric["lpips"] <= LEGACY_LPIPS_MAX
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

    def qualified_success(name, detector, condition, image_id, screen="qualified"):
        return bool(valid(name, image_id) and quality[(name, image_id)][screen] and
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
                legacy_success = [image_id for image_id in eligible if qualified_success(name, detector, condition, image_id, "legacy_qualified")]
                metric = {"run": name, "diagnostic_only": name not in BOUNDED, "detector": detector, "condition": condition,
                          "n": EXPECTED_N, "n_clean_detected": len(eligible), "n_classifier_success": len(classified_success),
                          "n_valid_success": len(valid_success), "n_quality_qualified_success": len(quality_success),
                          "n_legacy_quality_success": len(legacy_success),
                          "legacy_quality_asr": fraction(len(legacy_success), len(eligible)),
                          "classifier_flip_asr": fraction(len(classified_success), len(eligible)),
                          "cond_asr": fraction(len(valid_success), len(eligible)),
                          "quality_qualified_asr": fraction(len(quality_success), len(eligible)),
                          "uncond_escape": sum(row["adv_pred"] == 1 for row in rows.values()) / EXPECTED_N,
                          "uncond_valid_escape": sum(valid(name, image_id) and rows[image_id]["adv_pred"] == 1 for image_id in image_ids) / EXPECTED_N,
                          "real_n": EXPECTED_N, "real_fpr": sum(value == 0 for value in real[(detector, condition)].values()) / EXPECTED_N,
                          "n_real_false_positive": sum(value == 0 for value in real[(detector, condition)].values()),
                          "n_budget_violations": sum(not valid(name, image_id) for image_id in image_ids),
                          "n_quality_qualified": sum(quality[(name, image_id)]["qualified"] for image_id in image_ids)}
                metric.update({field: avg(quality[(name, image_id)][field] for image_id in image_ids)
                               for field in ("psnr", "lpips", "ssim", "linf", "seconds")})
                metric["timing_is_shared"] = any(record.get("timing_is_shared") is True or record.get("seconds_is_shared") is True
                                               for record in records[name].values())
                metric["median_seconds"] = median(quality[(name, image_id)]["seconds"] for image_id in image_ids)
                for field in ("psnr", "lpips"):
                    ordered = sorted(quality[(name, image_id)][field] for image_id in image_ids)
                    metric[field + "_min"] = ordered[0]
                    metric[field + "_median"] = median(ordered)
                    metric[field + "_max"] = ordered[-1]
                    metric[field + "_p95"] = percentile(ordered, 0.95)
                metrics.append(metric)

    comparisons = []
    for comparator in ("OLD_LAO8", "PGD10", "PGD10_MATCHED", "NEW_RECON8", "OLD_RECON8"):
        for detector in DETECTORS:
            for condition in CONDITIONS:
                eligible = [image_id for image_id in image_ids if cells[("NEW_LAO8", detector, condition)][image_id]["clean_pred"] == 0]
                left = [qualified_success("NEW_LAO8", detector, condition, image_id) for image_id in eligible]
                right = [qualified_success(comparator, detector, condition, image_id) for image_id in eligible]
                added = [image_id for image_id, a, b in zip(eligible, left, right) if a and not b]
                lost = [image_id for image_id, a, b in zip(eligible, left, right) if b and not a]
                row = {"run": "NEW_LAO8", "comparator": comparator, "detector": detector, "condition": condition,
                       "primary": comparator == "OLD_LAO8" and detector == "E" and condition == "none",
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
    budget_violations = [row for row in artifacts if not row["budget_valid"]]
    primary["new_all_sample_mean_lpips"] = avg(quality[("NEW_LAO8", image_id)]["lpips"] for image_id in image_ids)
    primary["new_mean_lpips_limit"] = NEW_MEAN_LPIPS_MAX
    primary["mean_lpips_guard_passed"] = primary["new_all_sample_mean_lpips"] <= NEW_MEAN_LPIPS_MAX
    lower, p_value = primary["ci95_lower"], primary["exact_mcnemar_p"]
    primary["statistical_improvement_criterion"] = lower is not None and lower > 0 and p_value < 0.05
    primary["all_artifacts_valid"] = not budget_violations
    primary["improvement_supported"] = (primary["statistical_improvement_criterion"] and
                                        primary["mean_lpips_guard_passed"] and not budget_violations)
    primary["bootstrap"] = {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
                            "unit": "paired source image within the common clean-detected population",
                            "interval": "two-sided 95% percentile, linear interpolation", "rng": "Python random.Random"}
    diagnostics = collect_diagnostics(plan, configurations, records, cells, artifact_index, image_ids)
    hashes.update({str(path): digest(path) for path in [plan_path, manifest_path, *paths.values()]})
    notes = [
        "Second-round validation uses 50 new SD1.5 seeds, selected without detector outcomes after the development candidate and code were frozen. All 100 canonical file/pixel hashes, planned source choices, old exclusions including the observed first validation, and freeze timestamps were checked.",
        "Primary outcome is NEW_LAO8 minus frozen OLD_LAO8 on E/none: conditional success within the common clean-detected set, requiring saved uint8 L-infinity <= 8, PSNR >= 30 and LPIPS <= 0.03. All samples remain in the denominator; invalid or quality-failed outputs count as failures.",
        "A clear improvement requires a paired bootstrap 95% lower bound > 0, two-sided exact McNemar p < 0.05, NEW_LAO8 all-sample mean LPIPS <= 0.015, and passing budget/source/reset integrity. Failure to meet this rule is not proof of equivalence or noninferiority.",
        "The stricter quality screen is a new, predeclared research question, motivated by the previous round. The original PSNR >= 30 / LPIPS <= 0.15 screen and classifier-only ASR are also reported; previous results are not overwritten.",
        "OLD_LAO8 optimises its unprojected residual decode and is projected only when exported. NEW_LAO8 optimises exact saved-budget values through a straight-through derivative and selects outputs using E and LPIPS. This compares complete methods, not a single isolated mechanism or equal computation budgets.",
        "PGD10 retains its final iterate. PGD10_MATCHED uses the predeclared E-margin/quality selection rule on its fixed ten-step trajectory. These are secondary controls: PGD losing points under a stricter LPIPS screen does not prove that latent attacks are generally stronger.",
        "NEW_RECON8 and OLD_RECON8 are independently executed zero-step controls and must match each other byte for byte. Each attack reset must also match its respective baseline. The method is experimental residual rendering, not a guarantee that output lies on an ordinary decoder manifold.",
        "PSNR and LPIPS are numerical quality screens, not human confirmation of meaning or usability. Quality is measured on saved PNG before postprocessing; the 8/255 budget also applies before JPEG/resize.",
        "R/D/S and transformed-image cells are descriptive and must not select methods or outputs. Always interpret them alongside real-image FPR; changing preprocessing can invalidate a detector operating point. Detector cells share sources and must not be pooled as independent observations.",
        "New seeds from the same ten prompt categories do not establish generalisation to new prompt families, generators or detectors. Only the original ten development images selected the candidate; the previous 50 are already observed and explicitly excluded.",
        "Secondary intervals and tests are not adjusted for multiple comparisons. Conservative paired-risk-difference intervals remain available to show finite-sample uncertainty, including when all paired outcomes agree; no noninferiority claim is made.",
        "A stronger standalone latent contribution does not establish dominance inside the original hybrid PGD-plus-LAO pipeline. Timing follows recorded work, including diagnostics and any shared trajectory; medians accompany means.",
        "This reporter only reads saved PNGs, records and CSVs and writes a new report directory. It executes no detector, generator, training or attack.",
    ]
    return {"schema_version": 2, "status": "verified" if not budget_violations else "budget_violations_present",
            "n_generated": EXPECTED_N, "n_real": EXPECTED_N, "budget_pixels": BUDGET_PIXELS,
            "quality_screen": {"psnr_min": PSNR_MIN, "lpips_max": LPIPS_MAX},
            "legacy_quality_screen": {"psnr_min": PSNR_MIN, "lpips_max": LEGACY_LPIPS_MAX},
            "primary": primary, "metrics": metrics, "comparisons": comparisons, "artifacts": artifacts,
            "budget_violations": budget_violations, "reset_checks": resets, "diagnostics": diagnostics,
            "configurations": configurations, "plan": plan, "data_manifest": manifest,
            "provenance_verification": provenance, "input_sha256": hashes, "real_png_sha256": real_hashes,
            "notes": notes}


def collect_diagnostics(plan, configurations, records, cells, artifact_index, image_ids):
    """Retain diagnostics verbatim; summaries must not hide zero/failed updates."""
    output = []
    raw_folder = Path(configurations["OLD_LAO8"]["source_run"])
    originals = {image_id: Path(f"{image_id}.png") for image_id in image_ids}
    old_records = read_records(raw_folder, originals)
    for name, source_records in (("OLD_LAO8", old_records), ("NEW_LAO8", records["NEW_LAO8"])):
        for image_id in image_ids:
            record = source_records[image_id]
            if name == "OLD_LAO8":
                history = record.get("history", [])
                require(len(history) == 10, f"Frozen old trajectory is not ten updates: {image_id}")
                gradients = [numeric(step["lao_diagnostics"]["gradient"]["l2"], "old gradient L2") for step in history]
                updates = [numeric(step["lao_diagnostics"]["update"]["l2"], "old update L2") for step in history]
                details = record.get("lao_diagnostics", {})
                selected_step, actual_steps = 10, 10
            else:
                history = record.get("gradient_steps", [])
                gradients = [numeric(step["grad_l2"], "new gradient L2") for step in history]
                updates = [numeric(step["update_l2"], "new update L2") for step in history]
                actual_steps = record.get("actual_steps")
                require(isinstance(actual_steps, int) and actual_steps >= 0, f"Missing new actual-step count: {image_id}")
                require(actual_steps == len(history), f"New gradient history does not cover actual updates: {image_id}")
                selected_step = record.get("selected_step")
                require(isinstance(selected_step, int) and 0 <= selected_step <= actual_steps,
                        f"Invalid new selected step: {image_id}")
                details = {key: record.get(key) for key in ("selected", "selected_phase", "stop_reason", "gradient_steps",
                           "history", "refinement_history", "shared_actual_steps", "lao_diagnostics")}
                selected = record.get("selected", {})
                saved = selected.get("saved_png", {})
                artifact = artifact_index[(name, image_id)]
                require(saved.get("sha256") == artifact["output_sha256"], f"Selected PNG hash differs: {image_id}")
                require(saved.get("max_difference_pixels") == artifact["max_abs_delta_uint8"], f"Selected PNG budget differs: {image_id}")
                row = cells[(name, "E", "none")][image_id]
                require(saved.get("pred") == row["adv_pred"], f"Selected PNG prediction differs from evaluation: {image_id}")
                for key, field in (("real_margin", "adv_margin"), ("lpips", "lpips"), ("psnr", "psnr")):
                    observed = math.inf if key == "psnr" and saved.get("psnr_infinite") is True else numeric(
                        saved.get(key), "selected " + key, infinity=key == "psnr")
                    require(math.isclose(observed, row[field], rel_tol=1e-5, abs_tol=1e-5),
                            f"Selected PNG {key} differs from independent evaluation: {image_id}")
            require(all(value >= 0 for value in gradients + updates), "A gradient/update norm is negative")
            item = {"run": name, "image_id": image_id, "selected_step": selected_step, "actual_steps": actual_steps,
                    "finite_gradient_steps": len(gradients), "nonzero_gradient_steps": sum(value > 0 for value in gradients),
                    "nonzero_update_steps": sum(value > 0 for value in updates),
                    "first_gradient_l2": gradients[0] if gradients else None,
                    "last_gradient_l2": gradients[-1] if gradients else None, "details": details}
            if name == "NEW_LAO8":
                margin = record.get("selected", {}).get("raw_float_real_margin")
                item["raw_float_real_margin"] = numeric(margin, "raw float margin", optional=True)
                item["saved_png_pred"] = cells[(name, "E", "none")][image_id]["adv_pred"]
                item["float_to_saved_real_retained"] = None if margin is None or float(margin) <= 0 else item["saved_png_pred"] == 1
            output.append(item)
    return output


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
    lines = ["# LAO v2: frozen low-distortion candidate on 50 new images", "",
             f"Status: **{report['status']}**. Generated images: 50; real images: 50. Saved uint8 L-infinity budget: **8/255, zero tolerance**.", "",
             "## Primary: NEW_LAO8 minus OLD_LAO8, E / original PNG", "",
             "Low-distortion success requires PSNR ≥ 30 dB and LPIPS ≤ 0.03. The all-sample mean LPIPS guard for NEW_LAO8 is ≤ 0.015.", "",
             table([primary], ["n_clean_detected", "left_success", "right_success", "added_success", "lost_success", "difference",
                               "ci95_lower", "ci95_upper", "conservative_ci95_lower", "conservative_ci95_upper", "exact_mcnemar_p"]), "",
             f"Clear improvement supported: **{primary['improvement_supported']}**. "
             f"NEW_LAO8 mean LPIPS: **{primary['new_all_sample_mean_lpips']:.6f}**; mean-LPIPS guard passed: **{primary['mean_lpips_guard_passed']}**.", "",
             "## Interpretation and limits", ""]
    lines.extend("- " + note for note in report["notes"])
    lines.extend(["", "## Original-PNG results: every source retained", "",
                  table([row for row in report["metrics"] if row["condition"] == "none"],
                        ["run", "detector", "n", "n_clean_detected", "n_valid_success", "cond_asr", "n_quality_qualified_success",
                         "quality_qualified_asr", "legacy_quality_asr", "uncond_escape", "real_fpr", "lpips", "lpips_p95", "lpips_max",
                         "psnr_median", "seconds", "median_seconds"]),
                  "", "## Secondary postprocessing: inspect real-image FPR", "",
                  table([row for row in report["metrics"] if row["condition"] != "none"],
                        ["run", "detector", "condition", "n_clean_detected", "n_valid_success", "cond_asr",
                         "n_quality_qualified_success", "quality_qualified_asr", "legacy_quality_asr", "real_fpr"]),
                  "", "## Paired low-distortion comparisons", "",
                  table(report["comparisons"], ["comparator", "detector", "condition", "n_clean_detected", "added_success", "lost_success",
                                                "difference", "ci95_lower", "ci95_upper", "exact_mcnemar_p"]),
                  "", "## Integrity and diagnostics", "",
                  f"Reset checks: **{len(report['reset_checks'])}/{len(report['reset_checks'])}** identical to independently executed zero-step controls. "
                  f"Budget violations: **{len(report['budget_violations'])}**. All canonical inputs and origins were checked; old-method exports were verified pixel for pixel.", "",
                  table(report["diagnostics"], ["run", "image_id", "selected_step", "actual_steps", "finite_gradient_steps",
                                                "nonzero_gradient_steps", "nonzero_update_steps", "first_gradient_l2", "last_gradient_l2"]), ""])
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
                      "primary_ci95": [report["primary"]["ci95_lower"], report["primary"]["ci95_upper"]],
                      "improvement_supported": report["primary"]["improvement_supported"]}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f"[LAO v2 report] {error}")


