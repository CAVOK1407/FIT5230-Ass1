#!/usr/bin/env python3
"""Audit saved-file LAO development experiments without loading any model.

Inputs are the CSV files produced by m2_evaluate.py and immutable run directories.
The standard-library-only reporter never changes an input or overwrites a report.
Example: python m3_lao_report.py --eval_dir RESULTS/eval --run_root RESULTS --out REPORT
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


QUALITY_PSNR_MIN = 30.0
QUALITY_LPIPS_MAX = 0.15
DEFAULT_RUNS = ("R0", "R1", "R2", "R3")
MATCHED_CONFIG_KEYS = (
    "images_root", "pretrained_diffusion_path", "encoder_weights", "surrogates",
    "evaluate", "res", "seed", "diffusion_steps", "start_step", "guidance",
    "device", "torch", "source_manifest_sha256", "checkpoint_sha256",
    "encoder_weights_sha256", "model_sha256", "code_sha256",
    "loss", "kappa", "kappa_resolved", "lpips_weight", "lr", "attack_loss_weight",
    "lao_weight_decay", "lao_eot", "di_prob",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"Empty CSV: {path}")
    return rows


def number(value, label, optional=False, allow_inf=False):
    if value is None or str(value).strip() in ("", "None", "null", "nan", "NaN"):
        require(optional, f"Missing number: {label}")
        return None
    result = float(value)
    require(not math.isnan(result) and (allow_inf or math.isfinite(result)),
            f"Non-finite number: {label}={value}")
    return result


def prediction(value, label):
    result = number(value, label)
    require(result in (0, 1), f"Prediction must be 0 (AI) or 1 (real): {label}")
    return int(result)


def average(values):
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def check_close(actual, expected, label):
    if actual is None or expected is None:
        require(actual is expected, f"Summary mismatch at {label}: {actual} != {expected}")
    else:
        require(math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7),
                f"Summary mismatch at {label}: {actual} != {expected}")


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(val) for val in value]
    return value


def config_gate(config):
    present = [(key, config[key]) for key in
               ("skip_scale", "skip_gate", "control_scale", "control_strength") if key in config]
    if not present:
        return 1.0
    values = [number(value, key) for key, value in present]
    require(all(value == values[0] for value in values), f"Conflicting gate settings: {present}")
    return values[0]


def output_decoder(config):
    # decode_path selects the loss decoder, NOT the saved output decoder.
    present = [config[key] for key in ("output_decoder", "save_decoder", "save_decode_path") if key in config]
    require(not present or all(value == present[0] for value in present), "Conflicting output decoder settings")
    return present[0] if present else "final"


def load_run(run_root, name):
    folder = run_root / name
    config_path, records_path = folder / "run_config.json", folder / "records.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = {}
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        image_id = str(record["image_id"])
        require(image_id not in records, f"Duplicate image_id in {records_path}:{line_number}: {image_id}")
        require(bool(record.get("source")), f"Missing source for {name}/{image_id}")
        require(record.get("pgd") is None and not record.get("pgd_info"),
                f"Pixel PGD is present in {name}/{image_id}; this is not a pure-LAO comparison")
        require(isinstance(record.get("final"), dict), f"Missing final float predictions for {name}/{image_id}")
        require((folder / f"{image_id}_originImage.png").is_file(), f"Missing origin PNG for {name}/{image_id}")
        require((folder / f"{image_id}_adv_image.png").is_file(), f"Missing final PNG for {name}/{image_id}")
        records[image_id] = record
    require(bool(records), f"Empty records: {records_path}")
    require(config.get("encode_source") == "clean" and config.get("control_source") == "clean",
            f"{name}: both encode_source and control_source must be clean")
    require(not config.get("pgd_only", False), f"{name}: pgd_only cannot be enabled")
    for key in ("images_root", "seed", "res", "diffusion_steps", "start_step", "iterations", "decode_path"):
        require(key in config, f"{name}: missing required run_config key {key}")
    require(int(config["iterations"]) >= 0, f"{name}: iterations cannot be negative")
    if name in ("R0", "G0"):
        require(int(config["iterations"]) == 0, f"{name} must be a zero-LAO reconstruction baseline")
    if "n_images" in config:
        require(int(config["n_images"]) == len(records), f"{name}: n_images does not match records")
    sources = [record["source"] for record in records.values()]
    require(len(sources) == len(set(sources)), f"Duplicate source filenames in {name}")
    return {"folder": folder, "config": config, "records": records,
            "input_hashes": {str(config_path): sha256(config_path), str(records_path): sha256(records_path)}}


def align_runs(runs, groups):
    reference_name = next(iter(runs))
    reference = runs[reference_name]
    ids = set(reference["records"])
    for name, run in runs.items():
        require(set(run["records"]) == ids, f"Image IDs differ between {reference_name} and {name}")
        for key in MATCHED_CONFIG_KEYS:
            require(run["config"].get(key) == reference["config"].get(key),
                    f"Configuration mismatch: {reference_name} vs {name}: {key}")
        require(output_decoder(run["config"]) == output_decoder(reference["config"]),
                f"Saved output decoder differs between {reference_name} and {name}")
        for image_id in sorted(ids):
            current, original = run["records"][image_id], reference["records"][image_id]
            require(current["source"] == original["source"], f"Source mapping differs at {name}/{image_id}")
            for key in ("source_sha256", "input_sha256"):
                require(current.get(key) == original.get(key), f"{key} differs at {name}/{image_id}")
            filename = f"{image_id}_originImage.png"
            require(sha256(run["folder"] / filename) == sha256(reference["folder"] / filename),
                    f"Origin PNG differs at {name}/{image_id}")
            original_clean, current_clean = original.get("clean", {}), current.get("clean", {})
            require(set(original_clean) == set(current_clean) and bool(current_clean),
                    f"Clean detector keys differ at {name}/{image_id}")
            for detector in original_clean:
                require(prediction(current_clean[detector]["pred"], "clean record") ==
                        prediction(original_clean[detector]["pred"], "reference clean record"),
                        f"Float clean predictions differ at {name}/{image_id}/{detector}")
    for baseline, variant in groups:
        base, candidate = runs[baseline]["config"], runs[variant]["config"]
        require(int(base["iterations"]) == 0, f"{baseline} must be a zero-LAO reconstruction baseline")
        require(int(candidate["iterations"]) > 0, f"{variant} must have LAO iterations > 0")
        require(config_gate(base) == config_gate(candidate), f"Skip gate differs: {baseline} vs {variant}")
    return sorted(ids)


def numeric_diagnostics(value, prefix=""):
    """Keep scalar diagnostics, never latent tensors or image arrays."""
    result = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        relevant = any(token in path.lower() for token in
                       ("grad", "latent", "delta", "update", "loss", "attack", "lpips", "distance", "changed", "margin"))
        if isinstance(item, dict):
            result.update(numeric_diagnostics(item, path))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            if relevant:
                result[path] = item if math.isfinite(item) else str(item)
        elif relevant and isinstance(item, list) and len(item) <= 16:
            for index, scalar in enumerate(item):
                if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
                    result[f"{path}[{index}]"] = scalar if math.isfinite(scalar) else str(scalar)
    return result


def build_report(eval_dir, run_root, selected):
    runs = {name: load_run(run_root, name) for name in selected}
    groups = [("R0", name) for name in selected if name in ("R1", "R2", "R3")]
    if "G1" in selected:
        groups.append(("G0", "G1"))
    require(all(base in runs for base, _ in groups), "Each requested LAO run requires its R0 or G0 baseline")
    image_ids = align_runs(runs, groups)
    input_hashes = {}
    for run in runs.values():
        input_hashes.update(run["input_hashes"])
    reset_checks = []
    for baseline, variant in groups:
        for image_id in image_ids:
            diagnostic = runs[variant]["records"][image_id].get("lao_diagnostics", {})
            reset_name = diagnostic.get("reset_image")
            require(bool(reset_name) or not runs[variant]["config"].get("lao_diagnostics", False),
                    f"Missing reset-latent diagnostic image for {variant}/{image_id}")
            if not reset_name:
                reset_checks.append({"baseline": baseline, "run": variant, "image_id": image_id,
                                     "available": False, "identical": None, "reset_sha256": None, "baseline_sha256": None})
                continue
            require(Path(reset_name).name == reset_name, f"Reset image must be a local filename: {variant}/{image_id}")
            reset_path = runs[variant]["folder"] / reset_name
            baseline_path = runs[baseline]["folder"] / f"{image_id}_adv_image.png"
            reset_hash, baseline_hash = sha256(reset_path), sha256(baseline_path)
            require(reset_hash == baseline_hash,
                    f"Reset-latent PNG differs from its paired baseline: {variant}/{image_id} vs {baseline}; "
                    "attribution is not verified. Investigate decoder/batch/quantisation differences before interpreting success.")
            reset_checks.append({"baseline": baseline, "run": variant, "image_id": image_id,
                                 "available": True, "identical": True, "reset_sha256": reset_hash, "baseline_sha256": baseline_hash})
    files = {key: eval_dir / f"{key}.csv" for key in ("per_image", "summary", "real_predictions")}
    raw, summaries, real_rows = [read_csv(files[key]) for key in ("per_image", "summary", "real_predictions")]
    input_hashes.update({str(path): sha256(path) for path in files.values()})
    cells = defaultdict(dict)
    for row in raw:
        name = row["run"]
        if name not in runs:
            continue
        key = (name, row["detector"], row["condition"])
        image_id = row["image_id"]
        require(image_id not in cells[key], f"Duplicate per-image row: {key}/{image_id}")
        for field in ("clean_pred", "adv_pred"):
            row[field] = prediction(row[field], f"{key}/{image_id}/{field}")
        for field in ("clean_margin", "adv_margin"):
            row[field] = number(row[field], f"{key}/{image_id}/{field}")
        for field in ("psnr", "lpips", "ssim", "linf", "seconds"):
            row[field] = number(row.get(field), f"{key}/{image_id}/{field}", optional=True, allow_inf=field == "psnr")
        cells[key][image_id] = row
    require(bool(cells), "No selected runs found in per_image.csv")
    layouts = {name: {(det, cond) for run, det, cond in cells if run == name} for name in runs}
    reference_name = selected[0]
    for name in runs:
        require(layouts[name] and layouts[name] == layouts[reference_name], f"Detector/condition coverage differs for {name}")
    require(any(condition == "none" for _, condition in layouts[reference_name]), "The none condition is required")
    for (name, detector, condition), rows in cells.items():
        require(set(rows) == set(image_ids), f"Incomplete image IDs for {name}/{detector}/{condition}")
        for image_id, row in rows.items():
            ref = cells[(reference_name, detector, condition)][image_id]
            require(row["clean_pred"] == ref["clean_pred"], f"Saved clean prediction differs at {name}/{detector}/{condition}/{image_id}")
            check_close(row["clean_margin"], ref["clean_margin"], f"saved clean margin {name}/{detector}/{condition}/{image_id}")
    quality = {}
    for (name, _, condition), rows in cells.items():
        if condition != "none":
            continue
        for image_id, row in rows.items():
            key = (name, image_id)
            candidate = {field: row[field] for field in ("psnr", "lpips", "ssim", "linf", "seconds")}
            require(key not in quality or quality[key] == candidate, f"Quality metrics differ across detectors for {key}")
            quality[key] = candidate
    def qualified(name, image_id):
        q = quality[(name, image_id)]
        return q["psnr"] is not None and q["lpips"] is not None and q["psnr"] >= QUALITY_PSNR_MIN and q["lpips"] <= QUALITY_LPIPS_MAX

    real = defaultdict(dict)
    for row in real_rows:
        key = (row["detector"], row["condition"])
        require(row["image"] not in real[key], f"Duplicate real-image prediction: {key}/{row['image']}")
        real[key][row["image"]] = prediction(row["pred"], "real prediction")
    summary_lookup = {}
    for row in summaries:
        key = (row["run"], row["detector"], row["condition"])
        if row["run"] in runs:
            require(key not in summary_lookup, f"Duplicate summary row: {key}")
            summary_lookup[key] = row
    require(set(summary_lookup) == set(cells), "summary.csv and per_image.csv coverage differ")
    metrics, paired, quantization, diagnostics = [], [], [], []
    for (name, detector, condition), rows in sorted(cells.items()):
        eligible = [image_id for image_id, row in rows.items() if row["clean_pred"] == 0]
        successful = [image_id for image_id in eligible if rows[image_id]["adv_pred"] == 1]
        real_cell = real[(detector, condition)]
        require(bool(real_cell), f"No real-image predictions for {detector}/{condition}")
        metric = {"run": name, "detector": detector, "condition": condition, "n": len(rows),
                  "n_clean_detected": len(eligible), "n_success": len(successful),
                  "cond_asr": rate(len(successful), len(eligible)),
                  "uncond_escape": rate(sum(row["adv_pred"] == 1 for row in rows.values()), len(rows)),
                  "real_n": len(real_cell), "real_fpr": rate(sum(pred == 0 for pred in real_cell.values()), len(real_cell)),
                  "n_quality_qualified": sum(qualified(name, image_id) for image_id in rows),
                  "n_quality_missing": sum(quality[(name, image_id)]["psnr"] is None or quality[(name, image_id)]["lpips"] is None for image_id in rows),
                  "quality_qualified_success": sum(qualified(name, image_id) for image_id in successful)}
        metric.update({field: average(quality[(name, image_id)][field] for image_id in rows)
                       for field in ("psnr", "lpips", "ssim", "linf", "seconds")})
        summary_row = summary_lookup[(name, detector, condition)]
        for field in ("n", "n_clean_detected", "cond_asr", "uncond_escape", "real_fpr", "psnr", "lpips", "seconds"):
            check_close(metric[field], number(summary_row.get(field), f"summary/{name}/{field}", optional=True, allow_inf=field == "psnr"), f"{name}/{detector}/{condition}/{field}")
        metrics.append(metric)
        if condition == "none":
            floats = {}
            for image_id in image_ids:
                final = runs[name]["records"][image_id]["final"]
                if detector in final:
                    floats[image_id] = prediction(final[detector]["pred"], "float final prediction")
            require(not floats or len(floats) == len(image_ids), f"Partial float prediction coverage for {name}/{detector}")
            if not floats:
                quantization.append({"run": name, "detector": detector, "n": len(rows), "n_clean_detected": len(eligible),
                                     "float_predictions_available": False, "float_success": None, "png_success": len(successful),
                                     "retained_float_success": None, "retention_rate": None,
                                     "float_success_lost_on_png": None, "png_only_success": None,
                                     "all_image_prediction_disagreements": None})
                continue
            float_success = [image_id for image_id in eligible if floats[image_id] == 1]
            retained = sum(rows[image_id]["adv_pred"] == 1 for image_id in float_success)
            quantization.append({"run": name, "detector": detector, "n": len(rows), "n_clean_detected": len(eligible),
                                 "float_predictions_available": True,
                                 "float_success": len(float_success), "png_success": len(successful),
                                 "retained_float_success": retained, "retention_rate": rate(retained, len(float_success)),
                                 "float_success_lost_on_png": len(float_success) - retained,
                                 "png_only_success": sum(floats[image_id] == 0 for image_id in successful),
                                 "all_image_prediction_disagreements": sum(floats[image_id] != rows[image_id]["adv_pred"] for image_id in image_ids)})
    for baseline, variant in groups:
        for detector, condition in sorted(layouts[baseline]):
            base, candidate = cells[(baseline, detector, condition)], cells[(variant, detector, condition)]
            eligible = [image_id for image_id in image_ids if base[image_id]["clean_pred"] == 0]
            baseline_caught = [image_id for image_id in eligible if base[image_id]["adv_pred"] == 0]
            added = [image_id for image_id in baseline_caught if candidate[image_id]["adv_pred"] == 1]
            lost = [image_id for image_id in eligible if base[image_id]["adv_pred"] == 1 and candidate[image_id]["adv_pred"] == 0]
            q_added = [image_id for image_id in added if qualified(variant, image_id)]
            paired.append({"baseline": baseline, "run": variant, "detector": detector, "condition": condition,
                           "n": len(image_ids), "n_clean_detected": len(eligible), "n_clean_and_baseline_detected": len(baseline_caught),
                           "added_success": len(added), "lost_success": len(lost), "net_added_success": len(added) - len(lost),
                           "net_asr_change": rate(len(added) - len(lost), len(eligible)),
                           "success_on_clean_and_baseline_detected": len(added),
                           "rate_on_clean_and_baseline_detected": rate(len(added), len(baseline_caught)),
                           "quality_qualified_new_flips": len(q_added),
                           "quality_qualified_new_flip_rate": rate(len(q_added), len(baseline_caught)),
                           "new_flip_ids": added, "lost_flip_ids": lost, "quality_qualified_new_flip_ids": q_added})
    for name, run in runs.items():
        for image_id, record in sorted(run["records"].items()):
            history = record.get("history") or []
            item = {"run": name, "image_id": image_id, "source": record["source"], "history_steps": len(history),
                    "first": numeric_diagnostics(history[0]) if history else {},
                    "last": numeric_diagnostics(history[-1]) if history else {},
                    "diagnostics": numeric_diagnostics(record.get("diagnostics", {})),
                    "lao_diagnostics": record.get("lao_diagnostics", {}),
                    "record_diagnostics": numeric_diagnostics({key: value for key, value in record.items() if key not in ("history", "diagnostics")})}
            item["gradient_diagnostics_available"] = any("grad" in key.lower() for key in list(item["first"]) + list(item["last"]))
            diagnostics.append(item)
    notes = [
        "Exploratory architecture diagnosis, not independent final-test evidence. Development images and previously examined test images must not be presented as a fresh confirmatory test.",
        "Class convention: 0 = AI/detected, 1 = real/escape. Every reported attack outcome comes from saved-file evaluation.",
        "All samples remain in the reported denominators. Quality screening never removes failures or damaged images from ASR.",
        "Fixed exploratory quality screen: PSNR >= 30 dB and LPIPS <= 0.15 on the unprocessed saved image; this is not semantic or human review.",
        "For transformed conditions the quality screen still describes the unprocessed PNG, not the transformed image.",
        "LAO attribution uses paired comparisons against the zero-step reconstruction at the same skip gate and saved decoder.",
        "Float/PNG retention is reported only for condition=none and uses the same saved-clean eligibility mask.",
        "Logged first/last-step scores precede their respective optimiser updates; final_after_all_updates records the actual final point when available.",
        "Diagnostic-enabled timings include additional scoring and reset-latent checks and are not production-runtime benchmarks.",
        "Detector outcomes on the same source image are correlated; do not treat images times detectors as independent samples.",
        "The reporter reads files only. It performs no model inference, optimisation, training, or source-file edits.",
    ]
    return {"schema_version": 1, "runs": selected, "n_source_images": len(image_ids),
            "quality_screen": {"psnr_min": QUALITY_PSNR_MIN, "lpips_max": QUALITY_LPIPS_MAX, "semantic_confirmation": False},
            "input_sha256": input_hashes, "configurations": {name: run["config"] for name, run in runs.items()},
            "metrics": metrics, "paired": paired, "quantization": quantization, "diagnostics": diagnostics,
            "reset_checks": reset_checks, "notes": notes}


def format_value(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def markdown_table(rows, columns):
    result = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    result.extend("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(result)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)


def write_report(out, report):
    # Parent creation is harmless; the leaf is exclusive and must not already exist.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    with (out / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    for field in ("metrics", "paired", "quantization", "diagnostics", "reset_checks"):
        write_csv(out / f"{field}.csv", report[field])
    content = ["# LAO development-set diagnosis", "", f"Source images: **{report['n_source_images']}**. No model was run by this reporter.", ""]
    content.extend("- " + note for note in report["notes"])
    content.extend(["", "## Saved-image outcomes (all samples retained)", "",
                    markdown_table(report["metrics"], ["run", "detector", "condition", "n", "n_clean_detected", "n_success", "cond_asr", "real_fpr", "n_quality_qualified", "psnr", "lpips", "seconds"]),
                    "", "## Paired LAO contribution beyond reconstruction", "",
                    "Added/lost flips use the clean-detected population. The stricter subset also requires the zero-step baseline to remain detected. Quality-qualified new flips keep that full subset as the denominator.", "",
                    markdown_table(report["paired"], ["baseline", "run", "detector", "condition", "n_clean_detected", "n_clean_and_baseline_detected", "added_success", "lost_success", "net_added_success", "quality_qualified_new_flips"]),
                    "", "## Reset-latent counterfactual integrity", "",
                    "When available, each optimised run's reset-latent PNG must match its paired R0/G0 final PNG byte for byte. A mismatch stops reporting and must be investigated before claiming attribution.", "",
                    markdown_table(report["reset_checks"], ["baseline", "run", "image_id", "available", "identical"]),
                    "", "## Float-to-PNG retention (no post-processing)", "",
                    markdown_table(report["quantization"], ["run", "detector", "n_clean_detected", "float_predictions_available", "float_success", "png_success", "retained_float_success", "retention_rate", "float_success_lost_on_png", "png_only_success"]),
                    "", "## Gradient and latent diagnostics", "",
                    "First/last-step scalar values and available per-image diagnostics are preserved in diagnostics.csv and report.json. Missing diagnostics are explicitly marked unavailable; missing gradients do not establish that the actual gradient was zero.", "",
                    markdown_table(report["diagnostics"], ["run", "image_id", "history_steps", "gradient_diagnostics_available"]), ""])
    (out / "report.md").write_text("\n".join(content), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New report directory; existing paths are refused")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="Comma-separated standard names; R0,R1,R2,R3 by default")
    args = parser.parse_args(argv)
    selected = [name.strip() for name in args.runs.split(",") if name.strip()]
    require(selected and len(selected) == len(set(selected)), "--runs must contain distinct names")
    require(set(selected) <= set(DEFAULT_RUNS) | {"G0", "G1"}, "Allowed run names are R0,R1,R2,R3,G0,G1")
    if "G0" not in selected and "G1" not in selected:
        present = [(args.run_root / name).is_dir() for name in ("G0", "G1")]
        require(not any(present) or all(present), "Optional gate experiment requires both G0 and G1")
        if all(present):
            selected += ["G0", "G1"]
    require(not args.out.exists(), f"Report output already exists: {args.out}")
    report = build_report(args.eval_dir.resolve(), args.run_root.resolve(), selected)
    write_report(args.out.resolve(), report)
    print(json.dumps({"status": "verified", "n_source_images": report["n_source_images"], "runs": selected,
                      "out": str(args.out.resolve()), "model_executed": False}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f"[LAO report] {error}")
