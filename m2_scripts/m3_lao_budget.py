"""Strict saved-PNG budgets and read-only derivation of frozen LAO candidates.

This module uses only NumPy/PIL and the standard library. It never loads a model.
The integer output is the authoritative artifact; no float re-quantization occurs.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels_sha256(pixels):
    validate_rgb_u8(pixels, "pixels")
    return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()


def validate_budget(budget_pixels):
    if isinstance(budget_pixels, (bool, np.bool_)) or not isinstance(budget_pixels, (int, np.integer)):
        raise ValueError("budget_pixels must be an integer")
    if not 0 <= budget_pixels <= 255:
        raise ValueError("budget_pixels must be in [0, 255]")
    return int(budget_pixels)


def validate_rgb_u8(pixels, name="image"):
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
        raise TypeError(f"{name} must be a uint8 NumPy array")
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty HWC RGB shape")


def check_pair(candidate, reference):
    validate_rgb_u8(candidate, "candidate")
    validate_rgb_u8(reference, "reference")
    if candidate.shape != reference.shape:
        raise ValueError("candidate and canonical reference shapes differ")


def max_difference_pixels(candidate, reference):
    check_pair(candidate, reference)
    return int(np.abs(candidate.astype(np.int16) - reference.astype(np.int16)).max())


def project_uint8(candidate, reference, budget_pixels):
    """Project in signed integer arithmetic, with no uint8 wraparound."""
    budget_pixels = validate_budget(budget_pixels)
    check_pair(candidate, reference)
    clean = reference.astype(np.int16)
    lower = np.maximum(clean - budget_pixels, 0)
    upper = np.minimum(clean + budget_pixels, 255)
    return np.clip(candidate.astype(np.int16), lower, upper).astype(np.uint8)


def projection_stats(candidate, reference, output, budget_pixels):
    budget_pixels = validate_budget(budget_pixels)
    check_pair(candidate, reference)
    check_pair(output, reference)
    expected = project_uint8(candidate, reference, budget_pixels)
    if not np.array_equal(expected, output):
        raise ValueError("output is not the declared integer projection")
    changed = candidate != output
    return {"budget_pixels": budget_pixels,
            "pre_max_difference_pixels": max_difference_pixels(candidate, reference),
            "post_max_difference_pixels": max_difference_pixels(output, reference),
            "clipped_pixel_fraction": float(np.any(changed, axis=2).mean()),
            "clipped_channel_fraction": float(changed.mean()),
            "candidate_pixels_sha256": pixels_sha256(candidate),
            "reference_pixels_sha256": pixels_sha256(reference),
            "output_pixels_sha256": pixels_sha256(output)}


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def canonical_source(path, res=224):
    """Canonical input uses RGB/PIL only, matching the attack's Lanczos resize."""
    if isinstance(res, bool) or not isinstance(res, int) or res <= 0:
        raise ValueError("res must be a positive integer")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (res, res):
            image = image.resize((res, res), resample=Image.LANCZOS)
        return np.array(image, dtype=np.uint8)


def save_png_verified(path, pixels, reference=None, budget_pixels=None):
    """Exclusive PNG creation followed by exact pixel and optional budget checks."""
    path = Path(path)
    validate_rgb_u8(pixels)
    if (reference is None) != (budget_pixels is None):
        raise ValueError("reference and budget_pixels must be supplied together")
    if reference is not None:
        budget_pixels = validate_budget(budget_pixels)
        if max_difference_pixels(pixels, reference) > budget_pixels:
            raise ValueError("refusing to save an image outside the strict pixel budget")
    with path.open("xb") as handle:
        Image.fromarray(pixels, mode="RGB").save(handle, format="PNG")
    reloaded = load_rgb(path)
    if not np.array_equal(reloaded, pixels):
        raise RuntimeError(f"PNG serialization changed pixels: {path}")
    maximum = None
    if reference is not None:
        maximum = max_difference_pixels(reloaded, reference)
        if maximum > budget_pixels:
            raise RuntimeError(f"saved PNG violates the strict integer budget: {path}")
    return {"sha256": sha256(path), "pixels_sha256": pixels_sha256(reloaded),
            "max_difference_pixels": maximum}


def image_paths(images_root):
    def natural_key(path):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]
    paths = sorted((path for path in Path(images_root).iterdir()
                    if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")), key=natural_key)
    if not paths:
        raise ValueError(f"no input images in {images_root}")
    return paths


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def export_budget(source_run, baseline_run, images_root, out_root, budget_pixels=8):
    source_run, baseline_run, images_root, out_root = [Path(path).resolve() for path in
                                                     (source_run, baseline_run, images_root, out_root)]
    budget_pixels = validate_budget(budget_pixels)
    if out_root.exists() and not out_root.is_dir():
        raise FileExistsError(f"output root is not a directory: {out_root}")
    for name in ("LAO8", "RECON8", "export_manifest.json"):
        if (out_root / name).exists():
            raise FileExistsError(f"output target already exists: {out_root / name}")
    source_config = json.loads((source_run / "run_config.json").read_text())
    baseline_config = json.loads((baseline_run / "run_config.json").read_text())
    if source_run == baseline_run or baseline_config.get("iterations") != 0 or source_config.get("iterations", 0) <= 0:
        raise ValueError("separate zero-step baseline and optimized source runs are required")
    if (source_config.get("encode_source") != "clean" or source_config.get("control_source") != "clean"
            or source_config.get("decode_path") != "final" or source_config.get("lao_residual_anchor") is not True):
        raise ValueError("budget export requires the frozen clean/clean final-path residual-anchor source")
    for key in ("images_root", "res", "seed", "encode_source", "control_source", "control_scale",
                "decode_path", "lao_residual_anchor", "diffusion_steps", "start_step", "guidance",
                "pretrained_diffusion_path", "encoder_weights"):
        if source_config.get(key) != baseline_config.get(key):
            raise ValueError(f"source and independent baseline differ in {key}")
    records = [json.loads(line) for line in (source_run / "records.jsonl").read_text().splitlines() if line.strip()]
    baseline_records = [json.loads(line) for line in (baseline_run / "records.jsonl").read_text().splitlines() if line.strip()]
    baselines = {str(record["image_id"]): record for record in baseline_records}
    if len(baselines) != len(baseline_records) or set(baselines) != {str(record["image_id"]) for record in records}:
        raise ValueError("independent baseline image IDs differ or contain duplicates")
    if not records:
        raise ValueError("source run has no records")
    if len({r["image_id"] for r in records}) != len(records) or len({r["source"] for r in records}) != len(records):
        raise ValueError("duplicate input IDs or source filenames")
    if source_config.get("n_images", len(records)) != len(records):
        raise ValueError("source run is incomplete")
    if baseline_config.get("n_images", len(baseline_records)) != len(baseline_records):
        raise ValueError("independent baseline run is incomplete")
    res = source_config.get("res", 224)
    # Validate every input and mapping before creating any deliverable directories.
    prepared = []
    for record in records:
        image_id, source_name = str(record["image_id"]), record["source"]
        if not image_id.isdigit() or Path(source_name).name != source_name:
            raise ValueError("unsafe image ID or source mapping")
        source_path = images_root / source_name
        baseline_record = baselines[image_id]
        if baseline_record["source"] != source_name:
            raise ValueError(f"independent baseline source mapping differs for {image_id}")
        reference = canonical_source(source_path, res)
        for folder in (source_run, baseline_run):
            original = load_rgb(folder / f"{image_id}_originImage.png")
            if not np.array_equal(original, reference):
                raise ValueError(f"canonical source differs from the original input recorded by {folder.name}: {image_id}")
        raw_paths = {"LAO8": source_run / f"{image_id}_adv_image.png",
                     "RECON8": baseline_run / f"{image_id}_adv_image.png"}
        reset_path = source_run / f"{image_id}_reset_image.png"
        if sha256(reset_path) != sha256(raw_paths["RECON8"]):
            raise ValueError(f"raw LAO reset differs from the independently run baseline: {image_id}")
        raw = {name: load_rgb(path) for name, path in raw_paths.items()}
        reset = load_rgb(reset_path)
        for candidate in raw.values():
            check_pair(candidate, reference)
        for original_record in (record, baseline_record):
            seconds = original_record.get("seconds")
            if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"missing/invalid original run timing for {image_id}")
        prepared.append((record, baseline_record, source_path, reference, raw_paths, raw, reset_path, reset))
    out_root.mkdir(parents=True, exist_ok=True)
    metadata = {"created_at": datetime.now().astimezone().isoformat(), "source_run": str(source_run),
                "baseline_run": str(baseline_run),
                "images_root": str(images_root), "budget_pixels": budget_pixels, "n_images": len(records),
                "source_config_sha256": sha256(source_run / "run_config.json"),
                "source_records_sha256": sha256(source_run / "records.jsonl"),
                "baseline_config_sha256": sha256(baseline_run / "run_config.json"),
                "baseline_records_sha256": sha256(baseline_run / "records.jsonl"),
                "model_executed": False,
                "independent_reset_check": "raw reset PNG SHA-256 equals the independent zero-step output",
                "timing_note": "seconds is the corresponding original LAO/baseline run time, including diagnostics; projection is not a new optimization"}
    for name in ("LAO8", "RECON8"):
        folder = out_root / name
        folder.mkdir()
        original_config = source_config if name == "LAO8" else baseline_config
        config = {**original_config, **metadata, "save_dir": str(folder), "variant": name,
                  "derived_from_frozen_output": True, "source_raw_config": original_config,
                  "saved_image_budget_pixels": budget_pixels,
                  "scores_available": False}
        write_json_exclusive(folder / "run_config.json", config)
    outputs = {name: [] for name in ("LAO8", "RECON8")}
    for index, (record, baseline_record, source_path, reference, raw_paths, raw, reset_path, reset) in enumerate(prepared):
        image_id = str(record["image_id"])
        projected = {name: project_uint8(candidate, reference, budget_pixels) for name, candidate in raw.items()}
        for name in outputs:
            folder = out_root / name
            candidate = raw[name]
            result = projected[name]
            clean_saved = save_png_verified(folder / f"{image_id}_originImage.png", reference)
            saved = save_png_verified(folder / f"{image_id}_adv_image.png", result, reference, budget_pixels)
            reset_saved = None
            if name == "LAO8":
                reset_saved = save_png_verified(folder / f"{image_id}_reset_image.png",
                                                project_uint8(reset, reference, budget_pixels), reference, budget_pixels)
            original_record = record if name == "LAO8" else baseline_record
            item = {"image_id": image_id, "source": record["source"], "seconds": original_record["seconds"],
                    "clean": None, "final": None, "pgd": None, "pgd_info": {}, "history": [],
                    "scores_available": False, "source_file_sha256": sha256(source_path),
                    "canonical_source_png_sha256": clean_saved["sha256"],
                    "raw_candidate_file": str(raw_paths[name]), "raw_candidate_sha256": sha256(raw_paths[name]),
                    "raw_reset_sha256": sha256(reset_path),
                    "independent_baseline_sha256": sha256(raw_paths["RECON8"]),
                    "output_sha256": saved["sha256"], "reset_sha256": reset_saved["sha256"] if reset_saved else None,
                    "max_difference_pixels": max_difference_pixels(result, reference),
                    "projection": projection_stats(candidate, reference, result, budget_pixels),
                    "timing_source": "corresponding_original_run_including_diagnostics"}
            outputs[name].append(item)
        if outputs["LAO8"][-1]["reset_sha256"] != outputs["RECON8"][-1]["output_sha256"]:
            raise RuntimeError("projected reset PNG differs from RECON8")
        print(f"[budget export] {index + 1}/{len(records)} {image_id}: saved and verified", flush=True)
    for name, rows in outputs.items():
        with (out_root / name / "records.jsonl").open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_json_exclusive(out_root / "export_manifest.json", {**metadata, "status": "verified",
                         "variants": {name: {row["image_id"]: row["output_sha256"] for row in rows}
                                      for name, rows in outputs.items()}})
    return out_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", required=True, type=Path)
    parser.add_argument("--baseline_run", required=True, type=Path)
    parser.add_argument("--images_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--budget_pixels", type=int, default=8)
    args = parser.parse_args(argv)
    result = export_budget(args.source_run, args.baseline_run, args.images_root, args.out_root, args.budget_pixels)
    print(json.dumps({"status": "verified", "out_root": str(result), "model_executed": False}))


if __name__ == "__main__":
    main()
