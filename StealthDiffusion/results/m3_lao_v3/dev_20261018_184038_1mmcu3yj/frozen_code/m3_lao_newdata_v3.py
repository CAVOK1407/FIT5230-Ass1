"""Create the third frozen 50+50 SD1.5 holdout after v3 candidate selection.

Default: create a new directory, write its plan and real photos, then generate.
--prepare_only stops before loading a diffusion model. --generate continues only
that verified prepared directory; partial generation is never resumed or replaced.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time

from PIL import Image


HERE = Path(__file__).resolve().parent
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SEED_START = 5233000
FIRST_BATCH_ROOT = HERE / "results/m3_lao_validation/validation_20261018_154651_tm5zwx4p"
FIRST_BATCH_MANIFEST_SHA256 = "eb475c014ae6453ba6a35d14cde7c6b25d0421beaba4b833840bb83b82993737"
SECOND_BATCH_ROOT = HERE / "results/m3_lao_v2/validate_20261018_172351_7bu0ig1v"
SECOND_BATCH_MANIFEST_SHA256 = "15cfb748f7f4a69b85d9b37bce0bf5b3515e039a51563439163acde2f0a66fcc"
EXCLUDED_RGB_COUNT = 446
EXCLUDED_REAL_ID_COUNT = 210
CLASSES = (
    ("n01440764", "tench"), ("n02102040", "English springer spaniel"),
    ("n02979186", "cassette player"), ("n03000684", "chain saw"),
    ("n03028079", "church"), ("n03394916", "French horn"),
    ("n03417042", "garbage truck"), ("n03425413", "gas pump"),
    ("n03445777", "golf ball"), ("n03888257", "parachute"),
)
MODEL_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "unet/config.json", "unet/diffusion_pytorch_model.bin",
    "vae/config.json", "vae/diffusion_pytorch_model.bin",
    "text_encoder/config.json", "text_encoder/pytorch_model.bin",
    "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
    "tokenizer/vocab.json", "tokenizer/merges.txt",
    "feature_extractor/preprocessor_config.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_timestamp(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {label}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def verify_file(path_value, expected, label):
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{label} must exist before data preparation: {path}")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"{label} requires a lowercase SHA-256")
    if sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def validate_candidate_freeze(path, expected=None):
    """Verify the pre-existing selection evidence; never create or update it."""
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("--candidate_freeze must be an existing absolute file")
    digest = sha256(path)
    if expected is not None and (str(path) != expected.get("path") or digest != expected.get("sha256")):
        raise ValueError("Candidate freeze differs from the frozen generation plan")
    freeze = read_json(path)
    if not isinstance(freeze, dict) or freeze.get("status") != "frozen":
        raise ValueError("Candidate freeze status must be 'frozen'")
    created = parse_timestamp(freeze.get("created_at"), "candidate freeze created_at")
    if created > datetime.now(timezone.utc):
        raise ValueError("Candidate freeze cannot be dated in the future")
    if not isinstance(freeze.get("selected_configuration"), dict):
        raise ValueError("Candidate freeze requires selected_configuration as an object")
    evidence = freeze.get("development_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Candidate freeze requires development_evidence")
    verify_file(evidence.get("path"), evidence.get("sha256"), "Development evidence")
    sources = freeze.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Candidate freeze requires a nonempty source_sha256 object")
    for source, expected_hash in sources.items():
        verify_file(source, expected_hash, "Frozen candidate source")
    result = {"path": str(path), "sha256": digest, "created_at": freeze["created_at"],
              "selected_configuration": freeze["selected_configuration"],
              "development_evidence": evidence, "source_sha256": sources}
    if expected is not None and result != expected:
        raise ValueError("Candidate freeze metadata differs from the generation plan")
    return result


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resize224(image, *, center_crop=False):
    image = image.convert("RGB")
    if center_crop:
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    return image.resize((224, 224), Image.Resampling.LANCZOS)


def rgb_hash(image):
    if image.mode != "RGB" or image.size != (224, 224):
        raise ValueError("Novelty hashes require RGB 224x224 images")
    return hashlib.sha256(image.tobytes()).hexdigest()


def source_rgb_hash(path, *, center_crop=False):
    with Image.open(path) as image:
        return rgb_hash(resize224(image, center_crop=center_crop))


def audit_prior_batch(batch_root, manifest_sha256, seed_start, dataset):
    """Verify the immutable prior holdout and all of its actual input pixels."""
    manifest_path = batch_root / "data_manifest.json"
    verify_file(str(manifest_path), manifest_sha256, "Prior validation manifest")
    manifest = read_json(manifest_path)
    data_root = batch_root / "data"
    verify_file(str(data_root / "data_manifest.json"), manifest_sha256,
                "Prior validation manifest mirror")
    if manifest.get("status") != "verified" or manifest.get("data_root") != str(data_root):
        raise ValueError("Prior validation is not the expected verified data set")
    plan_path = data_root / "generation_plan.json"
    verify_file(str(plan_path), manifest.get("plan_sha256"), "Prior validation generation plan")
    rows, real_ids, generation_ids, all_pixels = [], set(), set(), set()
    for kind, key in (("gen", "generated"), ("real", "real")):
        entries = manifest.get(key)
        if manifest.get("n_generated" if kind == "gen" else "n_real") != 50 or not isinstance(entries, list) or len(entries) != 50:
            raise ValueError("Prior validation must contain exactly 50 generated and 50 real inputs")
        folder = data_root / kind / "test"
        if {path.name for path in folder.glob("*.png")} != {f"{index:04d}.png" for index in range(50)}:
            raise ValueError("Prior validation canonical file inventory changed")
        verify_file(str(folder / "manifest.csv"), manifest[f"{key}_manifest_sha256"],
                    f"Prior validation {kind} CSV")
        for index, entry in enumerate(entries):
            relative = f"{kind}/test/{index:04d}.png"
            if entry.get("image_id") != f"{index:04d}" or entry.get("relative_path") != relative:
                raise ValueError("Prior validation sample order or relative path changed")
            path = verify_file(str(data_root / relative), entry.get("sha256"), "Prior validation canonical input")
            with Image.open(path) as image:
                pixel_hash = rgb_hash(image)
            if pixel_hash != entry.get("rgb224_sha256") or pixel_hash in all_pixels:
                raise ValueError("Prior validation pixel provenance or uniqueness failed")
            all_pixels.add(pixel_hash)
            if kind == "real":
                source = verify_file(entry.get("source"), entry.get("source_sha256"), "Prior validation real source")
                if source.name != entry.get("source_id") or source_rgb_hash(source, center_crop=True) != pixel_hash:
                    raise ValueError("Prior validation real source mapping changed")
                real_ids.add(source.name)
            else:
                if entry.get("model") != MODEL_ID or entry.get("model_revision") != MODEL_REVISION or entry.get("seed") != seed_start + index:
                    raise ValueError("Prior validation generation identity changed")
                if entry.get("source512") != f"gen512/{index:04d}.png":
                    raise ValueError("Prior validation full-resolution path changed")
                source = verify_file(str(data_root / entry["source512"]), entry.get("source512_sha256"),
                                     "Prior validation 512px source")
                if source_rgb_hash(source) != pixel_hash:
                    raise ValueError("Prior validation 512px source does not reproduce canonical pixels")
                generation_ids.add((MODEL_ID, entry["seed"]))
            rows.append({"path": str(path), "kind": kind, "dataset": dataset,
                         "split": "test", "sha256": entry["sha256"],
                         "rgb224_sha256": pixel_hash, "source": entry})
    evidence = {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256,
                "data_root": str(data_root), "plan_path": str(plan_path),
                "plan_sha256": manifest["plan_sha256"], "n_generated": 50, "n_real": 50,
                "canonical_files_verified": 100}
    return rows, real_ids, generation_ids, evidence


def audit_first_batch():
    return audit_prior_batch(FIRST_BATCH_ROOT, FIRST_BATCH_MANIFEST_SHA256, 5231000, "validation_v1")


def audit_second_batch():
    """Bind the second batch to actual inputs and its pre-generation freeze."""
    rows, real_ids, generation_ids, evidence = audit_prior_batch(
        SECOND_BATCH_ROOT, SECOND_BATCH_MANIFEST_SHA256, 5232000, "validation_v2")
    manifest = read_json(Path(evidence["manifest_path"]))
    plan_path = Path(evidence["plan_path"])
    plan = read_json(plan_path)
    if manifest.get("generation_plan") != plan or manifest.get("old_inputs") != plan["old_inputs"]:
        raise ValueError("Second validation manifest does not bind its full frozen data plan")
    reference = plan.get("candidate_freeze")
    if not isinstance(reference, dict) or reference != manifest.get("candidate_freeze"):
        raise ValueError("Second validation candidate-freeze references differ")
    frozen = validate_candidate_freeze(Path(reference["path"]), reference)
    if parse_timestamp(frozen["created_at"], "second candidate freeze") > parse_timestamp(plan["created_at"], "second data plan"):
        raise ValueError("Second validation data preceded its candidate freeze")
    state_path = plan_path.parent / "generation_state.json"
    state = read_json(state_path)
    if (state.get("status") != "verified" or state.get("completed_images") != 50
            or state.get("plan_sha256") != evidence["plan_sha256"]
            or state.get("data_manifest_sha256") != SECOND_BATCH_MANIFEST_SHA256):
        raise ValueError("Second validation did not finish with the expected data provenance")
    records_path = plan_path.parent / "generation_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if records != manifest["generated"]:
        raise ValueError("Second validation generation log differs from its fixed manifest")
    evidence.update(candidate_freeze=frozen, generation_state_sha256=sha256(state_path),
                    generation_records_sha256=sha256(records_path))
    return rows, real_ids, generation_ids, evidence


def audit_old_inputs():
    """Build the exclusion set from every old canonical split and saved origin."""
    rows, manifests, real_ids, generation_ids = [], {}, set(), set()
    for dataset in ("data", "data15"):
        for kind in ("gen", "real"):
            for split in ("dev", "test"):
                folder = HERE / dataset / kind / split
                manifest = folder / "manifest.csv"
                manifests[str(manifest)] = sha256(manifest)
                entries = read_csv(manifest)
                expected = {entry["image_id"] + ".png" for entry in entries}
                if expected != {path.name for path in folder.glob("*.png")}:
                    raise ValueError(f"Old source manifest/file mismatch: {folder}")
                for entry in entries:
                    path = folder / (entry["image_id"] + ".png")
                    if kind == "real":
                        real_ids.add(Path(entry["source"]).name)
                    else:
                        generation_ids.add((entry["model"], int(entry["seed"])))
                    rows.append({"path": str(path), "kind": kind, "dataset": dataset,
                                 "split": split, "sha256": sha256(path),
                                 "rgb224_sha256": source_rgb_hash(path), "source": entry})
    original_hashes = {row["rgb224_sha256"] for row in rows}
    if len(original_hashes) != 246 or len(real_ids) != 110:
        raise ValueError("Original canonical exclusion inventory changed; expected 246 RGB contents and 110 real source IDs")
    first_rows, first_real_ids, first_generation_ids, first_evidence = audit_first_batch()
    prior_exclusions = read_json(Path(first_evidence["plan_path"]))["old_inputs"]
    if prior_exclusions["manifest_sha256"] != manifests or prior_exclusions["canonical_images"] != rows:
        raise ValueError("Original inputs differ from the first validation's frozen exclusion audit")
    if original_hashes.intersection(row["rgb224_sha256"] for row in first_rows) or real_ids.intersection(first_real_ids):
        raise ValueError("First validation unexpectedly overlaps the original inputs")
    rows.extend(first_rows)
    real_ids.update(first_real_ids)
    generation_ids.update(first_generation_ids)
    hashes = {row["rgb224_sha256"] for row in rows}
    if len(hashes) != 346 or len(real_ids) != 160:
        raise ValueError("First validation exclusion union has an unexpected size")
    second_rows, second_real_ids, second_generation_ids, second_evidence = audit_second_batch()
    second_exclusions = read_json(Path(second_evidence["plan_path"]))["old_inputs"]
    if (second_exclusions["manifest_sha256"] != manifests or second_exclusions["canonical_images"] != rows
            or set(second_exclusions["old_rgb224_sha256"]) != hashes
            or set(second_exclusions["old_real_source_ids"]) != real_ids):
        raise ValueError("Historical inputs differ from the second validation's frozen exclusion audit")
    if hashes.intersection(row["rgb224_sha256"] for row in second_rows) or real_ids.intersection(second_real_ids):
        raise ValueError("Second validation unexpectedly overlaps earlier inputs")
    rows.extend(second_rows)
    real_ids.update(second_real_ids)
    generation_ids.update(second_generation_ids)
    hashes = {row["rgb224_sha256"] for row in rows}
    if len(hashes) != EXCLUDED_RGB_COUNT or len(real_ids) != EXCLUDED_REAL_ID_COUNT or len(generation_ids) != 236:
        raise ValueError("V3 exclusions require 446 unique RGB contents, 210 real IDs and 236 generation IDs")
    old_generated = {row["rgb224_sha256"] for row in rows if row["kind"] == "gen"}
    origins = sorted((HERE / "results").rglob("*_originImage.png"))
    unique_origins = {}
    for path in origins:
        unique_origins.setdefault(sha256(path), path)
    for path in unique_origins.values():
        if source_rgb_hash(path) not in old_generated:
            raise ValueError(f"An old experiment used an additional source image; extend exclusions: {path}")
    return {"canonical_images": rows, "manifest_sha256": manifests,
            "prior_validation_batches": [first_evidence, second_evidence],
            "original_unique_rgb224": 246, "prior_validation_unique_rgb224": 200,
            "old_rgb224_sha256": sorted(hashes), "old_real_source_ids": sorted(real_ids),
            "old_generation_ids": [list(item) for item in sorted(generation_ids)],
            "result_origin_files_audited": len(origins),
            "unique_origin_files_audited": len(unique_origins),
            "additional_result_origin_contents": 0}


def plan_real_images(exclusions):
    root = HERE / "data15/imagenette2-320/val"
    old_ids = set(exclusions["old_real_source_ids"])
    old_hashes = set(exclusions["old_rgb224_sha256"])
    selected, new_hashes = [], set()
    for wnid, name in CLASSES:
        count = 0
        candidates = sorted(path for path in (root / wnid).iterdir()
                            if path.suffix.lower() in (".jpeg", ".jpg", ".png"))
        for path in candidates:
            if path.name in old_ids:
                continue
            digest = source_rgb_hash(path, center_crop=True)
            if digest in old_hashes:
                continue
            if digest in new_hashes:
                raise ValueError(f"Duplicate new real image; no replacement is permitted: {path}")
            new_hashes.add(digest)
            selected.append({"image_id": f"{len(selected):04d}", "class": name,
                             "wnid": wnid, "source": str(path), "source_id": path.name,
                             "source_sha256": sha256(path), "rgb224_sha256": digest})
            count += 1
            if count == 5:
                break
        if count != 5:
            raise ValueError(f"Fewer than five novel readable images for class {wnid}")
    return selected


def model_provenance(model_path):
    if not model_path.is_absolute() or not model_path.is_dir():
        raise ValueError("--model_path must be an existing absolute local snapshot directory")
    if model_path.name != MODEL_REVISION:
        raise ValueError(f"Expected the pinned SD1.5 snapshot {MODEL_REVISION}")
    files = {name: sha256(model_path / name) for name in MODEL_FILES}
    first_manifest_path = FIRST_BATCH_ROOT / "data_manifest.json"
    verify_file(str(first_manifest_path), FIRST_BATCH_MANIFEST_SHA256, "First validation manifest")
    first_manifest = read_json(first_manifest_path)
    first_plan_path = FIRST_BATCH_ROOT / "data/generation_plan.json"
    verify_file(str(first_plan_path), first_manifest["plan_sha256"], "First validation generation plan")
    if files != read_json(first_plan_path)["model"]["file_sha256"]:
        raise ValueError("SD1.5 model files differ from the first validation's frozen model")
    scheduler = read_json(model_path / "scheduler/scheduler_config.json")
    if scheduler.get("_class_name") != "PNDMScheduler":
        raise ValueError("The fixed generation protocol requires PNDMScheduler")
    return {"id": MODEL_ID, "revision": MODEL_REVISION, "path": str(model_path),
            "file_sha256": files, "scheduler": scheduler}


def prepare(out, model_path, device, candidate_freeze):
    frozen = validate_candidate_freeze(candidate_freeze)
    if out.exists() or (out.parent / "data_manifest.json").exists():
        raise FileExistsError("Use a new output directory and a parent without data_manifest.json")
    model = model_provenance(model_path)
    exclusions = audit_old_inputs()
    real = plan_real_images(exclusions)
    planned = [{"image_id": f"{index:04d}", "prompt": f"a photo of a {CLASSES[index % 10][1]}",
                "seed": SEED_START + index, "class": CLASSES[index % 10][1],
                "wnid": CLASSES[index % 10][0]} for index in range(50)]
    old_ids = {tuple(item) for item in exclusions["old_generation_ids"]}
    if any((MODEL_ID, row["seed"]) in old_ids for row in planned):
        raise ValueError("New generation source IDs overlap an old manifest")
    plan = {"schema_version": 3, "created_at": now(), "n_generated": 50, "n_real": 50,
            "candidate_freeze": frozen,
            "method": "fresh_same_prompt_family_sd15_seeds_without_outcome_selection",
            "model": model, "device": device, "dtype": "float32", "seed_generator_device": "cpu",
            "batch_size": 1, "attention_slicing": "auto", "steps": 30, "guidance_scale": 7.5,
            "negative_prompt": None, "eta": 0.0, "generated_size": [512, 512],
            "saved_size": [224, 224], "generated_resize": "RGB then PIL LANCZOS; no crop",
            "real_resize": "RGB center-square crop then PIL LANCZOS",
            "safety_checker": "disabled to match original data generation; no content/outcome filtering",
            "runtime": {key: version(key) for key in ("torch", "diffusers", "transformers", "numpy", "Pillow")},
            "utility_sha256": sha256(Path(__file__).resolve()),
            "generated_samples": planned, "real_samples": real,
            "old_inputs": exclusions, "replacement_rule": "Abort on failure or duplicate; never replace samples",
            "scope": "New source images for a fixed held-out evaluation; same classes/prompts and generator family as development data."}
    validate_candidate_freeze(candidate_freeze, frozen)
    out.mkdir(parents=True, exist_ok=False)
    # This plan is saved before any image is written and before a model is loaded.
    write_json(out / "generation_plan.json", plan)
    write_json(out / "generation_state.json", {"status": "preparing_real", "created_at": now()})
    real_folder = out / "real/test"
    real_folder.mkdir(parents=True)
    rows = []
    for sample in real:
        source = Path(sample["source"])
        if sha256(source) != sample["source_sha256"]:
            raise ValueError(f"Real source changed after planning: {source}")
        target = real_folder / (sample["image_id"] + ".png")
        with Image.open(source) as image:
            resize224(image, center_crop=True).save(target, format="PNG")
        if source_rgb_hash(target) != sample["rgb224_sha256"]:
            raise ValueError("Saved real image differs from the planned RGB pixels")
        rows.append({**sample, "sha256": sha256(target), "relative_path": str(target.relative_to(out))})
    write_csv(real_folder / "manifest.csv", rows)
    write_json(out / "generation_state.json", {"status": "prepared", "prepared_at": now(),
               "plan_sha256": sha256(out / "generation_plan.json"),
               "real_manifest_sha256": sha256(real_folder / "manifest.csv")})
    print(f"Prepared immutable generation plan and 50 novel real images: {out}", flush=True)


def generate(out, model_path, device, candidate_freeze):
    validate_candidate_freeze(candidate_freeze)
    plan_path = out / "generation_plan.json"
    state = read_json(out / "generation_state.json")
    if state.get("status") != "prepared" or state["plan_sha256"] != sha256(plan_path):
        raise ValueError("Generation requires an unchanged prepared plan; partial runs cannot resume")
    plan = read_json(plan_path)
    frozen = validate_candidate_freeze(candidate_freeze, plan.get("candidate_freeze", {}))
    if parse_timestamp(frozen["created_at"], "candidate freeze created_at") > parse_timestamp(plan["created_at"], "generation plan created_at"):
        raise ValueError("Candidate selection must be frozen before data preparation")
    if str(model_path) != plan["model"]["path"] or device != plan["device"]:
        raise ValueError("Model path/device differ from the frozen generation plan")
    if sha256(Path(__file__).resolve()) != plan["utility_sha256"]:
        raise ValueError("The data utility changed after the plan was written")
    if any(version(name) != expected for name, expected in plan["runtime"].items()):
        raise ValueError("A recorded library version changed after planning")
    for name, expected in plan["model"]["file_sha256"].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f"Model file changed after planning: {name}")
    real_manifest = out / "real/test/manifest.csv"
    if sha256(real_manifest) != state["real_manifest_sha256"]:
        raise ValueError("Prepared real manifest changed")
    real_rows = read_csv(real_manifest)
    new_hashes = set()
    for row in real_rows:
        path = out / row["relative_path"]
        if sha256(path) != row["sha256"] or source_rgb_hash(path) != row["rgb224_sha256"]:
            raise ValueError("Prepared real image changed")
        new_hashes.add(row["rgb224_sha256"])
    old_hashes = set(plan["old_inputs"]["old_rgb224_sha256"])
    if len(old_hashes) != EXCLUDED_RGB_COUNT or len(plan["old_inputs"]["old_real_source_ids"]) != EXCLUDED_REAL_ID_COUNT:
        raise ValueError("Prepared V3 exclusion inventory is invalid")
    for path, expected in plan["old_inputs"]["manifest_sha256"].items():
        verify_file(path, expected, "Original input manifest")
    prior = plan["old_inputs"]["prior_validation_batches"]
    if len(prior) != 2 or [row["manifest_sha256"] for row in prior] != [FIRST_BATCH_MANIFEST_SHA256, SECOND_BATCH_MANIFEST_SHA256]:
        raise ValueError("Prepared V3 prior-validation exclusion bindings are invalid")
    for evidence in prior:
        verify_file(evidence["manifest_path"], evidence["manifest_sha256"], "Prior validation manifest")
        verify_file(str(Path(evidence["data_root"]) / "data_manifest.json"), evidence["manifest_sha256"],
                    "Prior validation manifest mirror")
        verify_file(evidence["plan_path"], evidence["plan_sha256"], "Prior validation generation plan")
    second_frozen = prior[1]["candidate_freeze"]
    validate_candidate_freeze(Path(second_frozen["path"]), second_frozen)
    second_root = Path(prior[1]["data_root"])
    verify_file(str(second_root / "generation_state.json"), prior[1]["generation_state_sha256"], "Second generation state")
    verify_file(str(second_root / "generation_records.jsonl"), prior[1]["generation_records_sha256"], "Second generation records")
    for row in plan["old_inputs"]["canonical_images"]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("An old canonical input changed after the novelty audit")
    full = out / "gen512"
    small = out / "gen/test"
    if full.exists() or (out / "gen").exists() or (out.parent / "data_manifest.json").exists():
        raise FileExistsError("Generated outputs or parent manifest already exist; partial runs are retained")
    state.update(status="generating", started_at=now(), completed_images=0)
    write_json(out / "generation_state.json", state)
    rows = []
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        import diffusers
        import numpy as np
        from diffusers import PNDMScheduler, StableDiffusionPipeline
        if torch.__version__ != plan["runtime"]["torch"] or diffusers.__version__ != plan["runtime"]["diffusers"]:
            raise ValueError("Torch/diffusers versions changed after planning")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable; no implicit device substitution")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        pipe = StableDiffusionPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.float32, safety_checker=None,
            requires_safety_checker=False, local_files_only=True).to(device)
        pipe.scheduler = PNDMScheduler.from_config(plan["model"]["scheduler"])
        for module in (pipe.unet, pipe.vae, pipe.text_encoder):
            module.eval().requires_grad_(False)
        pipe.enable_attention_slicing("auto")
        pipe.set_progress_bar_config(disable=True)
        write_json(out / "actual_runtime.json", {"device": str(pipe.device), "dtype": str(pipe.unet.dtype),
                   "scheduler_class": type(pipe.scheduler).__name__, "scheduler_config": dict(pipe.scheduler.config),
                   "attention_slicing": "auto", "seed_generator_device": "cpu", "model_training": False})
        full.mkdir()
        small.mkdir(parents=True)
        def check_latents(step, timestep, latents):
            if not bool(torch.isfinite(latents).all().item()):
                raise FloatingPointError(f"Non-finite generation latents at step {step}")
        for sample in plan["generated_samples"]:
            start = time.monotonic()
            generator = torch.Generator(device="cpu").manual_seed(sample["seed"])
            with torch.inference_mode():
                pixels = pipe(sample["prompt"], num_inference_steps=30, guidance_scale=7.5,
                              height=512, width=512, generator=generator, eta=0.0,
                              output_type="np", callback=check_latents, callback_steps=1).images
            if not np.isfinite(pixels).all():
                raise FloatingPointError("Non-finite decoded generation output")
            # Use the pipeline's own PIL conversion, exactly as output_type='pil'.
            image = pipe.numpy_to_pil(pixels)[0].convert("RGB")
            if image.size != (512, 512):
                raise ValueError("Unexpected generated image dimensions")
            target512 = full / (sample["image_id"] + ".png")
            target = small / target512.name
            # Retain the failed/duplicate candidate as evidence; do not replace it.
            image.save(target512, format="PNG")
            resized = resize224(image)
            resized.save(target, format="PNG")
            pixel_hash = rgb_hash(resized)
            if pixel_hash in old_hashes or pixel_hash in new_hashes:
                raise ValueError(f"Generated duplicate at seed {sample['seed']}; no replacement permitted")
            new_hashes.add(pixel_hash)
            row = {**sample, "model": MODEL_ID, "model_revision": MODEL_REVISION,
                   "steps": 30, "guidance": 7.5, "sha256": sha256(target),
                   "rgb224_sha256": pixel_hash, "source512": str(target512.relative_to(out)),
                   "source512_sha256": sha256(target512), "relative_path": str(target.relative_to(out)),
                   "seconds": round(time.monotonic() - start, 3)}
            rows.append(row)
            with (out / "generation_records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            state.update(completed_images=len(rows), updated_at=now())
            write_json(out / "generation_state.json", state)
            print(f"Generated {len(rows)}/50 | seed={sample['seed']} | {row['seconds']:.1f}s", flush=True)
        write_csv(small / "manifest.csv", rows)
        validate_candidate_freeze(candidate_freeze, frozen)
        manifest = {"schema_version": 3, "status": "verified", "created_at": now(),
                    "candidate_freeze": frozen, "old_inputs": plan["old_inputs"],
                    "generation_plan": plan,
                    "data_root": str(out), "plan_sha256": sha256(plan_path), "n_generated": len(rows),
                    "n_real": len(real_rows), "generated": rows, "real": real_rows,
                    "generated_manifest_sha256": sha256(small / "manifest.csv"),
                    "real_manifest_sha256": sha256(real_manifest),
                    "novelty": {"old_canonical_unique_rgb224": len(old_hashes),
                                "new_unique_rgb224": len(new_hashes), "old_overlap": 0,
                                "new_internal_duplicates": 0, "source_seed_overlap": 0,
                                "replacement_count": 0, "detector_scores_used_for_selection": False}}
        if len(rows) != 50 or len(real_rows) != 50 or len(new_hashes) != 100:
            raise ValueError("Final sample counts or uniqueness checks failed")
        write_json(out / "data_manifest.json", manifest)
        write_json(out.parent / "data_manifest.json", manifest)
        state.update(status="verified", finished_at=now(), data_manifest_sha256=sha256(out / "data_manifest.json"))
        write_json(out / "generation_state.json", state)
        print(f"Complete: 50 new generated + 50 new real; zero old-content overlap. {out / 'data_manifest.json'}", flush=True)
    except BaseException as exc:
        state.update(status="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}")
        write_json(out / "generation_state.json", state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="New data directory, as in v2")
    target.add_argument("--root", type=Path, help="Run directory; writes to ROOT/data")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--n", type=int, choices=(50,), default=50)
    parser.add_argument("--candidate_freeze", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare_only", action="store_true")
    mode.add_argument("--generate", action="store_true")
    args = parser.parse_args(argv)
    if not args.model_path.is_absolute():
        parser.error("--model_path must be absolute")
    if not args.candidate_freeze.is_absolute():
        parser.error("--candidate_freeze must be absolute")
    out = (args.out if args.out is not None else args.root / "data").resolve()
    model_path = args.model_path.resolve()
    if not args.generate:
        prepare(out, model_path, args.device, args.candidate_freeze.resolve())
    if not args.prepare_only:
        generate(out, model_path, args.device, args.candidate_freeze.resolve())


if __name__ == "__main__":
    main()
