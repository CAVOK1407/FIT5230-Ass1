"""SFA course extension: frozen E/R optimization, independent saved-PNG E/R/D/S evaluation.

No detector training, diffusion inference, or peer-system attack is performed.
Run with the same archived Python 3.10 ML environment as M3. The original course
bundle is read-only; this extension writes only a fresh --out directory.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

EXPECTED_BUNDLE_MANIFEST_SHA256 = "d4fc6edf2f8f445a5bbe2577e4e499a2c83c9dd942b94982966562aebc7270a4"
METHODS = ("PGD", "SFA", "SPEC")
DETECTORS = ("E", "R", "D", "S")
CONDITIONS = ("none", "jpeg75", "resize05")
CHECKPOINT_NAMES = {"E": "efficientnet-b0.pth", "R": "resnet50.pth", "D": "deit.pth", "S": "swin-t.pth"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_bundle(bundle):
    """Verify the exact original resource package before importing its model code."""
    manifest_path = bundle / "manifest.json"
    if sha256(manifest_path) != EXPECTED_BUNDLE_MANIFEST_SHA256:
        raise ValueError("This extension requires the original, unchanged 18 October M3 course bundle")
    manifest = json.loads(manifest_path.read_text())
    declared = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        p = bundle / relative
        if relative.is_absolute() or ".." in relative.parts or not p.resolve().is_relative_to(bundle):
            raise ValueError(f"Unsafe bundle path: {relative}")
        if row["path"] in declared or not p.is_file():
            raise ValueError(f"Missing or duplicate bundle file: {relative}")
        if p.stat().st_size != row["bytes"] or sha256(p) != row["sha256"]:
            raise ValueError(f"Bundle hash/size mismatch: {relative}")
        declared.add(row["path"])
    for subtree in (bundle / "src", bundle / "data"):
        for p in subtree.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.relative_to(bundle).as_posix() not in declared:
                raise ValueError(f"Unmanifested source or data file: {p}")
    return {"source_snapshot": manifest["source_snapshot"], "manifest_sha256": sha256(manifest_path),
            "verified_files": len(declared), "files": manifest["files"]}


def verify_weights(bundle, weights_root):
    checkpoint_dir = weights_root / "checkpoints" if (weights_root / "checkpoints").is_dir() else weights_root
    expected = {r["path"]: r for r in json.loads((bundle / "resources.json").read_text())["weights"]}
    rows = []
    for name, filename in CHECKPOINT_NAMES.items():
        p = checkpoint_dir / filename
        spec = expected[f"checkpoints/{filename}"]
        if not p.is_file() or p.stat().st_size != spec["bytes"] or sha256(p) != spec["sha256"]:
            raise ValueError(f"Frozen checkpoint missing or different: {p}")
        rows.append({"detector": name, "filename": filename, "bytes": spec["bytes"], "sha256": spec["sha256"]})
    return checkpoint_dir, rows


def select_data(bundle, limit, real_limit):
    from PIL import Image
    data = bundle / "data"
    selected = {}
    metadata = []
    all_rgb = {"dev": set(), "test": set()}
    for kind in ("gen", "real"):
        for split in ("dev", "test"):
            paths = sorted((data / kind / split).glob("*.png"))
            expected = 10 if split == "dev" else 100
            if len(paths) != expected:
                raise ValueError(f"Expected {expected} inputs in {kind}/{split}, got {len(paths)}")
            for p in paths:
                with Image.open(p) as im:
                    if im.size != (224, 224) or im.mode != "RGB":
                        raise ValueError(f"Canonical input must be RGB 224x224: {p}")
                    digest = hashlib.sha256(im.tobytes()).hexdigest()
                all_rgb[split].add(digest)
                metadata.append({"kind": kind, "split": split, "filename": p.name, "sha256": sha256(p), "rgb_sha256": digest})
            selected[kind, split] = paths
    if all_rgb["dev"] & all_rgb["test"]:
        raise ValueError("Decoded RGB dev/test overlap")
    for kind, column in (("gen", "seed"), ("real", "source")):
        values = {}
        for split in ("dev", "test"):
            with (data / kind / split / "manifest.csv").open() as f:
                values[split] = {r[column] for r in csv.DictReader(f)}
        if values["dev"] & values["test"]:
            raise ValueError(f"Provenance overlap in {kind}/{column}")
    if limit > 100 or real_limit > 100:
        raise ValueError("At most 100 test images exist")
    return selected["gen", "test"][:limit], selected["real", "dev"], selected["real", "test"][:real_limit], metadata


def transformed(path, condition):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as opened:
        im = opened.convert("RGB")
    if condition == "jpeg75":
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            im = compressed.convert("RGB")
    elif condition == "resize05":
        im = im.resize((112, 112), Image.Resampling.LANCZOS).resize((224, 224), Image.Resampling.BICUBIC)
    elif condition != "none":
        raise ValueError(condition)
    return torch.from_numpy(np.array(im, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0)


def ssim_rgb(base, adv):
    """11x11 Gaussian, sigma1.5, valid crop, RGB mean, range[0,1]."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    x = base[0].detach().cpu().numpy().astype(np.float64)
    y = adv[0].detach().cpu().numpy().astype(np.float64)
    mx = gaussian_filter(x, sigma=(0, 1.5, 1.5), truncate=10 / 3)
    my = gaussian_filter(y, sigma=(0, 1.5, 1.5), truncate=10 / 3)
    vx = gaussian_filter(x*x, sigma=(0, 1.5, 1.5), truncate=10 / 3) - mx*mx
    vy = gaussian_filter(y*y, sigma=(0, 1.5, 1.5), truncate=10 / 3) - my*my
    cov = gaussian_filter(x*y, sigma=(0, 1.5, 1.5), truncate=10 / 3) - mx*my
    score = ((2*mx*my+.01**2)*(2*cov+.03**2)) / ((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return float(score[:, 5:-5, 5:-5].mean())


def run(args):
    import numpy as np
    import torch
    core_dir = Path(args.core_dir).resolve() if args.core_dir else Path(__file__).resolve().parent
    sys.path.insert(0, str(core_dir))
    import m3_sfa_core as core
    bundle, out = Path(args.bundle).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    if out.is_relative_to(bundle):
        raise ValueError("Output must not be inside the immutable bundle")
    checked = verify_bundle(bundle)
    checkpoint_dir, model_hashes = verify_weights(bundle, Path(args.weights_root).resolve())
    inputs, references, reals, dataset_manifest = select_data(bundle, args.limit, args.real_limit)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    out.mkdir(parents=True)
    started = time.monotonic()
    config = {"status": "running", "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "scope": "Own frozen forensic-detector benchmark; not a PreDecodeGuard submission or independent peer engagement",
              "protocol": "Predeclared lambda30 from teammate mechanism study; no test-set selection or tuning",
              "methods": list(METHODS), "surrogates": ["E", "R"], "held_out_from_optimization": ["D", "S"],
              "ai_class": 0, "real_class": 1, "decision": "argmax([0,real_margin]); margin<=0 is AI",
              "lambda": 30.0, "budget_levels": 8, "budget": 8/255, "budget_measurement": "Integer RGB saved PNG differences against canonical original PNG; maximum 8 levels",
              "steps": args.steps, "alpha": 2/255, "seed": 5230, "random_start": False,
              "generated_inputs": len(inputs), "real_references": len(references), "real_evaluation_inputs": len(reals),
              "conditions": list(CONDITIONS), "spectrum_reference": "all10 real/dev only; no test references",
              "selection": "Lexicographically first fixed gen/test files; no outcome selection",
              "evaluation": "Separate post-attack model instances, all predictions recomputed from saved/reloaded PNGs",
              "lpips": "not computed in this SFA extension", "ssim": "RGB Gaussian11 sigma1.5 valid crop, data_range1; implementation explicit in runner",
              "runtime": {"python": sys.executable, "python_version": sys.version, "torch": torch.__version__, "device": device,
                          "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
                          "spectrum_backend": "CPU FFT with differentiable device copies" if device == "mps" else "native " + device},
              "source_snapshot": checked["source_snapshot"], "bundle_manifest_sha256": checked["manifest_sha256"],
              "addon_sources": {"runner": sha256(Path(__file__)), "core": sha256(Path(core.__file__))},
              "dataset_manifest": "dataset_manifest.json", "model_manifest": "model_manifest.json"}
    json_write(out / "config.json", config)
    json_write(out / "dataset_manifest.json", {"all_bundle_images": dataset_manifest,
             "attack_inputs": [p.relative_to(bundle).as_posix() for p in inputs],
             "spectrum_reference_inputs": [p.relative_to(bundle).as_posix() for p in references],
             "real_test_inputs": [p.relative_to(bundle).as_posix() for p in reals], "decoded_rgb_dev_test_overlap": 0})
    json_write(out / "model_manifest.json", model_hashes)
    json_write(out / "source_manifest.json", {"bundle": checked, "addon": config["addon_sources"]})
    target = core.radial_logpower(torch.cat([core.load_png(p) for p in references])).mean(0).detach()
    json_write(out / "spectrum_target.json", {"source": "real/dev", "count": 10, "values": target.cpu().tolist()})
    random.seed(5230); np.random.seed(5230); torch.manual_seed(5230)
    if device == "cuda":
        torch.cuda.manual_seed_all(5230)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    previous_cwd = Path.cwd()
    all_records, png_records = [], []
    try:
        with tempfile.TemporaryDirectory(prefix="m3_sfa_models_") as temporary:
            runtime = Path(temporary)
            for name in ("other_attacks.py", "network.py", "m2_device.py"):
                shutil.copyfile(bundle / "src" / name, runtime / name)
            (runtime / "checkpoints").symlink_to(checkpoint_dir, target_is_directory=True)
            sys.path.insert(0, str(runtime)); os.chdir(runtime)
            import m2_device
            m2_device.enable(device)
            import other_attacks
            mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
            std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
            def make_model(name):
                model = other_attacks.model_selection(name).eval()
                model.requires_grad_(False)
                if any(p.requires_grad for p in model.parameters()):
                    raise AssertionError("Detector parameters not frozen")
                return model
            def clear_memory():
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()
            surrogates = {name: make_model(name) for name in ("E", "R")}
            def logits_fn(x, names):
                if not set(names).issubset(surrogates):
                    raise ValueError("Only E/R may participate in optimization")
                return {name: surrogates[name]((x - mean) / std) for name in names}
            probe = core.load_png(inputs[0]).to(device).requires_grad_(True)
            probe_logits = logits_fn(probe, ("E", "R"))
            selfcheck = {}
            for name, logits in probe_logits.items():
                if logits.shape != (1, 2) or not torch.isfinite(logits).all() or not torch.equal(logits[:, 0], torch.zeros_like(logits[:, 0])):
                    raise AssertionError(f"Bad class/logit contract: {name}")
                grad = torch.autograd.grad(logits[:, 1].sum(), probe, retain_graph=True)[0]
                if not torch.isfinite(grad).all() or not bool(grad.abs().max() > 0):
                    raise AssertionError(f"Invalid input gradient: {name}")
                selfcheck[name] = {"shape": list(logits.shape), "class0_zero": True, "input_gradient_finite_nonzero": True,
                                   "first_input_real_margin": float(logits[0, 1].detach().cpu())}
            json_write(out / "adapter_selfcheck.json", selfcheck)
            del probe_logits, logits, grad, probe
            for method in METHODS:
                directory = out / method; directory.mkdir()
                records = []
                for index, source in enumerate(inputs):
                    image_id = source.stem
                    baseline = core.load_png(source)
                    original_path = directory / f"{image_id}_originImage.png"
                    shutil.copyfile(source, original_path)
                    before = time.monotonic()
                    result = core.attack(baseline.to(device), logits_fn, target, mode=method,
                                         surrogate_names=("E", "R"), ai_class=0, lam=30.0,
                                         budget_levels=8, alpha=2/255, steps=args.steps, seed=5230+index)
                    saved, artifacts = core.save_png_batch(result.images, baseline, [directory / f"{image_id}_adv_image.png"], budget_levels=8)
                    quality = core.image_quality(baseline, saved)[0]
                    # JSON forbids infinity; exact identical images have PSNR=null plus an explicit flag.
                    quality["identical_pixels"] = bool(torch.equal(core.quantize_rgb(baseline), core.quantize_rgb(saved)))
                    for key, value in list(quality.items()):
                        if isinstance(value, float) and not math.isfinite(value):
                            quality[key] = None
                    quality["ssim"] = ssim_rgb(baseline, saved)
                    with torch.no_grad():
                        quality["spectral_mse_before"] = float(((core.radial_logpower(baseline)-target)**2).mean())
                        quality["spectral_mse_after"] = float(((core.radial_logpower(saved)-target)**2).mean())
                    trace_path = directory / f"{image_id}_trace.json"
                    json_write(trace_path, {"config": result.config, "trace": result.trace})
                    record = {"method": method, "image_id": image_id, "source": source.name, "source_sha256": sha256(source),
                              "steps": args.steps, "seed": 5230+index, "seconds": time.monotonic()-before,
                              "original_png": original_path.relative_to(out).as_posix(),
                              "adversarial_png": (directory / f"{image_id}_adv_image.png").relative_to(out).as_posix(),
                              "adversarial_sha256": artifacts[0]["sha256"], **quality}
                    records.append(record); all_records.append(record)
                    png_records.extend(artifacts)
                    with (directory / "records.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    print(f"{method} {index+1}/{len(inputs)} {image_id} saved, L∞={quality['linf_levels']}/255", flush=True)
                    del result, saved
                    clear_memory()
                json_write(directory / "run_config.json", {**config, "method": method})
            # Destroy attack-time models; evaluation loads each detector independently.
            del surrogates
            clear_memory()
            per_image, real_predictions, summary = [], [], []
            record_index = {(r["method"], r["image_id"]): r for r in all_records}
            with torch.no_grad():
                for name in DETECTORS:
                    det = make_model(name)
                    def score_path(path, condition):
                        tensor = transformed(path, condition).to(device)
                        scores = det((tensor-mean)/std)
                        if scores.shape != (1, 2) or not torch.isfinite(scores).all():
                            raise ValueError(f"Invalid saved-file logits: {path}")
                        return {"pred": int(scores.argmax(1).item()), "ai_margin": float((scores[0, 0]-scores[0, 1]).cpu())}
                    for condition in CONDITIONS:
                        real_scores = []
                        for source in reals:
                            value = score_path(source, condition); real_scores.append(value["ai_margin"])
                            real_predictions.append({"detector": name, "condition": condition, "image_id": source.stem, **value})
                        condition_real = [r for r in real_predictions if r["detector"] == name and r["condition"] == condition]
                        clean = {source.stem: score_path(out / "PGD" / f"{source.stem}_originImage.png", condition) for source in inputs}
                        for method in METHODS:
                            group = []
                            for source in inputs:
                                record = record_index[method, source.stem]
                                value = score_path(out / record["adversarial_png"], condition)
                                original = clean[source.stem]
                                row = {"method": method, "detector": name, "condition": condition, "image_id": source.stem,
                                       "clean_pred": original["pred"], "adv_pred": value["pred"],
                                       "clean_ai_margin": original["ai_margin"], "adv_ai_margin": value["ai_margin"],
                                       "eligible": int(original["pred"] == 0), "flip": int(original["pred"] == 0 and value["pred"] == 1),
                                       "linf_levels": record["linf_levels"], "psnr_db": record["psnr_db"], "ssim": record["ssim"],
                                       "spectral_mse_after": record["spectral_mse_after"]}
                                per_image.append(row); group.append(row)
                            eligible = sum(r["eligible"] for r in group); flips = sum(r["flip"] for r in group)
                            summary.append({"method": method, "detector": name, "condition": condition, "n": len(group),
                                            "eligible": eligible, "flips": flips, "conditional_asr": flips/eligible if eligible else None,
                                            "unconditional_escape": sum(r["adv_pred"] == 1 for r in group)/len(group),
                                            "n_real": len(reals), "real_fpr": sum(r["pred"] == 0 for r in condition_real)/len(reals),
                                            "auc_adv_vs_real": core.auc_adv_vs_real([r["adv_ai_margin"] for r in group], real_scores),
                                            "mean_adv_ai_margin": sum(r["adv_ai_margin"] for r in group)/len(group),
                                            "mean_psnr_db": sum(r["psnr_db"] for r in group if r["psnr_db"] is not None)/sum(r["psnr_db"] is not None for r in group) if any(r["psnr_db"] is not None for r in group) else None,
                                            "mean_ssim": sum(r["ssim"] for r in group)/len(group),
                                            "max_linf_levels": max(r["linf_levels"] for r in group),
                                            "mean_spectral_mse": sum(r["spectral_mse_after"] for r in group)/len(group)})
                    del det
                    clear_memory()
                    print(f"Saved-file evaluation complete: {name}", flush=True)
            csv_write(out / "quality.csv", all_records)
            csv_write(out / "per_image.csv", per_image)
            csv_write(out / "real_predictions.csv", real_predictions)
            csv_write(out / "summary.csv", summary)
            json_write(out / "summary.json", summary)
            if len(summary) != 36 or len(per_image) != len(inputs)*36 or len(real_predictions) != len(reals)*12:
                raise AssertionError("Incomplete evaluation groups")
            # Reopen every produced PNG after evaluation; verify exact budgets and RGB originals.
            for record in all_records:
                original = core.load_png(out / record["original_png"])
                adversarial = core.load_png(out / record["adversarial_png"])
                actual = core.image_quality(original, adversarial)[0]
                if actual["linf_levels"] > 8 or actual["linf_levels"] != record["linf_levels"]:
                    raise AssertionError("Post-evaluation saved budget mismatch")
                if sha256(out / record["adversarial_png"]) != record["adversarial_sha256"]:
                    raise AssertionError("Saved PNG changed during evaluation")
                if sha256(out / record["original_png"]) != record["source_sha256"]:
                    raise AssertionError("Saved original differs from source")
            config.update({"status": "complete", "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "wall_seconds": time.monotonic()-started, "summary_rows": len(summary), "per_image_rows": len(per_image),
                           "real_prediction_rows": len(real_predictions), "saved_adversarial_pngs": len(all_records)})
            json_write(out / "config.json", config)
            artifact_hashes = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
            json_write(out / "SHA256SUMS.json", artifact_hashes)
            if any(sha256(out / name) != digest for name, digest in artifact_hashes.items()):
                raise AssertionError("Final artifact hash verification failed")
            print(json.dumps({"status": "complete", "out": str(out), "generated_inputs": len(inputs),
                              "summary_rows": len(summary), "saved_pngs": 2*len(all_records), "verified_files": len(artifact_hashes)}), flush=True)
    except BaseException as error:
        config.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "wall_seconds": time.monotonic()-started})
        json_write(out / "config.json", config)
        raise
    finally:
        os.chdir(previous_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--weights-root", required=True, help="Directory containing checkpoints/ or the checkpoint directory itself")
    parser.add_argument("--out", required=True, help="New directory; existing outputs are refused")
    parser.add_argument("--limit", type=int, default=10, help="Fixed first test inputs, 1..100; default10 is a small course demonstration, not a full benchmark")
    parser.add_argument("--real-limit", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--core-dir", help="Directory containing m3_sfa_core.py; default sibling of this runner")
    args = parser.parse_args()
    if min(args.limit, args.real_limit, args.steps) < 1:
        parser.error("Limits and steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
