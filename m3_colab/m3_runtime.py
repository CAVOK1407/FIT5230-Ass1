"""Portable, fresh-run entry points for the frozen FIT5230 M3 course bundle.

This module uses only the Python standard library. The attack and evaluator run
in separate processes using the supplied Python 3.10 environment. It never
imports attack code into the notebook kernel and never resumes an old run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


METHODS = ("A5", "A2", "X2", "X4", "ENS", "CAPB_ENS", "EOT_ENS")
DEFAULT_CONDITIONS = ("none", "jpeg75", "resize05")
DETECTORS = ("E", "R", "D", "S")
SEED = 42
REQUIRED_SOURCES = (
    "m2_stealth.py", "m2_evaluate.py", "m2_device.py", "other_attacks.py",
    "diff_latent_attack.py", "ControlVAE.py", "network.py", "utils.py", "distances.py",
)
CHECKPOINTS = (
    "efficientnet-b0.pth", "resnet50.pth", "deit.pth", "swin-t.pth", "Controlvae.pt",
)
VGG_CHECKPOINT = "vgg16-397923af.pth"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _relative_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe manifest path: {name!r}")
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Manifest path escapes bundle: {name!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    def natural(path: Path) -> list[Any]:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]
    return sorted((p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")), key=natural)


def _selected(directory: Path, limit: int | None) -> list[Path]:
    files = _image_files(directory)
    if not files:
        raise ValueError(f"No images in {directory}")
    if limit is not None and limit < 0:
        raise ValueError("Image limit must be non-negative; 0 means all available images")
    if limit and limit > len(files):
        raise ValueError(f"Requested {limit} images but {directory} contains {len(files)}")
    return files[:limit] if limit else files


def _course_source_check(src: Path) -> None:
    for name in REQUIRED_SOURCES:
        if not (src / name).is_file():
            raise FileNotFoundError(src / name)
    for source in src.rglob("*.py"):
        if source.stem.startswith("m3_lao"):
            raise ValueError(f"Paper LAO module is outside this course bundle: {source}")
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(part.startswith("m3_lao") for name in names for part in name.split(".")):
                raise ValueError(f"Paper LAO import is outside this course bundle: {source}")


def verify_bundle(bundle: str | Path) -> dict[str, Any]:
    """Verify declared file hashes, complete source/data coverage, and dev/test separation."""
    root = _path(bundle)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if not all(key in manifest for key in ("version", "source_snapshot", "files")):
        raise ValueError("manifest.json requires version, source_snapshot and files")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise ValueError("Manifest files must be a non-empty list")
    declared: dict[str, str] = {}
    total_bytes = 0
    for entry in manifest["files"]:
        name = entry["path"]
        canonical = Path(name).as_posix()
        if canonical in declared:
            raise ValueError(f"Duplicate manifest path: {name}")
        path = _relative_file(root, name)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
        declared[canonical] = actual
        total_bytes += path.stat().st_size
    _course_source_check(root / "src")
    # Prevent an unmanifested source file or extra input PNG from entering a run.
    for folder in (root / "src", root / "data"):
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                name = path.relative_to(root).as_posix()
                if name not in declared:
                    raise ValueError(f"Unmanifested source/data file: {name}")
    split_hashes: dict[str, dict[str, list[str]]] = {"dev": {}, "test": {}}
    counts = {}
    for kind in ("gen", "real"):
        for split in ("dev", "test"):
            paths = _selected(root / "data" / kind / split, None)
            counts[f"{kind}/{split}"] = len(paths)
            for path in paths:
                name = path.relative_to(root).as_posix()
                split_hashes[split].setdefault(declared[name], []).append(name)
    overlap = set(split_hashes["dev"]) & set(split_hashes["test"])
    if overlap:
        examples = [{s: split_hashes[s][h] for s in ("dev", "test")} for h in sorted(overlap)[:5]]
        raise ValueError(f"Dev/test image-content overlap: {examples}")
    return {"bundle": str(root), "version": manifest["version"],
            "source_snapshot": manifest["source_snapshot"], "manifest_sha256": sha256(manifest_path),
            "verified_files": len(declared), "verified_bytes": total_bytes,
            "image_counts": counts, "dev_test_hash_overlap": 0}


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _vgg_source(cache: str | Path) -> Path:
    source = _path(cache)
    candidates = (source, source / VGG_CHECKPOINT, source / "checkpoints" / VGG_CHECKPOINT,
                  source / ".cache" / "checkpoints" / VGG_CHECKPOINT)
    for candidate in candidates:
        if candidate.is_file() and candidate.name == VGG_CHECKPOINT:
            return candidate
    raise FileNotFoundError(f"Cannot find {VGG_CHECKPOINT} under {source}")


def prepare_runtime(bundle: str | Path, workdir: str | Path, checkpoint_dir: str | Path,
                    lpips_cache: str | Path | None = None) -> Path:
    """Copy verified sources unchanged into a fresh directory and attach supplied weights.

    Return the runtime directory. Data remain in bundle/data; weights are linked
    where supported. No checkpoint or model is downloaded by this helper.
    """
    checked = verify_bundle(bundle)
    root, runtime, checkpoints = _path(bundle), _path(workdir), _path(checkpoint_dir)
    if runtime.exists() and (not runtime.is_dir() or any(runtime.iterdir())):
        raise FileExistsError(f"Runtime must be new or empty: {runtime}")
    missing = [name for name in CHECKPOINTS if not (checkpoints / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing author checkpoints in {checkpoints}: {missing}")
    vgg = _vgg_source(lpips_cache) if lpips_cache is not None else None
    resources_path = root / "resources.json"
    expected_weights = None
    if resources_path.is_file():
        entries = _read_json(resources_path)["weights"]
        expected_weights = {entry["path"]: entry for entry in entries}
        if len(expected_weights) != len(entries):
            raise ValueError("Duplicate weight path in resources.json")
    weight_records = {}
    supplied_weights = [(f"checkpoints/{name}", checkpoints / name) for name in CHECKPOINTS]
    if vgg is not None:
        supplied_weights.append((f".cache/checkpoints/{VGG_CHECKPOINT}", vgg))
    for relative, path in supplied_weights:
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if expected_weights is not None:
            if relative not in expected_weights:
                raise ValueError(f"Weight is not declared in resources.json: {relative}")
            expected = expected_weights[relative]
            if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
                raise ValueError(f"Author weight differs from resources.json: {relative}")
        weight_records[relative] = actual
    runtime.mkdir(parents=True, exist_ok=True)
    source_files = {}
    for source in sorted((root / "src").rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(root / "src")
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_files[relative.as_posix()] = sha256(destination)
        if source_files[relative.as_posix()] != sha256(source):
            raise ValueError(f"Source changed while copying: {relative}")
    for name in CHECKPOINTS:
        _link_or_copy(checkpoints / name, runtime / "checkpoints" / name)
    if vgg is not None:
        _link_or_copy(vgg, runtime / ".cache" / "checkpoints" / VGG_CHECKPOINT)
    _write_json(runtime / "runtime_manifest.json", {
        "created_at": _utc(), "bundle_verification": checked, "source_sha256": source_files,
        "checkpoint_dir": str(checkpoints),
        "checkpoints": {name: weight_records[f"checkpoints/{name}"] for name in CHECKPOINTS},
        "lpips_cache": str(vgg) if vgg is not None else None,
        "lpips_weight": weight_records.get(f".cache/checkpoints/{VGG_CHECKPOINT}"),
        "resources_sha256": sha256(resources_path) if resources_path.is_file() else None,
    })
    return runtime


def _verify_runtime(runtime: Path) -> dict[str, Any]:
    metadata = _read_json(runtime / "runtime_manifest.json")
    for name, expected in metadata["source_sha256"].items():
        if sha256(_relative_file(runtime, name)) != expected:
            raise ValueError(f"Runtime source changed after preparation: {name}")
    _course_source_check(runtime)
    for name in CHECKPOINTS:
        path = runtime / "checkpoints" / name
        if not path.is_file() or path.stat().st_size != metadata["checkpoints"][name]["bytes"]:
            raise ValueError(f"Runtime checkpoint is missing or changed: {path}")
        if sha256(path) != metadata["checkpoints"][name]["sha256"]:
            raise ValueError(f"Runtime checkpoint hash changed: {path}")
    if metadata.get("lpips_weight"):
        path = runtime / ".cache" / "checkpoints" / VGG_CHECKPOINT
        expected = metadata["lpips_weight"]
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Runtime LPIPS checkpoint is missing or changed: {path}")
    return metadata


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": str(SEED),
                "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    return env


def environment_info(python: str | Path = sys.executable, device: str = "auto") -> dict[str, Any]:
    """Probe the child interpreter without allocating tensors or running a model."""
    if device not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"Unknown device: {device}")
    code = """import json, platform, sys, torch
cuda = torch.cuda.is_available()
print(json.dumps({'python':sys.executable,'python_version':platform.python_version(),
 'platform':platform.platform(),'torch':torch.__version__,'cuda_available':cuda,
 'cuda_device_name':torch.cuda.get_device_name(0) if cuda else None,
 'cuda_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory if cuda else None,
 'mps_available':bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())}))
"""
    result = subprocess.run([str(python), "-B", "-c", code], env=_child_env(), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    chosen = ("cuda" if info["cuda_available"] else "mps" if info["mps_available"] else "cpu") \
        if device == "auto" else device
    if chosen in ("cuda", "mps") and not info[f"{chosen}_available"]:
        raise RuntimeError(f"Requested {chosen} is unavailable in {info['python']}: {info}")
    info["device"] = chosen
    return info


def _stream(command: list[str], cwd: Path, logfile: Path, device: str) -> None:
    env = _child_env()
    env["M2_DEVICE"] = device
    with logfile.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cwd": str(cwd), "started_at": _utc()}) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def _fresh_output(root: Path, name: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    output = root / name
    logs = root / "_logs"
    logs.mkdir(exist_ok=True)
    logfile = logs / f"{name}.stdout.log"
    if output.exists() or logfile.exists():
        raise FileExistsError(f"Fresh output required; choose a new outroot: {output}")
    output.mkdir()
    return output, logfile


def _method_args(method: str, data: Path) -> list[str]:
    variants = {
        "A5": [], "A2": ["--iterations", "0"], "X2": ["--control_source", "clean"],
        "X4": ["--encode_source", "clean", "--control_source", "clean", "--iterations", "0"],
        "ENS": ["--surrogates", "E,R,D"],
        "CAPB_ENS": ["--surrogates", "E,R,D", "--pgd_impl", "own", "--cap", "auto",
                     "--cap_bisect", "--real_dev_dir", str(data / "real" / "dev"),
                     "--pgd_alpha", str(4 / 255 / 10), "--pgd_steps", "40",
                     "--loss", "margin", "--kappa", "cap"],
        "EOT_ENS": ["--surrogates", "E,R,D", "--pgd_impl", "own", "--pgd_eot", "both",
                    "--eot_jpeg_q", "75", "--eot_resize", "0.5"],
    }
    if method not in variants:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    return variants[method]


def _png_manifest(output: Path, expected: int) -> dict[str, Any]:
    records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines() if line.strip()]
    expected_ids = [f"{i:04d}" for i in range(expected)]
    if [record["image_id"] for record in records] != expected_ids:
        raise ValueError(f"Incomplete, duplicate or reordered attack records in {output}")
    for suffix in ("_originImage.png", "_adv_image.png"):
        if sorted(p.name[:-len(suffix)] for p in output.glob(f"*{suffix}")) != expected_ids:
            raise ValueError(f"Incomplete PNG outputs ({suffix}) in {output}")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(output.glob("*.png"))]
    if not files or any(item["bytes"] == 0 for item in files):
        raise ValueError(f"Empty PNG output in {output}")
    return {"n_images": expected, "image_ids": expected_ids, "files": files}


def run_method(runtime_dir: str | Path, data_dir: str | Path, outroot: str | Path,
               method: str, python: str | Path = sys.executable, device: str = "auto",
               model_dir: str | Path | None = None, max_images: int = 3) -> dict[str, Any]:
    """Run one method from scratch and validate its records and saved PNG hashes."""
    runtime, data, root = _path(runtime_dir), _path(data_dir), _path(outroot)
    source = _verify_runtime(runtime)
    extra = _method_args(method, data)
    selected = _selected(data / "gen" / "test", max_images)
    if root.is_relative_to(data):
        raise ValueError("Outputs must not be written inside the input dataset")
    if model_dir is None:
        raise ValueError("model_dir must point to the supplied local SD2.1 model")
    model = _path(model_dir)
    if not (model / "model_index.json").is_file():
        raise FileNotFoundError(f"Local diffusion model is incomplete: {model / 'model_index.json'}")
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        if not (model / component).is_dir():
            raise FileNotFoundError(model / component)
    if method == "CAPB_ENS":
        _selected(data / "real" / "dev", None)
    environment = environment_info(python, device)
    output, logfile = _fresh_output(root, method)
    command = [environment["python"], "-B", "-u", str(runtime / "m2_stealth.py"),
               "--images_root", str(data / "gen" / "test"), "--save_dir", str(output),
               "--pretrained_diffusion_path", str(model),
               "--encoder_weights", str(runtime / "checkpoints" / "Controlvae.pt"),
               "--device", environment["device"], "--seed", str(SEED),
               "--max_images", str(len(selected)), "--iterations", "10", "--evaluate", "E,R,D,S", *extra]
    metadata = {"method": method, "status": "running", "started_at": _utc(),
                "command": command, "environment": environment, "model_dir": str(model),
                "seed": SEED, "n_images": len(selected), "stdout_log": str(logfile),
                "source_sha256": source["source_sha256"],
                "bundle_manifest_sha256": source["bundle_verification"]["manifest_sha256"],
                "inputs": [{"path": str(p), "sha256": sha256(p)} for p in selected]}
    started = time.monotonic()
    try:
        _stream(command, runtime, logfile, environment["device"])
        pngs = _png_manifest(output, len(selected))
        config = _read_json(output / "run_config.json")
        if config.get("seed") != SEED or config.get("resume"):
            raise ValueError("Attack config violated fixed-seed/fresh-run policy")
        records = [json.loads(s) for s in (output / "records.jsonl").read_text().splitlines() if s.strip()]
        if [r["source"] for r in records] != [p.name for p in selected]:
            raise ValueError("Attack source order does not match the selected input images")
        _write_json(output / "saved_png_sha256.json", pngs)
        metadata.update({"status": "complete", "saved_png_count": len(pngs["files"])})
    except BaseException as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        metadata.update({"finished_at": _utc(), "wall_seconds": round(time.monotonic() - started, 3)})
        _write_json(output / "execution.json", metadata)
    return {"method": method, "output_dir": str(output), "n_images": len(selected),
            "saved_png_count": metadata["saved_png_count"], "stdout_log": str(logfile), "status": "complete"}


def run_all(runtime_dir: str | Path, data_dir: str | Path, outroot: str | Path,
            python: str | Path = sys.executable, device: str = "auto", model_dir: str | Path | None = None,
            max_images: int = 3, methods: tuple[str, ...] | list[str] = METHODS) -> list[dict[str, Any]]:
    """Run methods sequentially; every method still gets a separate clean process."""
    return [run_method(runtime_dir, data_dir, outroot, method, python, device, model_dir, max_images)
            for method in methods]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Evaluation output is empty: {path}")
    return rows


def evaluate(runtime_dir: str | Path, data_dir: str | Path, outroot: str | Path,
             python: str | Path = sys.executable, device: str = "auto",
             methods: tuple[str, ...] | list[str] = METHODS,
             conditions: tuple[str, ...] | list[str] = DEFAULT_CONDITIONS,
             real_limit: int | None = None, no_lpips: bool = False,
             include_intermediates: bool = False) -> dict[str, Any]:
    """Independently evaluate saved PNGs; default real FPR uses all 100 real test images.

    A new process loads detectors one at a time. Attack-time predictions are not
    reused. The default three conditions match the archived course evaluator.
    """
    runtime, data, root = _path(runtime_dir), _path(data_dir), _path(outroot)
    _verify_runtime(runtime)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Supply a non-empty, unique method list")
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Supply a non-empty, unique condition list")
    if any(c not in ("none", "jpeg95", "jpeg75", "jpeg50", "resize075", "resize05") for c in conditions):
        raise ValueError(f"Unsupported evaluation conditions: {conditions}")
    if not no_lpips and not (runtime / ".cache" / "checkpoints" / VGG_CHECKPOINT).is_file():
        raise FileNotFoundError("LPIPS cache is missing; prepare with lpips_cache or explicitly use no_lpips=True")
    runs, expected_counts = [], {}
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown course method: {method}")
        output = root / method
        execution = _read_json(output / "execution.json")
        if execution["status"] != "complete":
            raise ValueError(f"Method did not complete: {method}")
        recorded = _read_json(output / "saved_png_sha256.json")
        if _png_manifest(output, execution["n_images"]) != recorded:
            raise ValueError(f"Saved PNGs changed since attack completion: {method}")
        runs.append(f"{method}={output}")
        expected_counts[method] = execution["n_images"]
    if include_intermediates:
        if "A5" not in methods:
            raise ValueError("Intermediate A1/X1 evaluation requires A5")
        runs += [f"A1={root / 'A5'}:_pgd1.png", f"X1={root / 'A5'}:_control.png"]
        expected_counts.update({"A1": expected_counts["A5"], "X1": expected_counts["A5"]})
    real_files = _selected(data / "real" / "test", real_limit)
    environment = environment_info(python, device)
    output, logfile = _fresh_output(root, "evaluation")
    real_dir = data / "real" / "test"
    if real_limit:
        real_dir = output / "real_test_subset"
        real_dir.mkdir()
        for path in real_files:
            shutil.copy2(path, real_dir / path.name)
    command = [environment["python"], "-B", "-u", str(runtime / "m2_evaluate.py"), "--runs", *runs,
               "--real_dir", str(real_dir), "--out", str(output), "--detectors", ",".join(DETECTORS),
               "--conditions", ",".join(conditions), "--device", environment["device"]]
    if no_lpips:
        command.append("--no_lpips")
    metadata = {"status": "running", "started_at": _utc(), "command": command,
                "environment": environment, "stdout_log": str(logfile), "source": "saved_png_files",
                "n_real": len(real_files), "methods": list(expected_counts), "conditions": list(conditions),
                "lpips_computed": not no_lpips,
                "real_inputs": [{"path": str(p), "sha256": sha256(p)} for p in real_files]}
    started = time.monotonic()
    try:
        _stream(command, runtime, logfile, environment["device"])
        summary = _csv_rows(output / "summary.csv")
        per_image = _csv_rows(output / "per_image.csv")
        real_rows = _csv_rows(output / "real_predictions.csv")
        factor = len(DETECTORS) * len(conditions)
        if len(summary) != len(expected_counts) * factor or len(per_image) != sum(expected_counts.values()) * factor:
            raise ValueError("Evaluation summary/per-image row counts are incomplete")
        if len(real_rows) != len(real_files) * factor:
            raise ValueError("Real-image prediction row count is incomplete")
        expected_keys = {(m, d, c) for m in expected_counts for d in DETECTORS for c in conditions}
        if {(r["run"], r["detector"], r["condition"]) for r in summary} != expected_keys:
            raise ValueError("Evaluation summary has missing or duplicate groups")
        if any(int(r["n"]) != expected_counts[r["run"]] for r in summary):
            raise ValueError("Evaluation summary uses an unexpected sample count")
        metadata.update({"status": "complete", "summary_rows": len(summary),
                         "per_image_rows": len(per_image), "real_prediction_rows": len(real_rows),
                         "output_files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                                          for p in sorted(output.iterdir()) if p.is_file()]})
    except BaseException as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        metadata.update({"finished_at": _utc(), "wall_seconds": round(time.monotonic() - started, 3)})
        _write_json(output / "execution.json", metadata)
    return {"output_dir": str(output), "summary_rows": len(summary), "per_image_rows": len(per_image),
            "n_real": len(real_files), "stdout_log": str(logfile), "status": "complete"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--bundle", required=True)
    prepare.add_argument("--workdir", required=True)
    prepare.add_argument("--checkpoint-dir", required=True)
    prepare.add_argument("--lpips-cache")
    env = commands.add_parser("environment")
    for name in ("run", "run-all", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--runtime-dir", required=True)
        command.add_argument("--data-dir", required=True)
        command.add_argument("--outroot", required=True)
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
        if name in ("run", "run-all"):
            command.add_argument("--model-dir", required=True)
            command.add_argument("--max-images", type=int, default=3)
        if name == "run":
            command.add_argument("--method", required=True, choices=METHODS)
        else:
            command.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        if name == "evaluate":
            command.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
            command.add_argument("--real-limit", type=int, default=None)
            command.add_argument("--no-lpips", action="store_true")
            command.add_argument("--include-intermediates", action="store_true")
    env.add_argument("--python", default=sys.executable)
    env.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    args = vars(parser.parse_args(argv))
    name = args.pop("command")
    if name == "verify":
        result = verify_bundle(**args)
    elif name == "prepare":
        result = {"runtime_dir": str(prepare_runtime(**args))}
    elif name == "environment":
        result = environment_info(**args)
    elif name == "run":
        result = run_method(**args)
    elif name == "run-all":
        result = run_all(**args)
    else:
        result = evaluate(**args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
