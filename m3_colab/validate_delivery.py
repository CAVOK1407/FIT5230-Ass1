"""Read-only checks for an executed M3 notebook and optional exported output ZIP.

No notebook cell, model, or downloaded source is executed. JSON on stdout reports
the hardware evidenced by outputs/records; it does not assume Colab completed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile

import nbformat


METHODS = ("A5", "A2", "X2", "X4", "ENS", "CAPB_ENS", "EOT_ENS")
DETECTORS = ("E", "R", "D", "S")
CONDITIONS = ("none", "jpeg75", "resize05")
N_IMAGES = 3


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stream_hash(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_hash(stream)


def output_text(output) -> str:
    value = output.get("text", output.get("data", {}).get("text/plain", ""))
    return "".join(value) if isinstance(value, list) else str(value)


def dicts_in_text(text: str):
    """Read literal display dictionaries and JSON objects embedded in stdout."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError, TypeError):
            pass
        else:
            if isinstance(value, dict):
                yield value
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def check_environment(environment: dict, expected_device: str | None = None) -> None:
    device = environment.get("device")
    require(device in ("cuda", "mps"), f"Expected recorded CUDA/MPS execution, found {device!r}")
    if expected_device:
        require(device == expected_device, f"Expected {expected_device}, recorded {device}")
    require(environment.get(f"{device}_available") is True, f"Recorded {device} availability is not true")
    require(bool(environment.get("python")) and bool(environment.get("torch")), "Incomplete environment record")
    if device == "cuda":
        require(bool(environment.get("cuda_device_name")), "CUDA device name is missing")
        require(isinstance(environment.get("cuda_total_memory_bytes"), int)
                and environment["cuda_total_memory_bytes"] > 0, "CUDA memory record is missing")


def inspect_notebook(path: Path, expected_device: str | None = None) -> tuple[dict, dict]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [(i, c) for i, c in enumerate(notebook.cells) if c.cell_type == "code"]
    require(len(code_cells) == 19, f"Expected 19 code cells, found {len(code_cells)}")
    texts, dictionaries, images = [], [], 0
    for index, cell in code_cells:
        compile(ast.parse(cell.source, filename=f"notebook_cell_{index}"), f"notebook_cell_{index}", "exec")
        require(cell.execution_count is not None and bool(cell.get("outputs")),
                f"Code cell {index} has no completed execution/output")
        for output in cell.outputs:
            require(output.output_type != "error", f"Code cell {index} contains an error output")
            text = output_text(output)
            texts.append(text)
            dictionaries.extend(dicts_in_text(text))
            data = output.get("data", {})
            if isinstance(data.get("application/json"), dict):
                dictionaries.append(data["application/json"])
            images += int(any(key.startswith("image/") for key in data))
    joined = "\n".join(texts)
    environments = [d for d in dictionaries if {"python", "torch", "device", "cuda_available", "mps_available"} <= set(d)]
    require(bool(environments), "No actual environment dictionary was found in notebook outputs")
    environment = environments[0]
    require(all(d == environment for d in environments), "Notebook contains conflicting environment outputs")
    check_environment(environment, expected_device)
    declared_devices = set(re.findall(r"Actual hardware:\s*(cuda|mps|cpu)", joined))
    require(declared_devices == {environment["device"]}, "Final hardware summary disagrees with environment output")
    child_devices = set(re.findall(r"\[m2_device\] using device:\s*(cuda|mps|cpu)", joined))
    require(child_devices == {environment["device"]}, "Model/evaluator stdout disagrees with recorded device")
    run_ids = set(re.findall(r"Run ID:\s*([A-Za-z0-9_]+)", joined))
    require(len(run_ids) == 1, "Notebook does not identify exactly one completed run ID")
    run_id = next(iter(run_ids))
    method_results = {}
    for method in METHODS:
        matches = [d for d in dictionaries if d.get("method") == method and "output_dir" in d]
        require(bool(matches), f"Missing completed {method} result JSON in notebook outputs")
        require(all(d.get("status") == "complete" and d.get("n_images") == N_IMAGES for d in matches),
                f"Notebook {method} result is not complete for three inputs")
        require(all(run_id in d["output_dir"] for d in matches), f"Notebook {method} output belongs to another run")
        require(all(d == matches[0] for d in matches), f"Conflicting notebook {method} result records")
        method_results[method] = matches[0]
    evaluations = [d for d in dictionaries if {"summary_rows", "per_image_rows", "n_real", "output_dir"} <= set(d)]
    require(bool(evaluations), "Missing independent evaluation result JSON")
    for result in evaluations:
        require(result.get("status") == "complete" and result["summary_rows"] == 84
                and result["per_image_rows"] == 252 and result["n_real"] == 100,
                "Notebook evaluation is incomplete or has unexpected counts")
        require(run_id in result["output_dir"], "Notebook evaluation belongs to another run")
    require("COMPLETE: seven methods" in joined, "Final notebook completion output is missing")
    require(images > 0, "Notebook has no rendered image outputs")
    output_hashes = set(re.findall(r"Output SHA-256:\s*([0-9a-f]{64})", joined))
    require(len(output_hashes) == 1, "Notebook does not identify exactly one output ZIP SHA-256")
    warnings = []
    scope_note = str(notebook.metadata.get("m3_scope", {}).get("execution_outputs", ""))
    if "not executed yet" in scope_note.lower() or \
            (environment["device"] == "cuda" and "colab cuda not yet executed" in scope_note.lower()):
        warnings.append("Custom m3_scope.execution_outputs metadata is stale; actual output evidence was checked instead.")
    public = {"path": str(path), "sha256": file_hash(path), "code_cells": 19, "cells_with_outputs": 19,
              "error_outputs": 0, "ast_compile": "pass", "rendered_image_outputs": images,
              "run_id": run_id, "device": environment["device"],
              "cuda_device_name": environment.get("cuda_device_name"),
              "cuda_total_memory_bytes": environment.get("cuda_total_memory_bytes"),
              "methods_complete": 7, "generated_inputs_per_method": N_IMAGES,
              "summary_rows": 84, "per_image_rows": 252, "real_test_images": 100, "warnings": warnings}
    evidence = {"environment": environment, "run_id": run_id, "method_results": method_results,
                "output_zip_sha256": next(iter(output_hashes)), "evaluation": evaluations[0]}
    return public, evidence


def safe_zip_files(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    files, names = {}, set()
    infos = archive.infolist()
    require(len(infos) <= 20000, "Output ZIP contains an unexpected number of entries")
    require(sum(i.file_size for i in infos) <= 2 * 1024**3, "Output ZIP exceeds the 2 GiB output-artifact limit")
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        require(name not in names, f"Duplicate ZIP entry: {name}")
        names.add(name)
        require(bool(name) and not path.is_absolute() and ".." not in path.parts
                and "\\" not in name and not re.match(r"^[A-Za-z]:", name)
                and path.as_posix() == name.rstrip("/"), f"Unsafe ZIP path: {name}")
        kind = stat.S_IFMT(info.external_attr >> 16)
        require(kind in (0, stat.S_IFREG, stat.S_IFDIR), f"ZIP entry is not a regular file/directory: {name}")
        require(not info.flag_bits & 1, f"Encrypted ZIP entry is unsupported: {name}")
        if not info.is_dir():
            files[name] = info
    return files


def inspect_zip(path: Path, notebook_evidence: dict) -> dict:
    archive_digest = file_hash(path)
    require(archive_digest == notebook_evidence["output_zip_sha256"], "Output ZIP hash differs from notebook export output")
    with zipfile.ZipFile(path) as archive:
        files = safe_zip_files(archive)
        require("SHA256SUMS.json" in files, "Output ZIP lacks SHA256SUMS.json")
        hashes = json.loads(archive.read("SHA256SUMS.json"))
        require(isinstance(hashes, dict) and bool(hashes), "Invalid output SHA-256 manifest")
        require(set(hashes) == set(files) - {"SHA256SUMS.json"}, "Output SHA-256 manifest does not cover exactly all files")
        for name, expected in hashes.items():
            require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid SHA-256 for {name}")
            with archive.open(name) as stream:
                require(stream_hash(stream) == expected, f"Output checksum mismatch: {name}")
        def read_json(name):
            require(name in files, f"Output ZIP lacks {name}")
            return json.loads(archive.read(name))
        summary = read_json("execution_summary.json")
        environment = read_json("environment.json")["runtime"]
        require(environment == summary["environment"] == notebook_evidence["environment"],
                "Notebook, environment.json and execution_summary.json disagree on actual hardware/runtime")
        check_environment(environment)
        require(summary["device"] == environment["device"], "Execution summary device mismatch")
        require(summary["run_id"] == notebook_evidence["run_id"], "Notebook and output ZIP run IDs differ")
        if summary["execution_location"] == "Colab CUDA":
            require(summary["device"] == "cuda", "Colab CUDA summary records a non-CUDA device")
        require(set(summary["methods"]) == set(METHODS), "Execution summary lacks exactly the seven course methods")
        require(summary["fresh_generated_inputs"] == N_IMAGES, "Execution summary is not the three-image demonstration")
        require(summary.get("peer_review") == "pending", "Unexpected peer review status in this frozen course delivery")
        expected_ids = [f"{i:04d}" for i in range(N_IMAGES)]
        png_count = 0
        for method in METHODS:
            prefix = f"new_execution/{method}/"
            brief = summary["methods"][method]
            execution = read_json(prefix + "execution.json")
            require(brief["status"] == execution["status"] == "complete"
                    and brief["n_images"] == execution["n_images"] == N_IMAGES,
                    f"Incomplete {method} execution in output ZIP")
            require(execution["environment"] == environment, f"{method} ran with a different recorded environment")
            command = execution["command"]
            require("--save_dir" in command and command[command.index("--save_dir") + 1]
                    == notebook_evidence["method_results"][method]["output_dir"], f"{method} notebook/record output path mismatch")
            config = read_json(prefix + "run_config.json")
            require(config["seed"] == 42 and config.get("resume") is False and config["n_images"] == N_IMAGES,
                    f"{method} configuration violates seed/fresh-run/sample-count contract")
            require(config["device"] == environment["device"], f"{method} run_config device mismatch")
            records = [json.loads(s) for s in archive.read(prefix + "records.jsonl").decode().splitlines() if s.strip()]
            require([r["image_id"] for r in records] == expected_ids, f"{method} records are missing, duplicated or reordered")
            pngs = read_json(prefix + "saved_png_sha256.json")
            require(pngs["n_images"] == N_IMAGES and pngs["image_ids"] == expected_ids, f"{method} PNG manifest has wrong counts")
            require(len({r["path"] for r in pngs["files"]}) == len(pngs["files"]), f"{method} PNG manifest has duplicate paths")
            declared_pngs = {prefix + r["path"] for r in pngs["files"]}
            actual_pngs = {name for name in files if name.startswith(prefix) and name.endswith(".png")}
            require(declared_pngs == actual_pngs, f"{method} PNG manifest coverage mismatch")
            for item in pngs["files"]:
                name = prefix + item["path"]
                require(name in files and files[name].file_size == item["bytes"] and hashes[name] == item["sha256"],
                        f"{method} saved PNG hash/size mismatch: {item['path']}")
            for image_id in expected_ids:
                for suffix in ("_originImage.png", "_adv_image.png"):
                    require(prefix + image_id + suffix in actual_pngs, f"{method} lacks a required saved PNG")
            png_count += len(actual_pngs)
        evaluation = read_json("new_execution/evaluation/execution.json")
        require(evaluation["status"] == "complete" and evaluation["environment"] == environment,
                "Independent evaluator did not complete with the recorded environment")
        require(set(evaluation["methods"]) == set(METHODS) and set(evaluation["conditions"]) == set(CONDITIONS)
                and evaluation["n_real"] == 100, "Independent evaluation scope differs from seven methods/three transforms/100 real images")
        from m3_reporting import load_course_results, audit_score_auc
        with tempfile.TemporaryDirectory(prefix="m3_delivery_csv_audit_") as temp:
            directory = Path(temp)
            for name in ("summary.csv", "per_image.csv", "real_predictions.csv"):
                (directory / name).write_bytes(archive.read("new_execution/evaluation/" + name))
            results = load_course_results(directory)
            require(len(results.summary) == 84 and len(results.per_image) == 252
                    and len(results.real_predictions) == 1200, "Independent CSV audit found incorrect row counts")
            expected_groups = {(m, d, c) for m in METHODS for d in DETECTORS for c in CONDITIONS}
            require(set(zip(results.summary.run, results.summary.detector, results.summary.condition)) == expected_groups,
                    "CSV audit found incorrect method/detector/condition groups")
            require(results.summary.n.eq(N_IMAGES).all(), "CSV audit found an incorrect generated-image count")
            auc_audit = audit_score_auc(results)
        return {"path": str(path), "sha256": archive_digest, "safe_paths": True,
                "verified_file_hashes": len(hashes), "saved_pngs_verified": png_count,
                "execution_location": summary["execution_location"], "run_id": summary["run_id"],
                "methods_complete": 7, "generated_inputs_per_method": N_IMAGES,
                "csv_counts_asr_fpr_audit": "pass", "score_auc_audit": auc_audit,
                "notebook_environment_summary_match": True}


def validate_delivery(notebook: Path, output_zip: Path | None = None, expected_device: str | None = None) -> dict:
    notebook_result, evidence = inspect_notebook(notebook.resolve(), expected_device)
    zip_result = inspect_zip(output_zip.resolve(), evidence) if output_zip is not None else None
    return {"status": "pass", "notebook": notebook_result, "output_zip": zip_result,
            "models_executed_by_validator": False, "notebook_modified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--output-zip", type=Path)
    parser.add_argument("--expected-device", choices=("cuda", "mps"))
    args = parser.parse_args()
    try:
        result = validate_delivery(args.notebook, args.output_zip, args.expected_device)
    except Exception as error:
        print(json.dumps({"status": "fail", "error": f"{type(error).__name__}: {error}"}, separators=(",", ":")))
        raise SystemExit(1)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
