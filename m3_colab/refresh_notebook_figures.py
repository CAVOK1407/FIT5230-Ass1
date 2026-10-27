"""Refresh three presentation cells in an already completed M3 notebook.

Only cells 18 (gallery), 22 (historical plots), and 28 (export) are executed.
Their outputs come from a fresh ipykernel. All other code cells, including model
execution counts and outputs, are retained exactly. No attack or evaluation is
rerun. Run this with the notebook-authoring Python environment.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


TARGETS = (18, 22, 28)
CODE_CELL_COUNT = 19
MARKERS = {18: "new_execution_examples.png", 22: "plot_course_results", 28: "export_id"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def validate_outputs(notebook) -> dict[str, int]:
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(cells) != CODE_CELL_COUNT:
        raise ValueError(f"Expected {CODE_CELL_COUNT} code cells, found {len(cells)}")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None or not cell.get("outputs"):
            raise ValueError(f"Cell {index} lacks completed execution evidence; wait for the full run")
        if any(output.output_type == "error" for output in cell.outputs):
            raise ValueError(f"Cell {index} contains an execution error")
    nbformat.validate(notebook)
    return {"code_cells": len(cells), "code_cells_with_outputs": len(cells), "error_outputs": 0}


def prepare_notebook(source, executed):
    """Merge presentation source and Markdown only; leave other code cells intact."""
    validate_outputs(executed)
    if len(source.cells) != len(executed.cells):
        raise ValueError("Source/executed notebook structures differ")
    result = copy.deepcopy(executed)
    for index, (new, old) in enumerate(zip(source.cells, executed.cells)):
        if new.cell_type != old.cell_type:
            raise ValueError(f"Cell type changed at index {index}")
        if new.cell_type == "markdown":
            result.cells[index].source = new.source
    for index in TARGETS:
        cell = source.cells[index]
        if cell.cell_type != "code" or MARKERS[index] not in cell.source:
            raise ValueError(f"Unexpected presentation cell at index {index}")
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else \
                    node.func.id if isinstance(node.func, ast.Name) else ""
                if name in {"run_method", "run_all", "evaluate", "environment_info", "cuda", "mps"}:
                    raise ValueError(f"Model/device execution is forbidden in refresh cell {index}: {name}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(name.split(".")[0] in {"torch", "diffusers", "subprocess"} for name in names):
                    raise ValueError(f"Model/process import is forbidden in refresh cell {index}")
        result.cells[index].source = cell.source
    return result


def bootstrap_source(run_dir: Path, resource_dir: Path) -> str:
    # Values are literals inside Python code, never shell interpolation.
    return f'''import os, sys, json, hashlib, shutil, zipfile, uuid, datetime
from pathlib import Path
from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.dont_write_bytecode = True
WORK = Path({str(run_dir)!r})
RESOURCE = Path({str(resource_dir)!r})
BUNDLE = WORK / "bundle"
OUT = WORK / "new_execution"
DATA = BUNDLE / "data"
RUNTIME = WORK / "runtime"
IN_COLAB = False
saved_execution = json.loads((WORK / "execution_summary.json").read_text())
saved_environment = json.loads((WORK / "environment.json").read_text())
env_info = saved_environment["runtime"]
assert env_info == saved_execution["environment"], "Environment provenance mismatch"
RUN_ID = saved_execution["run_id"]
assert RUN_ID == WORK.name, "Run directory/summary mismatch"
DEVICE = saved_execution["device"]
PYTHON = env_info["python"]
dependencies = saved_environment["packages"]
METHODS = tuple(saved_execution["methods"])
N_IMAGES = saved_execution["fresh_generated_inputs"]
EXPECTED_BUNDLE_SHA256 = saved_execution["course_bundle_sha256"]
resources = json.loads((BUNDLE / "resources.json").read_text())
evaluation_record = json.loads((OUT / "evaluation" / "execution.json").read_text())
assert evaluation_record["status"] == "complete"
CONDITIONS = tuple(evaluation_record["conditions"])
for method in METHODS:
    method_record = json.loads((OUT / method / "execution.json").read_text())
    assert method_record["status"] == "complete" and method_record["n_images"] == N_IMAGES
sys.path.insert(0, str(BUNDLE))
import m3_runtime as runtime
import m3_reporting as report
bundle_check = runtime.verify_bundle(BUNDLE)
assert bundle_check["source_snapshot"] == saved_execution["source_snapshot"]
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
historical = report.load_course_results(BUNDLE)
fresh = report.load_course_results(OUT / "evaluation")
(WORK / "figures").mkdir(exist_ok=True)
print("Figure/export refresh only; no model execution.")
print("Existing model run:", RUN_ID, "| recorded device:", DEVICE)
print("Rendering kernel:", sys.executable)
'''


def refresh(source: Path, executed: Path, run_dir: Path) -> dict:
    source, executed, run_dir = source.resolve(), executed.resolve(), run_dir.resolve()
    original_digest = file_sha256(executed)
    original = nbformat.read(executed, as_version=4)
    updated = prepare_notebook(nbformat.read(source, as_version=4), original)
    summary_path = run_dir / "execution_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["run_id"] != run_dir.name:
        raise ValueError("Run directory does not match its execution_summary.json")
    if summary["execution_location"] != "Local Jupyter":
        raise ValueError("This restoration helper expects the completed local Jupyter run")
    output_dir = run_dir / "new_execution"
    before_outputs = tree_hashes(output_dir)
    if not before_outputs:
        raise ValueError("The actual model-output directory is empty")
    # Prevent refreshing an unrelated notebook using another run's images.
    original_text = json.dumps(original, ensure_ascii=False)
    if summary["run_id"] not in original_text:
        raise ValueError("The supplied notebook does not contain this model run ID")
    refresh_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    audit_dir = executed.parent / f"figure_refresh_{refresh_id}"
    audit_dir.mkdir(exist_ok=False)
    shutil.copy2(executed, audit_dir / "before.ipynb")
    html_path = executed.with_suffix(".html")
    if html_path.is_file():
        shutil.copy2(html_path, audit_dir / "before.html")
    bootstrap = nbformat.v4.new_code_cell(bootstrap_source(run_dir, executed.parent))
    updated.cells.append(bootstrap)
    bootstrap_index = len(updated.cells) - 1
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    client = NotebookClient(updated, timeout=600, kernel_name="python3", allow_errors=False,
                            record_timing=True, resources={"metadata": {"path": str(run_dir)}})
    provenance = {"refresh_id": refresh_id, "kind": "figure-only rerun and export",
                  "run_id": summary["run_id"], "source_notebook_sha256": file_sha256(source),
                  "original_executed_notebook_sha256": original_digest,
                  "rerun_cell_indices_zero_based": list(TARGETS), "model_execution_rerun": False,
                  "execution_count_note": "Actual new-kernel counts retained; counts may be non-monotonic.",
                  "original_notebook_backup": str(audit_dir / "before.ipynb")}
    try:
        with client.setup_kernel():
            client.execute_cell(updated.cells[bootstrap_index], bootstrap_index)
            for index in TARGETS:
                client.execute_cell(updated.cells[index], index)
                updated.cells[index].metadata["m3_figure_refresh"] = {
                    "refresh_id": refresh_id, "original_model_run_id": summary["run_id"],
                    "model_execution_rerun": False}
                print(f"Refreshed cell {index}: execution_count={updated.cells[index].execution_count}, "
                      f"outputs={len(updated.cells[index].outputs)}", flush=True)
        updated.cells.pop()
        validation = validate_outputs(updated)
        protected = [i for i, cell in enumerate(original.cells) if cell.cell_type == "code" and i not in TARGETS]
        for index in protected:
            if updated.cells[index] != original.cells[index]:
                raise AssertionError(f"Protected model/report code cell changed: {index}")
        if tree_hashes(output_dir) != before_outputs:
            raise AssertionError("Attack/evaluation artifacts changed during a presentation-only refresh")
        if file_sha256(executed) != original_digest:
            raise RuntimeError("The executed notebook changed concurrently; refusing to replace it")
        provenance.update({"validation": validation, "protected_code_cells": protected,
                           "protected_code_cells_unchanged": True, "model_artifact_hashes_unchanged": True,
                           "new_execution_counts": {str(i): updated.cells[i].execution_count for i in TARGETS}})
        updated.metadata.setdefault("m3_figure_refresh", []).append(provenance)
        html, _ = HTMLExporter().from_notebook_node(updated)
        candidate = audit_dir / "refreshed.ipynb"
        nbformat.write(updated, candidate)
        rendered = audit_dir / "refreshed.html"
        rendered.write_text(html, encoding="utf-8")
        shutil.copyfile(candidate, executed)
        shutil.copyfile(rendered, html_path)
        provenance["refreshed_notebook_sha256"] = file_sha256(executed)
        provenance["refreshed_html_sha256"] = file_sha256(html_path)
        print(f"PASS: {validation['code_cells_with_outputs']}/{CODE_CELL_COUNT} code cells have outputs, "
              "zero errors; original model outputs retained.", flush=True)
    except BaseException:
        failed = copy.deepcopy(updated)
        if len(failed.cells) == len(original.cells) + 1:
            failed.cells.pop()
        nbformat.write(failed, audit_dir / "refresh_failed.ipynb")
        provenance["status"] = "failed; original notebook retained unless final write had started"
        raise
    else:
        provenance["status"] = "complete"
    finally:
        provenance["bootstrap_cell"] = dict(bootstrap)
        (audit_dir / "refresh_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"notebook": str(executed), "html": str(html_path), "audit_dir": str(audit_dir),
            "validation": validation, "rerun_cells": list(TARGETS), "model_execution_rerun": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--executed", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(refresh(args.source, args.executed, args.run_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
