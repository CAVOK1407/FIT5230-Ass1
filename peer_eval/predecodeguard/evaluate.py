"""Manifest-complete evaluation of existing saved files, never tensor-only scores."""
from __future__ import annotations

import json
import math
import contextlib
import io
import runpy
import sys
import traceback
from pathlib import Path

from .contract import load_latent, perturbation_norms, sha256


def evaluate_batch(package, returned, guard, *, method, attempt_names=(), epsilon=None, records=None):
    package, returned = Path(package), Path(returned)
    manifest = guard.manifest
    epsilon = manifest["perturbation"]["epsilon"] if epsilon is None else epsilon
    expected = {r["filename"] for r in manifest["samples"]}
    attempt_names = set(attempt_names)
    if not attempt_names.issubset(expected):
        raise ValueError("Unknown attempted sample IDs")
    records = records or {}
    actual = {p.name for p in returned.iterdir()} if returned.is_dir() else set()
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    rows = []
    for sample in manifest["samples"]:
        name = sample["filename"]
        original = load_latent(package / "latents" / name)
        baseline = guard.score(original)
        if abs(baseline - sample["baseline_score"]) > 1e-6 or guard.flagged(baseline) != sample["baseline_flagged"]:
            raise ValueError(f"CPU baseline differs from frozen manifest: {name}")
        record = records.get(name, {})
        row = dict(filename=name, method=method, attempted=name in attempt_names,
                   baseline_score=baseline, baseline_flagged=guard.flagged(baseline),
                   score=None, flagged=None, score_displacement=None, linf=None, l2=None,
                   valid=False, status="missing", error="", sha256="",
                   original_sha256=sample["sha256"], identical=False,
                   seconds=record.get("seconds"), steps=record.get("steps"))
        path = returned / name
        errors = []
        if record.get("error"):
            errors.append("generation failed: " + record["error"])
        if extra:
            errors.append("unexpected batch entries: " + ", ".join(extra))
        if not path.is_file() or path.is_symlink():
            errors.append("missing file or unsupported symlink")
        else:
            try:
                row["sha256"] = sha256(path)
                array = load_latent(path)
                linf, l2 = perturbation_norms(array, original)
                row.update(linf=linf, l2=l2, identical=bool((array == original).all()))
                if linf > epsilon:
                    errors.append(f"budget exceeded: {linf} > {epsilon}")
                score = guard.score(array)
                if not math.isfinite(score):
                    raise ValueError("nonfinite reference score")
                row.update(score=score, flagged=guard.flagged(score), score_displacement=score - baseline)
            except (ValueError, OSError, EOFError) as exc:
                errors.append(str(exc))
        row["valid"] = not errors
        if errors:
            row["status"] = "missing" if not path.is_file() else "invalid"
            row["error"] = "; ".join(errors)
        elif not row["attempted"]:
            row["status"] = "control"
        elif not row["baseline_flagged"]:
            row["status"] = "baseline_miss"
        elif not row["flagged"]:
            row["status"] = "candidate_pending_review"
        else:
            row["status"] = "no_flip"
        rows.append(row)
    return rows, {"expected_files":len(expected), "missing":missing, "extra":extra,
                  "complete":not missing and not extra, "valid_files":sum(r["valid"] for r in rows)}


def official_verify(package: Path, out: Path, rows: list, returned: Path | None = None):
    """Run unchanged peer scorer, save evidence and compare every adapter score.

    Use runpy, since nesting a second torch/OpenMP interpreter under a torch
    process fails shared-memory registration in the local desktop sandbox.
    The supplied script and its model module are executed unchanged; a standalone
    command is also recorded for independent process reproduction.
    """
    package, out = Path(package).resolve(), Path(out)
    args = [sys.executable, str(package / "score.py")]
    args += ["--verify"] if returned is None else ["--returned", str(Path(returned).resolve())]
    stdout, stderr = io.StringIO(), io.StringIO()
    old_argv, old_path, old_bytecode = sys.argv, sys.path[:], sys.dont_write_bytecode
    old_model = sys.modules.pop("model", None)
    failure = None
    try:
        sys.argv = args[1:]
        sys.path.insert(0, str(package))
        sys.dont_write_bytecode = True
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runpy.run_path(str(package / "score.py"), run_name="__main__")
    except Exception as exc:
        failure = exc
        traceback.print_exc(file=stderr)
    finally:
        sys.argv, sys.path, sys.dont_write_bytecode = old_argv, old_path, old_bytecode
        sys.modules.pop("model", None)
        if old_model is not None:
            sys.modules["model"] = old_model
    (out / "official_score_stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")
    (out / "official_score_stderr.txt").write_text(stderr.getvalue(), encoding="utf-8")
    if failure is not None:
        raise ValueError("Official scorer failed; see saved stderr") from failure
    official = [json.loads(line) for line in stdout.getvalue().splitlines() if line.startswith("{")]
    by_name = {r["filename"]: r for r in official}
    if len(official) != len(rows) or set(by_name) != {r["filename"] for r in rows}:
        raise ValueError("Official scorer omitted or duplicated files")
    max_diff = 0.0
    for row in rows:
        ref = by_name[row["filename"]]
        delta = abs(ref["score"] - row["score"])
        max_diff = max(max_diff, delta)
        if delta > 1e-6 or ref["flagged"] != row["flagged"] or ref["linf"] != row["linf"]:
            raise ValueError("Adapter and official scorer disagree: " + row["filename"])
    return {"exit_code":0, "verified_samples":len(rows), "max_adapter_score_difference":max_diff,
            "execution_mode":"unchanged_script_runpy", "standalone_command":args,
            "stdout_sha256":sha256(out / "official_score_stdout.txt")}
