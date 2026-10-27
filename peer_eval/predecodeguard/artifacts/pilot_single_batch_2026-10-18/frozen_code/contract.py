"""Pinned input contract and explicit, fail-closed experiment authorization.

Preparation commands never require agreement. All real perturbations, including
random controls, require a confirmed proposal, saved confirmation evidence and
an explicit execution flag. This is an accidental-execution guard, not an
authorization service: an operator must verify that the evidence is genuine.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = (1, 16, 128, 128)
METHODS = ("random", "lpgd", "lstop")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def package_manifest(package: Path, lock_path: Path = HERE / "package_lock.json") -> dict:
    """Verify every pinned file before executing supplied model/scoring code."""
    package = Path(package).resolve()
    lock = read_json(lock_path)
    for name, expected in lock["files"].items():
        path = package / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Package integrity mismatch: {name}")
    manifest = read_json(package / "manifest.json")
    rows = manifest["samples"]
    names = [r["filename"] for r in rows]
    if len(rows) != 10 or len(set(names)) != 10:
        raise ValueError("Expected exactly ten unique source samples")
    for row in rows:
        name = row["filename"]
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError("Invalid manifest filename")
        if tuple(row["shape"]) != SHAPE or row["dtype"] != "float16":
            raise ValueError("Manifest shape/dtype mismatch")
        if row["sha256"] != lock["files"].get("latents/" + name):
            raise ValueError("Manifest sample hash mismatch")
    if manifest["operator"] != ">=":
        raise ValueError("Unexpected threshold convention")
    return manifest


def load_latent(path: Path) -> np.ndarray:
    array = np.load(Path(path), allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.shape != SHAPE or array.dtype != np.float16:
        raise ValueError(f"{Path(path).name}: expected finite float16 {SHAPE}")
    if not np.isfinite(array).all():
        raise ValueError(f"{Path(path).name}: nonfinite latent")
    return array


def perturbation_norms(returned: np.ndarray, original: np.ndarray) -> tuple[float, float]:
    # This exactly follows the peer scorer's arithmetic and zero tolerance.
    delta = returned.astype(np.float32) - original.astype(np.float32)
    return float(np.abs(delta).max()), float(np.linalg.norm(delta.astype(np.float64).ravel()))


def proposal_digest(protocol: dict) -> str:
    proposal = {k: v for k, v in protocol.items() if k not in ("status", "agreement")}
    return hashlib.sha256(json.dumps(proposal, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_protocol(protocol: dict, manifest: dict, manifest_sha: str):
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema")
    if protocol.get("package_manifest_sha256") != manifest_sha:
        raise ValueError("Protocol is not bound to this package manifest")
    if protocol.get("threshold") != manifest["threshold"]:
        raise ValueError("Frozen threshold mismatch")
    if protocol.get("seed_rule") != "base_seed_plus_manifest_index":
        raise ValueError("The per-sample seed rule must be frozen explicitly")
    budget = protocol.get("budget", {})
    if budget.get("norm") != "Linf" or budget.get("units") != "raw_latent" or budget.get("tolerance") != 0:
        raise ValueError("Raw latent Linf budget with zero tolerance is required")
    epsilon = budget.get("epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 <= epsilon <= manifest["perturbation"]["epsilon"]:
        raise ValueError("Invalid or increased protocol budget")
    expected = sorted(r["filename"] for r in manifest["samples"])
    if sorted(protocol.get("sample_filenames", [])) != expected:
        raise ValueError("Protocol must enumerate the ten original sample filenames exactly")
    requested = protocol.get("requested_methods", [])
    if not requested or len(set(requested)) != len(requested) or any(m not in METHODS for m in requested):
        raise ValueError("Invalid requested method list")
    if protocol.get("returned_batches") != len(requested):
        raise ValueError("Each requested method must have one declared return batch")
    for method in requested:
        cfg = protocol["methods"][method]
        if type(cfg.get("steps")) is not int or cfg["steps"] < 1:
            raise ValueError("steps must be a positive integer")
        if type(cfg.get("seed")) is not int or cfg["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        alpha = cfg.get("alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        margin = cfg.get("early_stop_margin", 0.0)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
            raise ValueError("early_stop_margin must be finite and nonnegative")
        if margin > 0 and (method != "lstop" or margin >= protocol["threshold"]):
            raise ValueError("A positive early_stop_margin is only valid for lstop and must be below the threshold")
        controls = cfg.get("control_filenames", [])
        if len(set(controls)) != len(controls) or not set(controls).issubset(expected):
            raise ValueError("Invalid predeclared control allocation")
        if len(controls) == len(expected):
            raise ValueError("A method batch must contain at least one actual attempt")


def require_agreement(protocol: dict, method: str, *, execute_approved: bool, protocol_path: Path):
    """Call before loading a model, creating output directories or perturbing data."""
    if protocol.get("status") != "confirmed":
        raise PermissionError("Real perturbations are blocked: peer protocol is not confirmed")
    if not execute_approved:
        raise PermissionError("Explicit --execute-approved is required")
    agreement = protocol.get("agreement")
    if not isinstance(agreement, dict):
        raise PermissionError("Missing saved peer confirmation evidence")
    for key in ("confirmed_by", "confirmed_at", "reference", "evidence_path", "evidence_sha256", "proposal_sha256"):
        if not isinstance(agreement.get(key), str) or not agreement[key].strip():
            raise PermissionError(f"Incomplete confirmation: {key}")
    try:
        when = datetime.fromisoformat(agreement["confirmed_at"])
        if when.tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError as exc:
        raise PermissionError("Confirmation time must include a timezone") from exc
    if agreement["proposal_sha256"] != proposal_digest(protocol):
        raise PermissionError("Proposal changed after confirmation; renew agreement")
    evidence = Path(agreement["evidence_path"])
    if not evidence.is_absolute():
        evidence = Path(protocol_path).resolve().parent / evidence
    if not evidence.is_file() or sha256(evidence) != agreement["evidence_sha256"]:
        raise PermissionError("Confirmation evidence is absent or changed")
    if method not in protocol["requested_methods"]:
        raise PermissionError("Method is outside the confirmed batch scope")


def new_output(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
