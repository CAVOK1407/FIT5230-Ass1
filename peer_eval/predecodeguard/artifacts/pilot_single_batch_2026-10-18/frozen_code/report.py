"""Pure, failure-preserving summaries for the PreDecodeGuard pilot.

This module neither loads a model nor creates or decodes perturbations. Reviews
are evidence about a specific saved file, so matching a filename alone is never
sufficient. The positive-only pilot cannot estimate FPR, AUROC or generalization.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTROL_METHODS = {
    "baseline", "copy", "roundtrip", "round_trip", "round-trip", "r0",
    "noop", "no-op", "no_op", "unmodified", "c0", "c1",
}
REVIEW_FIELDS = [
    "filename", "sha256", "decode_status", "usability", "content_label",
    "reviewer", "decoder_reference",
]
ROW_FIELDS = [
    "filename", "method", "attempted", "baseline_score", "baseline_flagged",
    "score", "flagged", "score_displacement", "linf", "l2", "valid", "status",
    "error", "sha256", "original_sha256", "identical", "seconds", "steps",
]
SCOPE = (
    "Exploratory white-box pilot on the supplied historical training examples. "
    "Repeated methods do not increase the number of independent source samples. "
    "Positive-only samples do not support FPR, AUROC, population confidence "
    "intervals, or held-out generalization claims."
)


def _attempted(row: dict) -> bool:
    return row.get("attempted") is True and str(row.get("method", "")).lower() not in CONTROL_METHODS


def _review_index(review_rows: list[dict] | None) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    allowed = {
        "decode_status": {"", "ok", "failed", "uncertain"},
        "usability": {"", "usable", "unusable", "uncertain"},
        "content_label": {"", "retained", "changed", "uncertain"},
    }
    for row in review_rows or []:
        # A freshly exported template is not review evidence. In particular a
        # missing output has no returned hash, and its blank row must be safe to
        # leave in a partially completed CSV.
        if not any(str(row.get(key) or "").strip() for key in REVIEW_FIELDS[2:]):
            continue
        filename = str(row.get("filename") or "").strip()
        sha = str(row.get("sha256") or "").strip().lower()
        if not filename or not sha:
            raise ValueError("Each review requires both filename and returned sha256")
        clean = {key: str(row.get(key) or "").strip() for key in REVIEW_FIELDS}
        clean["sha256"] = sha
        clean["filename"] = filename
        clean["original_sha256"] = str(row.get("original_sha256") or "").strip().lower()
        for key, values in allowed.items():
            if clean[key] not in values:
                raise ValueError(f"Invalid {key} in review for {filename}: {clean[key]!r}")
        key = (filename, sha)
        if key in index and index[key] != clean:
            raise ValueError(f"Conflicting reviews for {filename} and returned sha256")
        index[key] = clean
    return index


def _outcome(row: dict, reviews: dict[tuple[str, str], dict]) -> dict:
    result = {
        "filename": row.get("filename", ""), "method": row.get("method", ""),
        "sha256": row.get("sha256"), "review_matched": False,
        "review_status": "control", "confirmed": None,
    }
    if not _attempted(row):
        return result
    result["confirmed"] = False
    if row.get("baseline_flagged") is not True:
        result["review_status"] = "baseline_not_flagged" if row.get("baseline_flagged") is False else "baseline_unknown"
        return result
    if row.get("valid") is not True:
        result["review_status"] = "invalid_output"
        return result
    if row.get("flagged") is not False:
        result["review_status"] = "no_flip" if row.get("flagged") is True else "score_missing"
        return result
    result["confirmed"] = None
    key = (str(row.get("filename", "")), str(row.get("sha256") or "").lower())
    review = reviews.get(key)
    if review is None:
        result["review_status"] = "review_missing"
        return result
    result["review_matched"] = True
    # Evidence needs provenance even when the final determination is negative.
    if not review["reviewer"] or not review["decoder_reference"]:
        result["review_status"] = "review_incomplete"
        return result
    negative = (
        ("decode_status", "failed", "decode_failed"),
        ("usability", "unusable", "unusable"),
        ("content_label", "changed", "content_changed"),
    )
    for field, value, status in negative:
        if review[field] == value:
            result.update(review_status=status, confirmed=False)
            return result
    if (review["decode_status"], review["usability"], review["content_label"]) == ("ok", "usable", "retained"):
        result.update(review_status="confirmed", confirmed=True)
    else:
        result["review_status"] = "review_uncertain"
    return result


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _summarize_group(rows: list[dict], outcomes: list[dict]) -> dict:
    attempts = [row for row in rows if _attempted(row)]
    eligible = [row for row in attempts if row.get("baseline_flagged") is True]
    valid = [row for row in eligible if row.get("valid") is True]
    flips = [row for row in valid if row.get("flagged") is False]
    required_outcomes = [
        outcome for outcome in outcomes
        if outcome["review_status"] in {
            "review_missing", "review_incomplete", "review_uncertain", "decode_failed",
            "unusable", "content_changed", "confirmed",
        }
    ]
    pending = sum(outcome["confirmed"] is None for outcome in required_outcomes)
    submitted = sum(outcome["review_matched"] for outcome in required_outcomes)
    confirmed = sum(outcome["confirmed"] is True for outcome in required_outcomes)
    # No review is not evidence of failure. Counts become explicit as review
    # evidence arrives; rates wait until every required review is adjudicated.
    confirmed_count = None if not attempts or (pending and submitted == 0) else confirmed
    complete = bool(attempts) and pending == 0
    return {
        "rows": len(rows),
        "source_samples": len({row.get("filename") for row in rows}),
        "control_rows": len(rows) - len(attempts),
        "baseline_flagged_rows": sum(row.get("baseline_flagged") is True for row in rows),
        "attempted": len(attempts),
        "eligible_attempts": len(eligible),
        "baseline_not_flagged_attempts": sum(row.get("baseline_flagged") is False for row in attempts),
        "baseline_unknown_attempts": sum(row.get("baseline_flagged") not in (True, False) for row in attempts),
        "valid_eligible_attempts": len(valid),
        "invalid_eligible_attempts": len(eligible) - len(valid),
        "detector_flips": len(flips),
        "detector_flip_rate": _rate(len(flips), len(eligible)),
        "detector_flip_rate_all_attempts": _rate(len(flips), len(attempts)),
        "detector_flip_rate_all_rows_including_controls": _rate(len(flips), len(rows)),
        "review_required": len(required_outcomes),
        "review_submitted": submitted,
        "review_adjudicated": len(required_outcomes) - pending,
        "review_pending": pending,
        "confirmed_count": confirmed_count,
        "confirmed_rate": _rate(confirmed, len(eligible)) if complete else None,
        "confirmed_rate_all_attempts": _rate(confirmed, len(attempts)) if complete else None,
        "confirmed_rate_all_rows_including_controls": _rate(confirmed, len(rows)) if complete else None,
        "review_complete": complete,
        "candidate_review_complete": complete,
        "status_counts": dict(sorted(Counter(str(row.get("status", "unknown")) for row in attempts).items())),
        "outcome_counts": dict(sorted(Counter(outcome["review_status"] for outcome in outcomes).items())),
    }


def summarize(rows: list[dict], review_rows: list[dict] | None = None) -> dict:
    """Summarize manifest-complete rows without excluding failed attempts.

    ``attempted`` must be a boolean. Control method names are excluded even if
    accidentally marked attempted. A candidate requires an originally flagged
    sample, a valid saved output, and a returned negative detector decision.
    Human reviews match both filename and saved-output SHA-256. Blank/uncertain
    reviews remain pending. Decisive reviews require reviewer and decoder IDs.
    """
    for row in rows:
        if not isinstance(row.get("attempted"), bool):
            raise ValueError("Every evaluation row requires a boolean attempted field")
    reviews = _review_index(review_rows)
    for row in rows:
        key = (str(row.get("filename", "")), str(row.get("sha256") or "").lower())
        review = reviews.get(key)
        if review and review["original_sha256"]:
            if review["original_sha256"] != str(row.get("original_sha256") or "").lower():
                raise ValueError(f"Review original_sha256 does not match source for {row.get('filename')}")
    outcomes = [_outcome(row, reviews) for row in rows]
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    grouped_outcomes: dict[str, list[dict]] = defaultdict(list)
    for row, outcome in zip(rows, outcomes):
        method = str(row.get("method", "unknown"))
        grouped_rows[method].append(row)
        grouped_outcomes[method].append(outcome)
    observed_keys = {
        (str(row.get("filename", "")), str(row.get("sha256") or "").lower())
        for row in rows if row.get("sha256")
    }
    valid_saved_keys = {
        (str(row.get("filename", "")), str(row.get("sha256") or "").lower())
        for row in rows if row.get("sha256") and row.get("valid") is True
    }
    reviewed_saved_keys = valid_saved_keys & set(reviews)
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "denominator_rule": "Originally flagged actual attempts; invalid, missing and failed outputs remain included. Controls excluded.",
        "review_scope": "review_* and candidate_review_complete describe valid detector-flip candidates only. all_output_* separately accounts for unique valid saved files, including controls.",
        **_summarize_group(rows, outcomes),
        "all_output_unique_valid_files": len(valid_saved_keys),
        "all_output_review_submitted": len(reviewed_saved_keys),
        "all_output_review_not_submitted": len(valid_saved_keys - reviewed_saved_keys),
        "unmatched_review_rows": len(set(reviews) - observed_keys),
        "by_method": {method: _summarize_group(grouped_rows[method], grouped_outcomes[method]) for method in sorted(grouped_rows)},
        "sample_outcomes": outcomes,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display(value: Any) -> str:
    if value is None:
        return "pending / not applicable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_reports(out: Path, rows: list, summary: dict) -> None:
    """Write four local audit artifacts; this function does not submit results.

    ``review_template.csv`` is an internal coordinator template, not a blinded
    assessor view: it contains filenames and hashes but no model scores or
    method labels. The coordinator must independently randomize decoded images
    and keep the mapping hidden from content reviewers.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    outcomes = summary.get("sample_outcomes", [])
    if len(outcomes) != len(rows):
        raise ValueError("Summary sample_outcomes must match the supplied rows")
    annotated = []
    for row, outcome in zip(rows, outcomes):
        if any(row.get(key) != outcome.get(key) for key in ("filename", "method", "sha256")):
            raise ValueError("Summary does not correspond to the supplied row order")
        annotated.append({**row, "review_status": outcome["review_status"], "confirmed": outcome["confirmed"]})
    extra_fields = sorted(set().union(*(set(row) for row in rows)) - set(ROW_FIELDS)) if rows else []
    fields = ROW_FIELDS + [key for key in extra_fields if key not in {"review_status", "confirmed"}] + ["review_status", "confirmed"]
    _write_csv(out / "per_sample.csv", fields, annotated)
    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    review_rows = []
    seen = set()
    for row in rows:
        key = (row.get("filename"), row.get("sha256"))
        if key in seen:
            continue
        seen.add(key)
        review_rows.append({"filename": row.get("filename", ""), "original_sha256": row.get("original_sha256", ""), "sha256": row.get("sha256") or ""})
    _write_csv(out / "review_template.csv", ["filename", "original_sha256", *REVIEW_FIELDS[1:]], review_rows)
    lines = [
        "# PreDecodeGuard pilot report", "", str(summary["scope"]), "",
        "Official verification: " + _display(summary.get("official_verification", {}).get("status", "verified" if summary.get("official_verification", {}).get("exit_code") == 0 else None)), "",
        "Formal perturbations executed: " + _display(summary.get("formal_perturbations_executed")), "",
        str(summary["denominator_rule"]), "",
        "| Method | Attempts | Eligible | Valid | Detector flips | Flip rate | Pending reviews | Confirmed | Final confirmed rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, group in summary.get("by_method", {}).items():
        values = [method.replace("|", "\\|"), group["attempted"], group["eligible_attempts"], group["valid_eligible_attempts"], group["detector_flips"], group["detector_flip_rate"], group["review_pending"], group["confirmed_count"], group["confirmed_rate"]]
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    lines += [
        "", "A detector flip is a candidate until independent decoding and content review are complete. Missing or uncertain review is not a confirmed failure. No-op and control rows do not measure attack success. Candidate review completion does not imply all returned files have been decoded or reviewed.", "",
        f"Detector flips over all input rows, including controls: {summary.get('detector_flips', 0)}/{summary.get('rows', 0)} ({_display(summary.get('detector_flip_rate_all_rows_including_controls'))}). This is separate from the attempt-only rate above.", "",
        f"All-output audit: {summary.get('all_output_review_submitted', 0)} of {summary.get('all_output_unique_valid_files', 0)} unique valid saved files have a submitted review (which may still be uncertain or incomplete).", "",
        "`review_template.csv` is an internal coordinator audit sheet. It contains no scores or method labels, but filenames may reveal source identity. Before human review, randomize decoded-image identifiers and hide this mapping, method labels, scores, and counterpart identity from assessors. Record reviewer and decoder references when merging their decisions.", "",
        f"Unmatched review records (filename/SHA mismatch): {summary.get('unmatched_review_rows', 0)}.", "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
