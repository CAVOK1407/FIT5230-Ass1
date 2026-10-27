"""Read-only reports for the archived Milestone 3 course experiments.

This module reads saved evaluation CSVs. It never loads a model or runs an
attack. Historical results and a new Colab demonstration must be reported
separately: do not concatenate their records before calling these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DETECTORS = ("E", "R", "D", "S")
MAIN_METHODS = ("A5", "A2", "X2", "X4", "ENS", "CAPB_ENS", "EOT_ENS")
METHOD_LABELS = {
    "A5": "Released pipeline (A5)",
    "A2": "No latent optimization (A2)",
    "X2": "Clean control image (X2)",
    "X4": "Reconstruction only (X4)",
    "ENS": "Three-surrogate ensemble (ENS)",
    "CAPB_ENS": "Capped ensemble with bisection (CAPB_ENS)",
    "EOT_ENS": "Ensemble with EOT (EOT_ENS)",
}
HISTORICAL_SCOPE = "Historical course results: saved SD1.5 test PNGs"
COURSE_LIMITATIONS = (
    "This report uses the archived course implementation and existing saved-file "
    "evaluations. It excludes later architecture research. A new Colab example "
    "is a separate run, not a replacement for these historical results.",
    "Conditional attack success counts only source images classified as AI "
    "before attack under the same evaluation condition. Label 0 means AI and "
    "label 1 means real in these course CSVs.",
    "The same source images are evaluated by several detectors and under several "
    "conditions. These rows are correlated observations, not additional "
    "independent samples.",
    "E is the single surrogate; ENS methods optimize E, R and D. S is not used "
    "in the optimization loss, but it comes from the same supplied detector "
    "collection. Its results do not establish broad external-model generalization.",
    "Small fixed-threshold decision differences do not imply that saved pixels "
    "or scores are unchanged, or that the latent path is disconnected.",
    "JPEG and resizing attack rates must be read alongside real-image false "
    "positive rates. A high false positive rate is not evidence of a useful defense.",
    "AUC near 0.5 for the chosen score does not prove distributional equivalence "
    "or inability of other detectors to distinguish the images. AUC near 0 can "
    "indicate reversed score ordering rather than indistinguishability.",
    "Image quality metrics and saved-file norms are measured on the outputs. "
    "A nominal optimizer budget need not be preserved by decoding and PNG "
    "quantization; quality metrics do not establish human semantic preservation.",
    "PreDecodeGuard is a separate exploratory white-box pilot on supplied "
    "historical training examples. Classifier flips are not human-confirmed "
    "content-preserving bypasses. Review remains pending until actual reviews "
    "are received; positive-only samples do not support FPR or AUROC claims.",
)


@dataclass(frozen=True)
class CourseResults:
    data_dir: Path
    summary: pd.DataFrame
    per_image: pd.DataFrame
    real_predictions: pd.DataFrame
    scope: str = HISTORICAL_SCOPE


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _no_duplicate_keys(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    if frame[list(keys)].isna().any().any():
        raise ValueError(f"{name} has missing record keys")
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{name} has duplicate record keys; refusing to inflate counts")


def _counts(results: CourseResults) -> pd.DataFrame:
    frame = results.per_image.assign(
        eligible=results.per_image["clean_pred"].eq(0),
        success=results.per_image["clean_pred"].eq(0)
        & results.per_image["adv_pred"].eq(1),
    )
    counts = frame.groupby(["run", "detector", "condition"], sort=False).agg(
        source_images=("image_id", "nunique"),
        eligible=("eligible", "sum"),
        successes=("success", "sum"),
    ).reset_index()
    counts["conditional_asr"] = counts["successes"].div(
        counts["eligible"].replace(0, np.nan)
    )
    return counts


def _real_counts(results: CourseResults) -> pd.DataFrame:
    return results.real_predictions.assign(
        false_positive=results.real_predictions["pred"].eq(0)
    ).groupby(["detector", "condition"], sort=False).agg(
        real_images=("image", "nunique"),
        real_false_positives=("false_positive", "sum"),
    ).reset_index()


def load_course_results(bundle_root: str | Path) -> CourseResults:
    """Read bundle/historical/*.csv, or an explicitly supplied CSV directory.

    Verify summary counts, conditional ASR and FPR against the saved per-image
    records. No hard-coded result numbers are used. Fail on missing or duplicate
    records instead of silently dropping them.
    """
    root = Path(bundle_root).expanduser().resolve()
    data_dir = root / "historical" if (root / "historical").is_dir() else root
    files = {name: data_dir / name for name in (
        "summary.csv", "per_image.csv", "real_predictions.csv"
    )}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Historical course CSV not found: {path}")
    summary = pd.read_csv(files["summary.csv"])
    per_image = pd.read_csv(files["per_image.csv"], dtype={"image_id": str})
    real = pd.read_csv(files["real_predictions.csv"], dtype={"image": str})
    keys = ("run", "detector", "condition")
    _require_columns(summary, (*keys, "n", "n_clean_detected", "cond_asr", "real_fpr"), "summary.csv")
    _require_columns(per_image, (*keys, "image_id", "clean_pred", "adv_pred"), "per_image.csv")
    _require_columns(real, ("detector", "condition", "image", "pred"), "real_predictions.csv")
    _no_duplicate_keys(summary, keys, "summary.csv")
    _no_duplicate_keys(per_image, (*keys, "image_id"), "per_image.csv")
    _no_duplicate_keys(real, ("detector", "condition", "image"), "real_predictions.csv")
    for frame, columns in ((per_image, ("clean_pred", "adv_pred")), (real, ("pred",))):
        for column in columns:
            if not frame[column].isin([0, 1]).all():
                raise ValueError(f"{column} must contain only binary labels 0 and 1")
    result = CourseResults(data_dir, summary, per_image, real)
    calculated = _counts(result).merge(
        _real_counts(result), on=["detector", "condition"], how="left", validate="many_to_one"
    )
    calculated["calculated_fpr"] = calculated["real_false_positives"] / calculated["real_images"]
    checked = summary.merge(calculated, on=list(keys), how="outer", indicator=True, validate="one_to_one")
    if not checked["_merge"].eq("both").all() or checked["real_images"].isna().any():
        raise ValueError("Historical summary and per-image/real-image evaluation groups do not match")
    for left, right in (("n", "source_images"), ("n_clean_detected", "eligible")):
        if not checked[left].eq(checked[right]).all():
            raise ValueError(f"Historical summary {left} disagrees with per-image counts")
    for left, right in (("cond_asr", "conditional_asr"), ("real_fpr", "calculated_fpr")):
        if not np.allclose(checked[left], checked[right], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"Historical summary {left} disagrees with independently counted rates")
    return result


def _method_counts(results: CourseResults, methods: Sequence[str], condition: str) -> pd.DataFrame:
    counts = _counts(results)
    counts = counts[counts["run"].isin(methods) & counts["condition"].eq(condition)]
    expected = {(method, detector) for method in methods for detector in DETECTORS}
    actual = set(zip(counts["run"], counts["detector"]))
    if actual != expected:
        raise ValueError(f"Historical comparison lacks a requested method/detector for {condition}")
    return counts


def course_main_table(results: CourseResults, methods: Sequence[str] = MAIN_METHODS) -> pd.DataFrame:
    """Success count / eligible count and percentage, for original saved PNGs."""
    counts = _method_counts(results, methods, "none")
    counts = counts.copy()
    counts["conditional ASR"] = [
        f"{int(row.successes)}/{int(row.eligible)} ({row.conditional_asr:.1%})"
        if row.eligible else "0/0 (not estimable)"
        for row in counts.itertuples()
    ]
    table = counts.pivot(index="run", columns="detector", values="conditional ASR")
    table = table.reindex(index=list(methods), columns=list(DETECTORS))
    table.index.name = "Method"
    table.columns.name = None
    table.attrs["scope"] = results.scope
    table.attrs["denominator"] = "Originally detected source images, independently for each detector"
    return table


def jpeg75_table(results: CourseResults, methods: Sequence[str] = ("ENS", "EOT_ENS")) -> pd.DataFrame:
    """JPEG75 success numerators/denominators with real-image false positives."""
    table = _method_counts(results, methods, "jpeg75").merge(
        _real_counts(results), on=["detector", "condition"], validate="many_to_one"
    )
    table["real_fpr"] = table["real_false_positives"] / table["real_images"]
    table["run"] = pd.Categorical(table["run"], categories=methods, ordered=True)
    table["detector"] = pd.Categorical(table["detector"], categories=DETECTORS, ordered=True)
    table = table.sort_values(["run", "detector"])[[
        "run", "detector", "successes", "eligible", "conditional_asr",
        "real_false_positives", "real_images", "real_fpr",
    ]].reset_index(drop=True)
    table.attrs["scope"] = results.scope + "; JPEG75 evaluation"
    return table


def quality_table(results: CourseResults, methods: Sequence[str] = MAIN_METHODS) -> pd.DataFrame:
    """Mean saved-PNG quality/cost, counting each image once (not four times)."""
    metrics = ("psnr", "ssim", "lpips", "linf", "seconds")
    _require_columns(results.per_image, metrics, "per_image.csv")
    _method_counts(results, methods, "none")
    selected = results.per_image[
        results.per_image["run"].isin(methods)
        & results.per_image["condition"].eq("none")
    ]
    # Evaluators repeat image-level quality and timing for each detector.
    for metric in metrics:
        if selected.groupby(["run", "image_id"])[metric].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Image-level {metric} is inconsistent across detector rows")
    selected = selected[selected["detector"].eq("E")]
    table = selected.groupby("run").agg(
        source_images=("image_id", "nunique"),
        mean_psnr_db=("psnr", "mean"), mean_ssim=("ssim", "mean"),
        mean_lpips=("lpips", "mean"), mean_saved_linf=("linf", "mean"),
        max_saved_linf=("linf", "max"), mean_seconds=("seconds", "mean"),
    ).reindex(list(methods))
    table.index.name = "Method"
    table.attrs["scope"] = results.scope
    table.attrs["timing"] = "Historical local runtime; not a controlled cross-hardware benchmark"
    return table


def paired_comparisons(
    results: CourseResults,
    pairs: Sequence[tuple[str, str]] = (("A5", "A2"), ("A5", "X2")),
    condition: str = "none",
) -> pd.DataFrame:
    """Paired all-source decision changes, not a pooled independent-sample ASR."""
    rows = []
    frame = results.per_image[results.per_image["condition"].eq(condition)]
    for before, after in pairs:
        left = frame[frame["run"].eq(before)].set_index(["image_id", "detector"])
        right = frame[frame["run"].eq(after)].set_index(["image_id", "detector"])
        if left.empty or right.empty or set(left.index) != set(right.index):
            raise ValueError(f"Paired source/detector records do not match: {before} versus {after}")
        right = right.reindex(left.index)
        if not left["clean_pred"].eq(right["clean_pred"]).all():
            raise ValueError(f"Clean decisions differ between paired runs: {before} versus {after}")
        changed = left["adv_pred"].ne(right["adv_pred"])
        row = {
            "comparison": f"{before} -> {after}",
            "condition": condition,
            "source_images": left.index.get_level_values("image_id").nunique(),
            "paired_decisions": len(left),
            "changed_decisions": int(changed.sum()),
            "affected_source_images": left.index[changed].get_level_values("image_id").nunique(),
        }
        for detector in DETECTORS:
            detector_rows = left.index.get_level_values("detector") == detector
            row[f"{detector}_changed"] = int(changed[detector_rows].sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["scope"] = results.scope
    table.attrs["warning"] = "Several paired decisions belong to each source image; do not treat them as independent samples"
    return table


def provenance_summary(results: CourseResults) -> dict:
    """Source hashes and honest counts for notebook display or saved provenance."""
    files = []
    for name, frame in (("summary.csv", results.summary), ("per_image.csv", results.per_image),
                        ("real_predictions.csv", results.real_predictions)):
        path = results.data_dir / name
        files.append({"file": name, "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {
        "scope": results.scope,
        "source_directory": str(results.data_dir),
        "distinct_generated_source_ids": int(results.per_image["image_id"].nunique()),
        "distinct_real_source_ids": int(results.real_predictions["image"].nunique()),
        "evaluated_configurations": int(results.per_image["run"].nunique()),
        "detectors": list(DETECTORS),
        "conditions": sorted(results.per_image["condition"].unique().tolist()),
        "new_model_execution": False,
        "files": files,
        "limitations": list(COURSE_LIMITATIONS),
    }


def _real_positive_auc(real_scores: np.ndarray, generated_scores: np.ndarray) -> float:
    """P(real score > generated score) + half the probability of a tie.

    This is the rank form of the original m2_evaluate.auc definition: real
    photographs are positives and a higher margin means more real. It uses
    every image in a group, not just the sources eligible for conditional ASR.
    """
    real_scores = np.asarray(real_scores, dtype=float)
    generated_scores = np.asarray(generated_scores, dtype=float)
    if not len(real_scores) or not len(generated_scores):
        return float("nan")
    if not np.isfinite(real_scores).all() or not np.isfinite(generated_scores).all():
        raise ValueError("AUC audit requires finite real and generated margins")
    ordered = np.sort(generated_scores)
    below = np.searchsorted(ordered, real_scores, side="left")
    at_or_below = np.searchsorted(ordered, real_scores, side="right")
    return float(np.sum(below + 0.5 * (at_or_below - below))
                 / (len(real_scores) * len(generated_scores)))


def audit_score_auc(results: CourseResults) -> dict:
    """Recompute both AUC columns from saved margins, without scikit-learn.

    Real photos are the positive class, matching m2_evaluate.auc. Each
    run/detector/condition uses all its generated images and the matching real
    image scores. Ties receive half credit. Refuse unmatched groups, non-finite
    scores or a summary discrepancy exceeding 1e-12; this is an audit of
    historical evaluations, not another model execution.
    """
    keys = ["run", "detector", "condition"]
    columns = {"auc_real_vs_adv": "adv_margin", "auc_real_vs_clean": "clean_margin"}
    _require_columns(results.summary, [*keys, *columns], "summary.csv")
    _require_columns(results.per_image, [*keys, *columns.values()], "per_image.csv")
    _require_columns(results.real_predictions, ["detector", "condition", "real_margin"], "real_predictions.csv")
    _no_duplicate_keys(results.summary, keys, "summary.csv")
    groups = results.per_image.groupby(keys, sort=False)
    summary = results.summary.set_index(keys)
    if not len(summary) or set(groups.groups) != set(summary.index):
        raise ValueError("AUC audit requires the same non-empty summary and per-image groups")
    real_groups = {
        key: frame["real_margin"].to_numpy(dtype=float)
        for key, frame in results.real_predictions.groupby(["detector", "condition"], sort=False)
    }
    errors = {column: 0.0 for column in columns}
    checked = 0
    for (run, detector, condition), frame in groups:
        real = real_groups.get((detector, condition))
        if real is None:
            raise ValueError("AUC audit is missing a matching real-image score group")
        for column, margin in columns.items():
            calculated = _real_positive_auc(real, frame[margin].to_numpy(dtype=float))
            recorded = float(summary.loc[(run, detector, condition), column])
            if not np.isfinite(calculated) or not np.isfinite(recorded):
                raise ValueError("AUC audit requires estimable recorded and recalculated AUCs")
            error = abs(calculated - recorded)
            errors[column] = max(errors[column], float(error))
            if error > 1e-12:
                raise ValueError(f"Historical {column} disagrees with saved margins for {run}/{detector}/{condition}")
        checked += 1
    return {"groups_checked": checked, "max_abs_errors": errors}


def peer_summary_table(peer_source: str | Path | Mapping) -> pd.DataFrame:
    """Return aggregate-only peer status; never expose names, scores or mappings.

    Prefer a sanitized bundle/peer/summary.json. Even if a complete internal
    summary is supplied, only the explicit aggregate fields below are used.
    """
    if isinstance(peer_source, Mapping):
        summary = peer_source
    else:
        path = Path(peer_source)
        if path.is_dir():
            path = path / "peer" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
    fields = ("source_samples", "control_rows", "attempted", "eligible_attempts", "detector_flips")
    values = {}
    for name in fields:
        value = summary.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Peer aggregate {name} must be a non-negative integer")
        values[name] = value
    if values["detector_flips"] > values["eligible_attempts"] or values["eligible_attempts"] > values["attempted"]:
        raise ValueError("Peer aggregate flip/attempt counts are inconsistent")
    if values["attempted"] + values["control_rows"] != values["source_samples"]:
        raise ValueError("Peer attempts and controls do not account for all source samples")
    completed = summary.get("review_complete") is True
    confirmed = summary.get("confirmed_count")
    if completed and (isinstance(confirmed, bool) or not isinstance(confirmed, int)
                      or not 0 <= confirmed <= values["detector_flips"]):
        raise ValueError("Completed peer review requires a valid aggregate confirmed count")
    rows = [
        ("Track", "Exploratory white-box latent-access pilot"),
        ("Source samples", values["source_samples"]),
        ("Attack attempts", values["attempted"]),
        ("Unmodified controls", values["control_rows"]),
        ("Classifier flips / eligible attempts", f"{values['detector_flips']}/{values['eligible_attempts']}"),
        ("Human content review", "Complete" if completed else "Pending"),
        ("Human-confirmed bypasses", f"{confirmed}/{values['eligible_attempts']}" if completed else "Pending; not yet an attack-success claim"),
    ]
    table = pd.DataFrame(rows, columns=["Measure", "Status"])
    table.attrs["scope"] = "Aggregate-only PreDecodeGuard pilot status"
    return table


def plot_course_results(results: CourseResults) -> dict:
    """Return matplotlib figures; no files are written and no model is loaded."""
    import matplotlib.pyplot as plt

    figures = {}
    counts = _method_counts(results, MAIN_METHODS, "none")
    rates = counts.pivot(index="run", columns="detector", values="conditional_asr").reindex(MAIN_METHODS)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    rates[list(DETECTORS)].plot.bar(ax=ax, width=0.8, rot=20)
    ax.set(title="Historical course results: conditional ASR on saved PNGs",
           xlabel="Method", ylabel="Conditional attack success rate", ylim=(0, 1.08))
    ax.legend(title="Detector", ncol=4, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    figures["saved_png_asr"] = fig

    jpeg = jpeg75_table(results)
    rates = jpeg.pivot(index="detector", columns="run", values="conditional_asr").reindex(DETECTORS)
    fpr = jpeg.drop_duplicates("detector").set_index("detector")["real_fpr"].reindex(DETECTORS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates.plot.bar(ax=ax, width=0.7, rot=0)
    ax.plot(np.arange(len(DETECTORS)), fpr.to_numpy(), "o--", color="#A83B32", label="Real-image FPR")
    ax.set(title="Historical JPEG75 results: attack success and false positives",
           xlabel="Detector", ylabel="Rate", ylim=(0, 1.08))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    figures["jpeg75_with_fpr"] = fig

    quality = quality_table(results)
    held_out = counts[counts["detector"].eq("S")].set_index("run")["conditional_asr"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in MAIN_METHODS:
        x, y = quality.loc[method, "mean_lpips"], held_out.loc[method]
        ax.scatter(x, y, s=45)
        ax.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set(title="Historical saved PNGs: perceptual distortion and held-out S",
           xlabel="Mean LPIPS across all source images (lower is better)",
           ylabel="S conditional attack success rate", ylim=(0, 1.08))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    figures["quality_vs_held_out"] = fig
    return figures
