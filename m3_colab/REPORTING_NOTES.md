# Course reporting helper

`m3_reporting.py` only reads historical saved-file evaluations. It does not load
models, train weights, generate images, or run perturbation optimization.

Expected bundle layout:

```text
bundle/
  historical/
    summary.csv
    per_image.csv
    real_predictions.csv
  peer/
    summary.json
```

Use the pre-architecture course archive as the source. Keep new Colab demo
outputs in a separate directory and label them as a new run. They must not be
merged into these historical results. Later architecture research is excluded.

```python
from m3_reporting import (
    load_course_results, course_main_table, jpeg75_table, quality_table,
    paired_comparisons, provenance_summary, peer_summary_table,
    plot_course_results, audit_score_auc, COURSE_LIMITATIONS,
)

historical = load_course_results(BUNDLE)
display(course_main_table(historical))
display(jpeg75_table(historical))
display(quality_table(historical))
display(paired_comparisons(historical))
display(peer_summary_table(BUNDLE))
provenance = provenance_summary(historical)
auc_audit = audit_score_auc(historical)
figures = plot_course_results(historical)
```

The loader independently checks summary sample counts, conditional ASR and
real-image FPR against the per-image CSVs. It rejects duplicate or mismatched
records. Quality and cost are summarized once per source image, not once per
detector. Paired changes use exact source/detector matches and do not silently
drop missing records. Reported denominators differ by detector and condition.
`audit_score_auc` separately recomputes both AUC columns from all saved scores
in every evaluation group, using real photographs as positives and half credit
for ties, as in the original evaluator. It needs only NumPy/pandas, reports
the number of checked groups and maximum absolute errors, and rejects a
discrepancy greater than 1e-12.

The peer JSON should contain only these aggregate fields:
`source_samples`, `control_rows`, `attempted`, `eligible_attempts`,
`detector_flips`, `review_complete`, `confirmed_count`.
Do not distribute internal `sample_outcomes`, filenames, scores, traces,
original-file hashes, or control/filename mappings. The helper only displays
aggregate counts; incomplete human review is labelled pending and cannot be
turned into a confirmed success claim.

The course rubric requires **one distinct peer strategy per person**. Its three
individual-strategy rows are grading dimensions, not three required strategies.
There is no fixed requirement for three robustness tests. This helper does not
infer individual ownership or fabricate peer feedback.
