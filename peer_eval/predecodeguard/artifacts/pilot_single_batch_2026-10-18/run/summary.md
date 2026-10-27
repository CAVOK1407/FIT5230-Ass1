# PreDecodeGuard pilot report

Exploratory white-box pilot on the supplied historical training examples. Repeated methods do not increase the number of independent source samples. Positive-only samples do not support FPR, AUROC, population confidence intervals, or held-out generalization claims.

Official verification: verified

Formal perturbations executed: True

Originally flagged actual attempts; invalid, missing and failed outputs remain included. Controls excluded.

| Method | Attempts | Eligible | Valid | Detector flips | Flip rate | Pending reviews | Confirmed | Final confirmed rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstop | 8 | 8 | 8 | 8 | 1.0000 | 8 | pending / not applicable | pending / not applicable |

A detector flip is a candidate until independent decoding and content review are complete. Missing or uncertain review is not a confirmed failure. No-op and control rows do not measure attack success. Candidate review completion does not imply all returned files have been decoded or reviewed.

Detector flips over all input rows, including controls: 8/10 (0.8000). This is separate from the attempt-only rate above.

All-output audit: 0 of 10 unique valid saved files have a submitted review (which may still be uncertain or incomplete).

`review_template.csv` is an internal coordinator audit sheet. It contains no scores or method labels, but filenames may reveal source identity. Before human review, randomize decoded-image identifiers and hide this mapping, method labels, scores, and counterpart identity from assessors. Record reviewer and decoder references when merging their decisions.

Unmatched review records (filename/SHA mismatch): 0.
