# LAO / PGD / JPEG validation

All 50 generated samples are retained. Budget: saved RGB uint8 maximum difference ≤8; quality: PSNR≥30 and LPIPS≤0.03; mean LPIPS guard≤0.015.

Primary E cohort: clean detected under every condition (none, JPEG75, JPEG85, JPEG95). Success requires the same output to be classified real under all four conditions and pass original-PNG quality.

| Method | Joint quality success / eligible | Mean LPIPS | Mean guard |
|---|---:|---:|---|
| Z | 0/33 | 0.004571 | True |
| P | 1/33 | 0.002388 | True |
| L | 3/33 | 0.008483 | True |
| H | 3/33 | 0.004388 | True |
| P_EOT | 3/33 | 0.004453 | True |
| L_EOT | 4/33 | 0.012426 | True |
| H_EOT | 5/33 | 0.008289 | True |

Primary H_EOT − P_EOT: 6.06%; paired bootstrap 95% CI [-9.09%, 21.21%]; exact McNemar p=0.6875. Added/lost successes: 4/2. Predeclared superiority criterion met: False.

| Detector | Condition | Real false positives / controls | FPR |
|---|---|---:|---:|
| E | none | 1/50 | 2.00% |
| E | jpeg75 | 3/50 | 6.00% |
| E | jpeg85 | 4/50 | 8.00% |
| E | jpeg95 | 1/50 | 2.00% |
| R | none | 0/50 | 0.00% |
| R | jpeg75 | 14/50 | 28.00% |
| R | jpeg85 | 10/50 | 20.00% |
| R | jpeg95 | 3/50 | 6.00% |
| D | none | 1/50 | 2.00% |
| D | jpeg75 | 13/50 | 26.00% |
| D | jpeg85 | 9/50 | 18.00% |
| D | jpeg95 | 3/50 | 6.00% |
| S | none | 0/50 | 0.00% |
| S | jpeg75 | 27/50 | 54.00% |
| S | jpeg85 | 18/50 | 36.00% |
| S | jpeg95 | 11/50 | 22.00% |

Interpretation limits:

- Predictions and LPIPS are read from independent saved-PNG evaluation; this report never loads a model.
- PNG budgets and PSNR are recomputed; LPIPS is cross-checked with records when available, not independently re-executed.
- Only the fixed E joint-condition H_EOT versus P_EOT comparison is primary; secondary/descriptive tests are not multiplicity-adjusted.
- Mean LPIPS <= .015 is a method-level guard, not a reason to remove any individual sample.
- Every transform uses original-PNG quality; the integer budget applies before JPEG, not to compressed pixels.
- FPR must be read beside ASR. Nonzero robust success alone does not establish improvement or a useful detector operating point.
- Joint robustness concerns only four fixed JPEG/identity conditions, not arbitrary transformations.
- This audit checks source consistency between outputs, not unseen-data provenance, source-code freezing, or human semantic fidelity.
- Standalone L/P/H comparisons do not establish adaptive causal stage dominance without matched renderer controls and reoptimization.
