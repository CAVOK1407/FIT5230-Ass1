# Frozen LAO: new-image, matched saved-budget validation

Status: **verified**. Generated images: 50; real images: 50. Saved uint8 L-infinity budget: **8/255, zero tolerance**.

## Primary comparison: quality-qualified LAO8 versus PGD40, E / none

| n_clean_detected | left_success | right_success | difference | ci95_lower | ci95_upper | conservative_ci95_lower | conservative_ci95_upper | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | 27 | 30 | -0.0698 | -0.2791 | 0.1395 | -0.3689 | 0.2423 | 0.6636 |

Superiority supported: **False**. Descriptive 5-point noninferiority criterion supported: **False**.

## Interpretation and limits

- New-image validation of a frozen residual-rendering architecture. All 100 canonical file/pixel hashes, planned source choices, generation plan hash, internal uniqueness and exclusion against old canonical inputs were verified; no image may be selected by clean or attack scores.
- RAW_LAO and RAW_RECON are diagnostic-only and are not admissible competitors in the 8/255 budget comparison.
- LAO8 is exactly the integer-budget projection of the frozen RAW_LAO PNG. There is no budget-aware latent re-optimisation in this round; projection is not inside the LAO objective.
- PGD10 uses quantised-save-forward margin(kappa=3)+LPIPS; PGD40 is the stronger CE reference with three starts and E-only iterate/restart selection. These are deliberately not identical optimisation objectives or equal compute budgets.
- R/D/S and transformed-image results are descriptive. They must not be used for selecting methods, iterations, restarts, thresholds or output files.
- Success rates retain every clean-detected source image. A budget violation counts as failure, never exclusion; classifier_flip_asr and uncond_escape additionally expose the raw classifier decisions.
- The fixed quality screen is PSNR >= 30 dB and LPIPS <= 0.15 on the unprocessed saved PNG. It is not a semantic or human-review verdict; all 50 images remain in every full-sample table.
- Budget is measured before JPEG/resize. For secondary transforms the reported quality screen still describes the original PNG, and real-image FPR is reported under that same transform.
- Primary outcome is quality-qualified LAO8 minus PGD40 on E/none, with a common clean-detected denominator, paired bootstrap CI and two-sided exact McNemar test.
- Superiority requires the primary bootstrap 95% CI lower bound > 0 and exact p < 0.05, with no invalid bounded artifact. Fixed-margin noninferiority is descriptive only and uses the conservative exact lower bound > -0.05, not the bootstrap bound.
- The conservative paired-risk-difference interval is [lo(LAO-only)-hi(PGD-only), hi(LAO-only)-lo(PGD-only)], using two separate 97.5% Clopper-Pearson intervals and Bonferroni coverage of at least 95%. It remains nondegenerate when all outcomes agree; bootstrap [0,0] alone cannot establish 5-point noninferiority. Passing noninferiority alone does not establish usefulness when both methods fail.
- Secondary tests and descriptive transfer cells are not multiplicity-adjusted confirmatory claims. Detector cells on the same images are correlated and must not be pooled as independent observations.
- The 50 images use the same ten prompt categories with new seeds; these intervals do not establish generalisation to unseen prompts, generator families or detector families.
- Zero-step and reset controls establish latent contribution within this residual architecture. Removing PGD does not establish LAO dominance in the original hybrid pipeline.
- Reported per-image times follow each run's records and can include diagnostic work. Ten LAO updates and ten PGD updates are not equal-cost computations.
- This reporter reads saved files only; it executes no model, training or attack and does not replace the independent source/data provenance checks.

## Budget-constrained methods: all samples retained

| run | detector | condition | n | n_clean_detected | n_valid_success | cond_asr | n_quality_qualified_success | quality_qualified_asr | real_fpr | n_budget_violations | psnr | lpips | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAO8 | E | none | 50 | 43 | 27 | 0.6279 | 27 | 0.6279 | 0.0000 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | E | jpeg75 | 50 | 39 | 8 | 0.2051 | 8 | 0.2051 | 0.0800 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | E | resize05 | 50 | 48 | 1 | 0.0208 | 1 | 0.0208 | 0.9000 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | R | none | 50 | 43 | 3 | 0.0698 | 3 | 0.0698 | 0.0000 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | R | jpeg75 | 50 | 46 | 2 | 0.0435 | 2 | 0.0435 | 0.2200 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | R | resize05 | 50 | 49 | 2 | 0.0408 | 2 | 0.0408 | 0.9400 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | D | none | 50 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0400 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | D | jpeg75 | 50 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.2400 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | D | resize05 | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.9800 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | S | none | 50 | 50 | 7 | 0.1400 | 7 | 0.1400 | 0.0000 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | S | jpeg75 | 50 | 48 | 1 | 0.0208 | 1 | 0.0208 | 0.5600 | 0 | 36.9901 | 0.0067 | 56.2282 |
| LAO8 | S | resize05 | 50 | 16 | 12 | 0.7500 | 12 | 0.7500 | 0.2000 | 0 | 36.9901 | 0.0067 | 56.2282 |
| RECON8 | E | none | 50 | 43 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | E | jpeg75 | 50 | 39 | 0 | 0.0000 | 0 | 0.0000 | 0.0800 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | E | resize05 | 50 | 48 | 0 | 0.0000 | 0 | 0.0000 | 0.9000 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | R | none | 50 | 43 | 4 | 0.0930 | 4 | 0.0930 | 0.0000 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | R | jpeg75 | 50 | 46 | 1 | 0.0217 | 1 | 0.0217 | 0.2200 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | R | resize05 | 50 | 49 | 1 | 0.0204 | 1 | 0.0204 | 0.9400 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | D | none | 50 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0400 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | D | jpeg75 | 50 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.2400 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | D | resize05 | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.9800 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | S | none | 50 | 50 | 6 | 0.1200 | 6 | 0.1200 | 0.0000 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | S | jpeg75 | 50 | 48 | 0 | 0.0000 | 0 | 0.0000 | 0.5600 | 0 | 39.7870 | 0.0037 | 2.9756 |
| RECON8 | S | resize05 | 50 | 16 | 8 | 0.5000 | 8 | 0.5000 | 0.2000 | 0 | 39.7870 | 0.0037 | 2.9756 |
| PGD10 | E | none | 50 | 43 | 43 | 1.0000 | 41 | 0.9535 | 0.0000 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | E | jpeg75 | 50 | 39 | 21 | 0.5385 | 21 | 0.5385 | 0.0800 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | E | resize05 | 50 | 48 | 1 | 0.0208 | 1 | 0.0208 | 0.9000 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | R | none | 50 | 43 | 23 | 0.5349 | 23 | 0.5349 | 0.0000 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | R | jpeg75 | 50 | 46 | 3 | 0.0652 | 3 | 0.0652 | 0.2200 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | R | resize05 | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.9400 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | D | none | 50 | 50 | 6 | 0.1200 | 6 | 0.1200 | 0.0400 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | D | jpeg75 | 50 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.2400 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | D | resize05 | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.9800 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | S | none | 50 | 50 | 48 | 0.9600 | 46 | 0.9200 | 0.0000 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | S | jpeg75 | 50 | 48 | 0 | 0.0000 | 0 | 0.0000 | 0.5600 | 0 | inf | 0.0678 | 0.7815 |
| PGD10 | S | resize05 | 50 | 16 | 8 | 0.5000 | 7 | 0.4375 | 0.2000 | 0 | inf | 0.0678 | 0.7815 |
| PGD40 | E | none | 50 | 43 | 43 | 1.0000 | 30 | 0.6977 | 0.0000 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | E | jpeg75 | 50 | 39 | 39 | 1.0000 | 26 | 0.6667 | 0.0800 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | E | resize05 | 50 | 48 | 1 | 0.0208 | 1 | 0.0208 | 0.9000 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | R | none | 50 | 43 | 39 | 0.9070 | 25 | 0.5814 | 0.0000 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | R | jpeg75 | 50 | 46 | 7 | 0.1522 | 6 | 0.1304 | 0.2200 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | R | resize05 | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.9400 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | D | none | 50 | 50 | 19 | 0.3800 | 10 | 0.2000 | 0.0400 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | D | jpeg75 | 50 | 50 | 1 | 0.0200 | 0 | 0.0000 | 0.2400 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | D | resize05 | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.9800 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | S | none | 50 | 50 | 49 | 0.9800 | 34 | 0.6800 | 0.0000 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | S | jpeg75 | 50 | 48 | 1 | 0.0208 | 1 | 0.0208 | 0.5600 | 0 | 33.3694 | 0.1352 | 5.7510 |
| PGD40 | S | resize05 | 50 | 16 | 9 | 0.5625 | 4 | 0.2500 | 0.2000 | 0 | 33.3694 | 0.1352 | 5.7510 |

## Raw outputs: diagnostic only

| run | detector | n_clean_detected | n_classifier_success | classifier_flip_asr | psnr | lpips | linf | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW_LAO | E | 43 | 28 | 0.6512 | 35.2855 | 0.0076 | 0.1866 | 56.2282 |
| RAW_LAO | R | 43 | 3 | 0.0698 | 35.2855 | 0.0076 | 0.1866 | 56.2282 |
| RAW_LAO | D | 50 | 0 | 0.0000 | 35.2855 | 0.0076 | 0.1866 | 56.2282 |
| RAW_LAO | S | 50 | 10 | 0.2000 | 35.2855 | 0.0076 | 0.1866 | 56.2282 |
| RAW_RECON | E | 43 | 1 | 0.0233 | 38.5465 | 0.0042 | 0.1747 | 2.9756 |
| RAW_RECON | R | 43 | 4 | 0.0930 | 38.5465 | 0.0042 | 0.1747 | 2.9756 |
| RAW_RECON | D | 50 | 0 | 0.0000 | 38.5465 | 0.0042 | 0.1747 | 2.9756 |
| RAW_RECON | S | 50 | 6 | 0.1200 | 38.5465 | 0.0042 | 0.1747 | 2.9756 |

## Paired quality-qualified comparisons

| comparator | detector | condition | n_clean_detected | added_success | lost_success | difference | ci95_lower | ci95_upper | conservative_ci95_lower | conservative_ci95_upper | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PGD40 | E | none | 43 | 9 | 12 | -0.0698 | -0.2791 | 0.1395 | -0.3689 | 0.2423 | 0.6636 |
| PGD40 | E | jpeg75 | 39 | 2 | 20 | -0.4615 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | E | resize05 | 48 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | R | none | 43 | 0 | 22 | -0.5116 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | R | jpeg75 | 46 | 0 | 4 | -0.0870 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | R | resize05 | 49 | 2 | 0 | 0.0408 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | D | none | 50 | 0 | 10 | -0.2000 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | S | none | 50 | 2 | 29 | -0.5400 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | S | jpeg75 | 48 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| PGD40 | S | resize05 | 16 | 8 | 0 | 0.5000 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | E | none | 43 | 1 | 15 | -0.3256 | -0.4884 | -0.1628 | -0.5304 | -0.0546 | 0.0005 |
| PGD10 | E | jpeg75 | 39 | 0 | 13 | -0.3333 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | E | resize05 | 48 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | R | none | 43 | 0 | 20 | -0.4651 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | R | jpeg75 | 46 | 1 | 2 | -0.0217 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | R | resize05 | 49 | 2 | 0 | 0.0408 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | D | none | 50 | 0 | 6 | -0.1200 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | S | none | 50 | 0 | 39 | -0.7800 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | S | jpeg75 | 48 | 1 | 0 | 0.0208 | n/a | n/a | n/a | n/a | n/a |
| PGD10 | S | resize05 | 16 | 6 | 1 | 0.3125 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | E | none | 43 | 27 | 0 | 0.6279 | 0.4884 | 0.7674 | 0.3491 | 0.7869 | 0.0000 |
| RECON8 | E | jpeg75 | 39 | 8 | 0 | 0.2051 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | E | resize05 | 48 | 1 | 0 | 0.0208 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | R | none | 43 | 0 | 1 | -0.0233 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | R | jpeg75 | 46 | 1 | 0 | 0.0217 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | R | resize05 | 49 | 1 | 0 | 0.0204 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | D | none | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | S | none | 50 | 2 | 1 | 0.0200 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | S | jpeg75 | 48 | 1 | 0 | 0.0208 | n/a | n/a | n/a | n/a | n/a |
| RECON8 | S | resize05 | 16 | 5 | 1 | 0.2500 | n/a | n/a | n/a | n/a | n/a |

## Integrity and latent diagnostics

Reset checks: **100/100** identical to independently run reconstruction baselines. Bounded-output violations: **0**. Integer export projections were compared pixel for pixel with the raw outputs and canonical originals.

| image_id | steps | finite_gradient_steps | nonzero_gradient_steps | nonzero_update_steps | first_gradient_l2 | last_gradient_l2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0000 | 10 | 10 | 10 | 10 | 3.2241 | 2.1320 |
| 0001 | 10 | 10 | 10 | 10 | 2.9026 | 0.0023 |
| 0002 | 10 | 10 | 10 | 10 | 1.8912 | 1.4294 |
| 0003 | 10 | 10 | 10 | 10 | 0.0025 | 0.0004 |
| 0004 | 10 | 10 | 10 | 10 | 2.3775 | 0.0031 |
| 0005 | 10 | 10 | 10 | 10 | 1.7988 | 2.6165 |
| 0006 | 10 | 10 | 10 | 10 | 3.4121 | 3.1215 |
| 0007 | 10 | 10 | 10 | 10 | 2.3067 | 2.5989 |
| 0008 | 10 | 10 | 10 | 10 | 3.0984 | 3.0325 |
| 0009 | 10 | 10 | 10 | 10 | 1.9615 | 1.3618 |
| 0010 | 10 | 10 | 10 | 10 | 4.1100 | 2.3205 |
| 0011 | 10 | 10 | 10 | 10 | 4.1509 | 3.0638 |
| 0012 | 10 | 10 | 10 | 10 | 2.1687 | 1.8752 |
| 0013 | 10 | 10 | 10 | 10 | 2.2937 | 0.0019 |
| 0014 | 10 | 10 | 10 | 10 | 2.1114 | 2.1090 |
| 0015 | 10 | 10 | 10 | 10 | 1.8633 | 0.9952 |
| 0016 | 10 | 10 | 10 | 10 | 2.9833 | 0.0020 |
| 0017 | 10 | 10 | 10 | 10 | 3.3974 | 0.0027 |
| 0018 | 10 | 10 | 10 | 10 | 2.8050 | 0.0041 |
| 0019 | 10 | 10 | 10 | 10 | 1.8607 | 0.0026 |
| 0020 | 10 | 10 | 10 | 10 | 2.0794 | 1.7088 |
| 0021 | 10 | 10 | 10 | 10 | 3.0445 | 2.2905 |
| 0022 | 10 | 10 | 10 | 10 | 2.1518 | 120.2576 |
| 0023 | 10 | 10 | 10 | 10 | 4.3394 | 0.0028 |
| 0024 | 10 | 10 | 10 | 10 | 3.5527 | 2.6539 |
| 0025 | 10 | 10 | 10 | 10 | 2.1289 | 3.3976 |
| 0026 | 10 | 10 | 10 | 10 | 2.2468 | 0.0014 |
| 0027 | 10 | 10 | 10 | 10 | 3.0445 | 0.0046 |
| 0028 | 10 | 10 | 10 | 10 | 8.3549 | 2.8926 |
| 0029 | 10 | 10 | 10 | 10 | 1.2375 | 0.7257 |
| 0030 | 10 | 10 | 10 | 10 | 1.6846 | 2.4361 |
| 0031 | 10 | 10 | 10 | 10 | 4.7408 | 2.0503 |
| 0032 | 10 | 10 | 10 | 10 | 2.0337 | 1.7484 |
| 0033 | 10 | 10 | 10 | 10 | 0.0015 | 0.0006 |
| 0034 | 10 | 10 | 10 | 10 | 2.1378 | 0.0020 |
| 0035 | 10 | 10 | 10 | 10 | 2.0769 | 1.6322 |
| 0036 | 10 | 10 | 10 | 10 | 2.6150 | 0.0029 |
| 0037 | 10 | 10 | 10 | 10 | 2.5198 | 2.3027 |
| 0038 | 10 | 10 | 10 | 10 | 5.3105 | 0.0062 |
| 0039 | 10 | 10 | 10 | 10 | 3.0051 | 0.0051 |
| 0040 | 10 | 10 | 10 | 10 | 2.2170 | 1.4722 |
| 0041 | 10 | 10 | 10 | 10 | 3.5204 | 2.0982 |
| 0042 | 10 | 10 | 10 | 10 | 2.7594 | 2.5780 |
| 0043 | 10 | 10 | 10 | 10 | 2.8624 | 0.0029 |
| 0044 | 10 | 10 | 10 | 10 | 2.6619 | 1.6809 |
| 0045 | 10 | 10 | 10 | 10 | 2.1647 | 7.6912 |
| 0046 | 10 | 10 | 10 | 10 | 3.3013 | 2.4033 |
| 0047 | 10 | 10 | 10 | 10 | 5.2723 | 3.4311 |
| 0048 | 10 | 10 | 10 | 10 | 2.7708 | 3.4886 |
| 0049 | 10 | 10 | 10 | 10 | 2.2357 | 0.0031 |
