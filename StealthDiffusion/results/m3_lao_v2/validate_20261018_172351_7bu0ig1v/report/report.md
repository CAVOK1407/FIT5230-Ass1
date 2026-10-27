# LAO v2: frozen low-distortion candidate on 50 new images

Status: **verified**. Generated images: 50; real images: 50. Saved uint8 L-infinity budget: **8/255, zero tolerance**.

## Primary: NEW_LAO8 minus OLD_LAO8, E / original PNG

Low-distortion success requires PSNR ≥ 30 dB and LPIPS ≤ 0.03. The all-sample mean LPIPS guard for NEW_LAO8 is ≤ 0.015.

| n_clean_detected | left_success | right_success | added_success | lost_success | difference | ci95_lower | ci95_upper | conservative_ci95_lower | conservative_ci95_upper | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 41 | 40 | 25 | 15 | 0 | 0.3659 | 0.2195 | 0.5122 | 0.1031 | 0.5524 | 0.0001 |

Clear improvement supported: **True**. NEW_LAO8 mean LPIPS: **0.007919**; mean-LPIPS guard passed: **True**.

## Interpretation and limits

- Second-round validation uses 50 new SD1.5 seeds, selected without detector outcomes after the development candidate and code were frozen. All 100 canonical file/pixel hashes, planned source choices, old exclusions including the observed first validation, and freeze timestamps were checked.
- Primary outcome is NEW_LAO8 minus frozen OLD_LAO8 on E/none: conditional success within the common clean-detected set, requiring saved uint8 L-infinity <= 8, PSNR >= 30 and LPIPS <= 0.03. All samples remain in the denominator; invalid or quality-failed outputs count as failures.
- A clear improvement requires a paired bootstrap 95% lower bound > 0, two-sided exact McNemar p < 0.05, NEW_LAO8 all-sample mean LPIPS <= 0.015, and passing budget/source/reset integrity. Failure to meet this rule is not proof of equivalence or noninferiority.
- The stricter quality screen is a new, predeclared research question, motivated by the previous round. The original PSNR >= 30 / LPIPS <= 0.15 screen and classifier-only ASR are also reported; previous results are not overwritten.
- OLD_LAO8 optimises its unprojected residual decode and is projected only when exported. NEW_LAO8 optimises exact saved-budget values through a straight-through derivative and selects outputs using E and LPIPS. This compares complete methods, not a single isolated mechanism or equal computation budgets.
- PGD10 retains its final iterate. PGD10_MATCHED uses the predeclared E-margin/quality selection rule on its fixed ten-step trajectory. These are secondary controls: PGD losing points under a stricter LPIPS screen does not prove that latent attacks are generally stronger.
- NEW_RECON8 and OLD_RECON8 are independently executed zero-step controls and must match each other byte for byte. Each attack reset must also match its respective baseline. The method is experimental residual rendering, not a guarantee that output lies on an ordinary decoder manifold.
- PSNR and LPIPS are numerical quality screens, not human confirmation of meaning or usability. Quality is measured on saved PNG before postprocessing; the 8/255 budget also applies before JPEG/resize.
- R/D/S and transformed-image cells are descriptive and must not select methods or outputs. Always interpret them alongside real-image FPR; changing preprocessing can invalidate a detector operating point. Detector cells share sources and must not be pooled as independent observations.
- New seeds from the same ten prompt categories do not establish generalisation to new prompt families, generators or detectors. Only the original ten development images selected the candidate; the previous 50 are already observed and explicitly excluded.
- Secondary intervals and tests are not adjusted for multiple comparisons. Conservative paired-risk-difference intervals remain available to show finite-sample uncertainty, including when all paired outcomes agree; no noninferiority claim is made.
- A stronger standalone latent contribution does not establish dominance inside the original hybrid PGD-plus-LAO pipeline. Timing follows recorded work, including diagnostics and any shared trajectory; medians accompany means.
- This reporter only reads saved PNGs, records and CSVs and writes a new report directory. It executes no detector, generator, training or attack.
- The original frozen reporter stopped at image0041 because single-image and batch16 MPS E margins differ by 1.017749309539795e-5. A saved-PNG re-score exactly reproduced both sets of all50 scores; all labels, margin-buffer decisions and LPIPS values agree. This report uses an archived margin-only absolute consistency tolerance of 2e-5 instead of 1e-5. No images, attack settings, quality thresholds, predictions or statistics were changed. Original failure and audit are retained under report_recovery/.

## Original-PNG results: every source retained

| run | detector | n | n_clean_detected | n_valid_success | cond_asr | n_quality_qualified_success | quality_qualified_asr | legacy_quality_asr | uncond_escape | real_fpr | lpips | lpips_p95 | lpips_max | psnr_median | seconds | median_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_RECON8 | E | 50 | 41 | 1 | 0.0244 | 1 | 0.0244 | 0.0244 | 0.1800 | 0.0000 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9322 | 2.8900 |
| OLD_RECON8 | R | 50 | 41 | 2 | 0.0488 | 2 | 0.0488 | 0.0488 | 0.2200 | 0.0000 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9322 | 2.8900 |
| OLD_RECON8 | D | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0200 | 0.0200 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9322 | 2.8900 |
| OLD_RECON8 | S | 50 | 48 | 7 | 0.1458 | 7 | 0.1458 | 0.1458 | 0.1800 | 0.0000 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9322 | 2.8900 |
| OLD_LAO8 | E | 50 | 41 | 25 | 0.6098 | 25 | 0.6098 | 0.6098 | 0.6800 | 0.0000 | 0.0073 | 0.0129 | 0.0216 | 36.8827 | 10.7882 | 10.6650 |
| OLD_LAO8 | R | 50 | 41 | 7 | 0.1707 | 7 | 0.1707 | 0.1707 | 0.3200 | 0.0000 | 0.0073 | 0.0129 | 0.0216 | 36.8827 | 10.7882 | 10.6650 |
| OLD_LAO8 | D | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0200 | 0.0200 | 0.0073 | 0.0129 | 0.0216 | 36.8827 | 10.7882 | 10.6650 |
| OLD_LAO8 | S | 50 | 48 | 9 | 0.1875 | 9 | 0.1875 | 0.1875 | 0.2200 | 0.0000 | 0.0073 | 0.0129 | 0.0216 | 36.8827 | 10.7882 | 10.6650 |
| NEW_RECON8 | E | 50 | 41 | 1 | 0.0244 | 1 | 0.0244 | 0.0244 | 0.1800 | 0.0000 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9993 | 2.9393 |
| NEW_RECON8 | R | 50 | 41 | 2 | 0.0488 | 2 | 0.0488 | 0.0488 | 0.2200 | 0.0000 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9993 | 2.9393 |
| NEW_RECON8 | D | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0200 | 0.0200 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9993 | 2.9393 |
| NEW_RECON8 | S | 50 | 48 | 7 | 0.1458 | 7 | 0.1458 | 0.1458 | 0.1800 | 0.0000 | 0.0039 | 0.0065 | 0.0120 | 40.0099 | 2.9993 | 2.9393 |
| NEW_LAO8 | E | 50 | 41 | 40 | 0.9756 | 40 | 0.9756 | 0.9756 | 0.9800 | 0.0000 | 0.0079 | 0.0209 | 0.0285 | 37.7551 | 9.4139 | 7.4830 |
| NEW_LAO8 | R | 50 | 41 | 3 | 0.0732 | 3 | 0.0732 | 0.0732 | 0.2400 | 0.0000 | 0.0079 | 0.0209 | 0.0285 | 37.7551 | 9.4139 | 7.4830 |
| NEW_LAO8 | D | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0200 | 0.0200 | 0.0079 | 0.0209 | 0.0285 | 37.7551 | 9.4139 | 7.4830 |
| NEW_LAO8 | S | 50 | 48 | 8 | 0.1667 | 8 | 0.1667 | 0.1667 | 0.2000 | 0.0000 | 0.0079 | 0.0209 | 0.0285 | 37.7551 | 9.4139 | 7.4830 |
| PGD10 | E | 50 | 41 | 41 | 1.0000 | 0 | 0.0000 | 0.9512 | 1.0000 | 0.0000 | 0.0721 | 0.1568 | 0.2437 | 36.0467 | 0.9263 | 0.8894 |
| PGD10 | R | 50 | 41 | 23 | 0.5610 | 0 | 0.0000 | 0.4878 | 0.6400 | 0.0000 | 0.0721 | 0.1568 | 0.2437 | 36.0467 | 0.9263 | 0.8894 |
| PGD10 | D | 50 | 49 | 7 | 0.1429 | 0 | 0.0000 | 0.1020 | 0.1600 | 0.0200 | 0.0721 | 0.1568 | 0.2437 | 36.0467 | 0.9263 | 0.8894 |
| PGD10 | S | 50 | 48 | 46 | 0.9583 | 0 | 0.0000 | 0.8750 | 0.9600 | 0.0000 | 0.0721 | 0.1568 | 0.2437 | 36.0467 | 0.9263 | 0.8894 |
| PGD10_MATCHED | E | 50 | 41 | 19 | 0.4634 | 19 | 0.4634 | 0.4634 | 0.5600 | 0.0000 | 0.0005 | 0.0019 | 0.0022 | inf | 0.9263 | 0.8894 |
| PGD10_MATCHED | R | 50 | 41 | 1 | 0.0244 | 1 | 0.0244 | 0.0244 | 0.2000 | 0.0000 | 0.0005 | 0.0019 | 0.0022 | inf | 0.9263 | 0.8894 |
| PGD10_MATCHED | D | 50 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0200 | 0.0200 | 0.0005 | 0.0019 | 0.0022 | inf | 0.9263 | 0.8894 |
| PGD10_MATCHED | S | 50 | 48 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0400 | 0.0000 | 0.0005 | 0.0019 | 0.0022 | inf | 0.9263 | 0.8894 |

## Secondary postprocessing: inspect real-image FPR

| run | detector | condition | n_clean_detected | n_valid_success | cond_asr | n_quality_qualified_success | quality_qualified_asr | legacy_quality_asr | real_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_RECON8 | E | jpeg75 | 38 | 4 | 0.1053 | 4 | 0.1053 | 0.1053 | 0.0800 |
| OLD_RECON8 | E | resize05 | 49 | 1 | 0.0204 | 1 | 0.0204 | 0.0204 | 0.8800 |
| OLD_RECON8 | R | jpeg75 | 44 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| OLD_RECON8 | R | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.9400 |
| OLD_RECON8 | D | jpeg75 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| OLD_RECON8 | D | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 1.0000 |
| OLD_RECON8 | S | jpeg75 | 50 | 1 | 0.0200 | 1 | 0.0200 | 0.0200 | 0.4600 |
| OLD_RECON8 | S | resize05 | 12 | 6 | 0.5000 | 6 | 0.5000 | 0.5000 | 0.1200 |
| OLD_LAO8 | E | jpeg75 | 38 | 10 | 0.2632 | 10 | 0.2632 | 0.2632 | 0.0800 |
| OLD_LAO8 | E | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.8800 |
| OLD_LAO8 | R | jpeg75 | 44 | 1 | 0.0227 | 1 | 0.0227 | 0.0227 | 0.2200 |
| OLD_LAO8 | R | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.9400 |
| OLD_LAO8 | D | jpeg75 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| OLD_LAO8 | D | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 1.0000 |
| OLD_LAO8 | S | jpeg75 | 50 | 3 | 0.0600 | 3 | 0.0600 | 0.0600 | 0.4600 |
| OLD_LAO8 | S | resize05 | 12 | 7 | 0.5833 | 7 | 0.5833 | 0.5833 | 0.1200 |
| NEW_RECON8 | E | jpeg75 | 38 | 4 | 0.1053 | 4 | 0.1053 | 0.1053 | 0.0800 |
| NEW_RECON8 | E | resize05 | 49 | 1 | 0.0204 | 1 | 0.0204 | 0.0204 | 0.8800 |
| NEW_RECON8 | R | jpeg75 | 44 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| NEW_RECON8 | R | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.9400 |
| NEW_RECON8 | D | jpeg75 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| NEW_RECON8 | D | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 1.0000 |
| NEW_RECON8 | S | jpeg75 | 50 | 1 | 0.0200 | 1 | 0.0200 | 0.0200 | 0.4600 |
| NEW_RECON8 | S | resize05 | 12 | 6 | 0.5000 | 6 | 0.5000 | 0.5000 | 0.1200 |
| NEW_LAO8 | E | jpeg75 | 38 | 5 | 0.1316 | 5 | 0.1316 | 0.1316 | 0.0800 |
| NEW_LAO8 | E | resize05 | 49 | 1 | 0.0204 | 1 | 0.0204 | 0.0204 | 0.8800 |
| NEW_LAO8 | R | jpeg75 | 44 | 1 | 0.0227 | 1 | 0.0227 | 0.0227 | 0.2200 |
| NEW_LAO8 | R | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.9400 |
| NEW_LAO8 | D | jpeg75 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| NEW_LAO8 | D | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 1.0000 |
| NEW_LAO8 | S | jpeg75 | 50 | 1 | 0.0200 | 1 | 0.0200 | 0.0200 | 0.4600 |
| NEW_LAO8 | S | resize05 | 12 | 4 | 0.3333 | 4 | 0.3333 | 0.3333 | 0.1200 |
| PGD10 | E | jpeg75 | 38 | 20 | 0.5263 | 0 | 0.0000 | 0.5000 | 0.0800 |
| PGD10 | E | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.8800 |
| PGD10 | R | jpeg75 | 44 | 1 | 0.0227 | 0 | 0.0000 | 0.0227 | 0.2200 |
| PGD10 | R | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.9400 |
| PGD10 | D | jpeg75 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| PGD10 | D | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 1.0000 |
| PGD10 | S | jpeg75 | 50 | 3 | 0.0600 | 0 | 0.0000 | 0.0600 | 0.4600 |
| PGD10 | S | resize05 | 12 | 6 | 0.5000 | 0 | 0.0000 | 0.4167 | 0.1200 |
| PGD10_MATCHED | E | jpeg75 | 38 | 5 | 0.1316 | 5 | 0.1316 | 0.1316 | 0.0800 |
| PGD10_MATCHED | E | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.8800 |
| PGD10_MATCHED | R | jpeg75 | 44 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| PGD10_MATCHED | R | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.9400 |
| PGD10_MATCHED | D | jpeg75 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.2200 |
| PGD10_MATCHED | D | resize05 | 49 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 1.0000 |
| PGD10_MATCHED | S | jpeg75 | 50 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 0.4600 |
| PGD10_MATCHED | S | resize05 | 12 | 3 | 0.2500 | 3 | 0.2500 | 0.2500 | 0.1200 |

## Paired low-distortion comparisons

| comparator | detector | condition | n_clean_detected | added_success | lost_success | difference | ci95_lower | ci95_upper | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_LAO8 | E | none | 41 | 15 | 0 | 0.3659 | 0.2195 | 0.5122 | 0.0001 |
| OLD_LAO8 | E | jpeg75 | 38 | 0 | 5 | -0.1316 | n/a | n/a | n/a |
| OLD_LAO8 | E | resize05 | 49 | 1 | 0 | 0.0204 | n/a | n/a | n/a |
| OLD_LAO8 | R | none | 41 | 0 | 4 | -0.0976 | n/a | n/a | n/a |
| OLD_LAO8 | R | jpeg75 | 44 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_LAO8 | R | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_LAO8 | D | none | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_LAO8 | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_LAO8 | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_LAO8 | S | none | 48 | 1 | 2 | -0.0208 | n/a | n/a | n/a |
| OLD_LAO8 | S | jpeg75 | 50 | 0 | 2 | -0.0400 | n/a | n/a | n/a |
| OLD_LAO8 | S | resize05 | 12 | 1 | 4 | -0.2500 | n/a | n/a | n/a |
| PGD10 | E | none | 41 | 40 | 0 | 0.9756 | 0.9268 | 1.0000 | 0.0000 |
| PGD10 | E | jpeg75 | 38 | 5 | 0 | 0.1316 | n/a | n/a | n/a |
| PGD10 | E | resize05 | 49 | 1 | 0 | 0.0204 | n/a | n/a | n/a |
| PGD10 | R | none | 41 | 3 | 0 | 0.0732 | n/a | n/a | n/a |
| PGD10 | R | jpeg75 | 44 | 1 | 0 | 0.0227 | n/a | n/a | n/a |
| PGD10 | R | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10 | D | none | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10 | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10 | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10 | S | none | 48 | 8 | 0 | 0.1667 | n/a | n/a | n/a |
| PGD10 | S | jpeg75 | 50 | 1 | 0 | 0.0200 | n/a | n/a | n/a |
| PGD10 | S | resize05 | 12 | 4 | 0 | 0.3333 | n/a | n/a | n/a |
| PGD10_MATCHED | E | none | 41 | 21 | 0 | 0.5122 | 0.3659 | 0.6585 | 0.0000 |
| PGD10_MATCHED | E | jpeg75 | 38 | 2 | 2 | 0.0000 | n/a | n/a | n/a |
| PGD10_MATCHED | E | resize05 | 49 | 1 | 0 | 0.0204 | n/a | n/a | n/a |
| PGD10_MATCHED | R | none | 41 | 2 | 0 | 0.0488 | n/a | n/a | n/a |
| PGD10_MATCHED | R | jpeg75 | 44 | 1 | 0 | 0.0227 | n/a | n/a | n/a |
| PGD10_MATCHED | R | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10_MATCHED | D | none | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10_MATCHED | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10_MATCHED | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| PGD10_MATCHED | S | none | 48 | 8 | 0 | 0.1667 | n/a | n/a | n/a |
| PGD10_MATCHED | S | jpeg75 | 50 | 1 | 0 | 0.0200 | n/a | n/a | n/a |
| PGD10_MATCHED | S | resize05 | 12 | 3 | 2 | 0.0833 | n/a | n/a | n/a |
| NEW_RECON8 | E | none | 41 | 39 | 0 | 0.9512 | 0.8780 | 1.0000 | 0.0000 |
| NEW_RECON8 | E | jpeg75 | 38 | 2 | 1 | 0.0263 | n/a | n/a | n/a |
| NEW_RECON8 | E | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| NEW_RECON8 | R | none | 41 | 2 | 1 | 0.0244 | n/a | n/a | n/a |
| NEW_RECON8 | R | jpeg75 | 44 | 1 | 0 | 0.0227 | n/a | n/a | n/a |
| NEW_RECON8 | R | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| NEW_RECON8 | D | none | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| NEW_RECON8 | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| NEW_RECON8 | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| NEW_RECON8 | S | none | 48 | 2 | 1 | 0.0208 | n/a | n/a | n/a |
| NEW_RECON8 | S | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| NEW_RECON8 | S | resize05 | 12 | 1 | 3 | -0.1667 | n/a | n/a | n/a |
| OLD_RECON8 | E | none | 41 | 39 | 0 | 0.9512 | 0.8780 | 1.0000 | 0.0000 |
| OLD_RECON8 | E | jpeg75 | 38 | 2 | 1 | 0.0263 | n/a | n/a | n/a |
| OLD_RECON8 | E | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_RECON8 | R | none | 41 | 2 | 1 | 0.0244 | n/a | n/a | n/a |
| OLD_RECON8 | R | jpeg75 | 44 | 1 | 0 | 0.0227 | n/a | n/a | n/a |
| OLD_RECON8 | R | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_RECON8 | D | none | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_RECON8 | D | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_RECON8 | D | resize05 | 49 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_RECON8 | S | none | 48 | 2 | 1 | 0.0208 | n/a | n/a | n/a |
| OLD_RECON8 | S | jpeg75 | 50 | 0 | 0 | 0.0000 | n/a | n/a | n/a |
| OLD_RECON8 | S | resize05 | 12 | 1 | 3 | -0.1667 | n/a | n/a | n/a |

## Integrity and diagnostics

Reset checks: **200/200** identical to independently executed zero-step controls. Budget violations: **0**. All canonical inputs and origins were checked; old-method exports were verified pixel for pixel.

| run | image_id | selected_step | actual_steps | finite_gradient_steps | nonzero_gradient_steps | nonzero_update_steps | first_gradient_l2 | last_gradient_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD_LAO8 | 0000 | 10 | 10 | 10 | 10 | 10 | 2.9427 | 2.7232 |
| OLD_LAO8 | 0001 | 10 | 10 | 10 | 10 | 10 | 3.5678 | 0.0031 |
| OLD_LAO8 | 0002 | 10 | 10 | 10 | 10 | 10 | 2.6636 | 2.1947 |
| OLD_LAO8 | 0003 | 10 | 10 | 10 | 10 | 10 | 1.7841 | 0.0029 |
| OLD_LAO8 | 0004 | 10 | 10 | 10 | 10 | 10 | 2.8049 | 0.0046 |
| OLD_LAO8 | 0005 | 10 | 10 | 10 | 10 | 10 | 1.9967 | 2.3566 |
| OLD_LAO8 | 0006 | 10 | 10 | 10 | 10 | 10 | 0.0012 | 0.0003 |
| OLD_LAO8 | 0007 | 10 | 10 | 10 | 10 | 10 | 4.8654 | 0.0028 |
| OLD_LAO8 | 0008 | 10 | 10 | 10 | 10 | 10 | 1.8431 | 2.3630 |
| OLD_LAO8 | 0009 | 10 | 10 | 10 | 10 | 10 | 2.1127 | 1.4733 |
| OLD_LAO8 | 0010 | 10 | 10 | 10 | 10 | 10 | 2.8671 | 1.8724 |
| OLD_LAO8 | 0011 | 10 | 10 | 10 | 10 | 10 | 4.1416 | 2.7501 |
| OLD_LAO8 | 0012 | 10 | 10 | 10 | 10 | 10 | 2.4582 | 1.7357 |
| OLD_LAO8 | 0013 | 10 | 10 | 10 | 10 | 10 | 1.7162 | 0.0024 |
| OLD_LAO8 | 0014 | 10 | 10 | 10 | 10 | 10 | 2.1282 | 1.8831 |
| OLD_LAO8 | 0015 | 10 | 10 | 10 | 10 | 10 | 2.2851 | 1.9443 |
| OLD_LAO8 | 0016 | 10 | 10 | 10 | 10 | 10 | 2.9158 | 0.0021 |
| OLD_LAO8 | 0017 | 10 | 10 | 10 | 10 | 10 | 2.6601 | 1.9858 |
| OLD_LAO8 | 0018 | 10 | 10 | 10 | 10 | 10 | 3.1563 | 0.0037 |
| OLD_LAO8 | 0019 | 10 | 10 | 10 | 10 | 10 | 2.6325 | 2.6663 |
| OLD_LAO8 | 0020 | 10 | 10 | 10 | 10 | 10 | 3.3123 | 3.0639 |
| OLD_LAO8 | 0021 | 10 | 10 | 10 | 10 | 10 | 3.4846 | 0.0024 |
| OLD_LAO8 | 0022 | 10 | 10 | 10 | 10 | 10 | 2.2738 | 2.1492 |
| OLD_LAO8 | 0023 | 10 | 10 | 10 | 10 | 10 | 3.8777 | 0.0019 |
| OLD_LAO8 | 0024 | 10 | 10 | 10 | 10 | 10 | 2.0362 | 1.6383 |
| OLD_LAO8 | 0025 | 10 | 10 | 10 | 10 | 10 | 5.5813 | 1.8962 |
| OLD_LAO8 | 0026 | 10 | 10 | 10 | 10 | 10 | 3.0974 | 2.1011 |
| OLD_LAO8 | 0027 | 10 | 10 | 10 | 10 | 10 | 2.5040 | 2.8184 |
| OLD_LAO8 | 0028 | 10 | 10 | 10 | 10 | 10 | 2.8679 | 2.4596 |
| OLD_LAO8 | 0029 | 10 | 10 | 10 | 10 | 10 | 1.0932 | 0.8053 |
| OLD_LAO8 | 0030 | 10 | 10 | 10 | 10 | 10 | 2.0988 | 0.0028 |
| OLD_LAO8 | 0031 | 10 | 10 | 10 | 10 | 10 | 5.1729 | 3.0475 |
| OLD_LAO8 | 0032 | 10 | 10 | 10 | 10 | 10 | 2.1345 | 2.1718 |
| OLD_LAO8 | 0033 | 10 | 10 | 10 | 10 | 10 | 0.0017 | 0.0004 |
| OLD_LAO8 | 0034 | 10 | 10 | 10 | 10 | 10 | 1.8962 | 0.0022 |
| OLD_LAO8 | 0035 | 10 | 10 | 10 | 10 | 10 | 2.0823 | 3.0237 |
| OLD_LAO8 | 0036 | 10 | 10 | 10 | 10 | 10 | 3.4537 | 0.0026 |
| OLD_LAO8 | 0037 | 10 | 10 | 10 | 10 | 10 | 3.0632 | 0.0136 |
| OLD_LAO8 | 0038 | 10 | 10 | 10 | 10 | 10 | 3.3481 | 0.0024 |
| OLD_LAO8 | 0039 | 10 | 10 | 10 | 10 | 10 | 18.0395 | 0.0082 |
| OLD_LAO8 | 0040 | 10 | 10 | 10 | 10 | 10 | 1.9557 | 2.2470 |
| OLD_LAO8 | 0041 | 10 | 10 | 10 | 10 | 10 | 3.2867 | 0.0029 |
| OLD_LAO8 | 0042 | 10 | 10 | 10 | 10 | 10 | 3.2796 | 2.3616 |
| OLD_LAO8 | 0043 | 10 | 10 | 10 | 10 | 10 | 2.6943 | 0.0030 |
| OLD_LAO8 | 0044 | 10 | 10 | 10 | 10 | 10 | 1.7471 | 0.0033 |
| OLD_LAO8 | 0045 | 10 | 10 | 10 | 10 | 10 | 2.1029 | 1.9618 |
| OLD_LAO8 | 0046 | 10 | 10 | 10 | 10 | 10 | 2.9710 | 0.0031 |
| OLD_LAO8 | 0047 | 10 | 10 | 10 | 10 | 10 | 2.5584 | 0.0029 |
| OLD_LAO8 | 0048 | 10 | 10 | 10 | 10 | 10 | 1.7547 | 2.0948 |
| OLD_LAO8 | 0049 | 10 | 10 | 10 | 10 | 10 | 2.0923 | 0.0083 |
| NEW_LAO8 | 0000 | 6 | 6 | 6 | 6 | 6 | 2.9410 | 2.4824 |
| NEW_LAO8 | 0001 | 2 | 2 | 2 | 2 | 2 | 3.5679 | 3.5619 |
| NEW_LAO8 | 0002 | 11 | 11 | 11 | 11 | 11 | 2.6533 | 2.6004 |
| NEW_LAO8 | 0003 | 3 | 3 | 3 | 3 | 3 | 1.7967 | 1.8864 |
| NEW_LAO8 | 0004 | 3 | 3 | 3 | 3 | 3 | 2.8164 | 2.6528 |
| NEW_LAO8 | 0005 | 4 | 4 | 4 | 4 | 4 | 2.0009 | 5.7744 |
| NEW_LAO8 | 0006 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0007 | 1 | 1 | 1 | 1 | 1 | 4.7454 | 4.7454 |
| NEW_LAO8 | 0008 | 6 | 6 | 6 | 6 | 6 | 1.8769 | 2.2755 |
| NEW_LAO8 | 0009 | 6 | 6 | 6 | 6 | 6 | 2.1176 | 1.5840 |
| NEW_LAO8 | 0010 | 5 | 5 | 5 | 5 | 5 | 2.8670 | 2.0712 |
| NEW_LAO8 | 0011 | 5 | 5 | 5 | 5 | 5 | 4.1603 | 2.9475 |
| NEW_LAO8 | 0012 | 17 | 17 | 17 | 17 | 17 | 2.4392 | 2.4305 |
| NEW_LAO8 | 0013 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0014 | 5 | 5 | 5 | 5 | 5 | 2.1306 | 1.8859 |
| NEW_LAO8 | 0015 | 10 | 10 | 10 | 10 | 10 | 2.1725 | 2.2573 |
| NEW_LAO8 | 0016 | 1 | 1 | 1 | 1 | 1 | 2.9852 | 2.9852 |
| NEW_LAO8 | 0017 | 15 | 15 | 15 | 15 | 15 | 2.8950 | 2.4509 |
| NEW_LAO8 | 0018 | 2 | 2 | 2 | 2 | 2 | 3.1904 | 2.9738 |
| NEW_LAO8 | 0019 | 5 | 5 | 5 | 5 | 5 | 2.6077 | 3.7433 |
| NEW_LAO8 | 0020 | 11 | 40 | 40 | 40 | 40 | 3.2905 | 0.0376 |
| NEW_LAO8 | 0021 | 2 | 2 | 2 | 2 | 2 | 3.4726 | 3.2076 |
| NEW_LAO8 | 0022 | 13 | 13 | 13 | 13 | 13 | 2.2836 | 2.0406 |
| NEW_LAO8 | 0023 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0024 | 7 | 7 | 7 | 7 | 7 | 2.0678 | 1.6330 |
| NEW_LAO8 | 0025 | 7 | 7 | 7 | 7 | 7 | 4.7941 | 2.1860 |
| NEW_LAO8 | 0026 | 3 | 3 | 3 | 3 | 3 | 3.0194 | 3.3432 |
| NEW_LAO8 | 0027 | 8 | 8 | 8 | 8 | 8 | 2.5208 | 3.2875 |
| NEW_LAO8 | 0028 | 4 | 4 | 4 | 4 | 4 | 2.8652 | 2.6604 |
| NEW_LAO8 | 0029 | 2 | 2 | 2 | 2 | 2 | 1.1092 | 0.8293 |
| NEW_LAO8 | 0030 | 3 | 3 | 3 | 3 | 3 | 2.1094 | 2.1247 |
| NEW_LAO8 | 0031 | 6 | 6 | 6 | 6 | 6 | 5.1585 | 2.9905 |
| NEW_LAO8 | 0032 | 5 | 5 | 5 | 5 | 5 | 2.1078 | 2.1492 |
| NEW_LAO8 | 0033 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0034 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0035 | 7 | 7 | 7 | 7 | 7 | 1.9878 | 2.7389 |
| NEW_LAO8 | 0036 | 3 | 3 | 3 | 3 | 3 | 3.4731 | 3.6861 |
| NEW_LAO8 | 0037 | 2 | 2 | 2 | 2 | 2 | 3.0545 | 4.2862 |
| NEW_LAO8 | 0038 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0039 | 2 | 2 | 2 | 2 | 2 | 17.6993 | 2.6370 |
| NEW_LAO8 | 0040 | 9 | 9 | 9 | 9 | 9 | 1.9584 | 3.0797 |
| NEW_LAO8 | 0041 | 1 | 1 | 1 | 1 | 1 | 3.2878 | 3.2878 |
| NEW_LAO8 | 0042 | 5 | 5 | 5 | 5 | 5 | 3.3662 | 2.6368 |
| NEW_LAO8 | 0043 | 2 | 2 | 2 | 2 | 2 | 2.6715 | 2.5737 |
| NEW_LAO8 | 0044 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0045 | 11 | 40 | 40 | 40 | 40 | 2.0397 | 0.0258 |
| NEW_LAO8 | 0046 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| NEW_LAO8 | 0047 | 3 | 3 | 3 | 3 | 3 | 2.5799 | 2.5759 |
| NEW_LAO8 | 0048 | 9 | 9 | 9 | 9 | 9 | 1.7729 | 2.0744 |
| NEW_LAO8 | 0049 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
