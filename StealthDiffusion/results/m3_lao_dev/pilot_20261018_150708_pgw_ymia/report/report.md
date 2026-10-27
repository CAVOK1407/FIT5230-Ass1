# LAO development-set diagnosis

Source images: **10**. No model was run by this reporter.

- Exploratory architecture diagnosis, not independent final-test evidence. Development images and previously examined test images must not be presented as a fresh confirmatory test.
- Class convention: 0 = AI/detected, 1 = real/escape. Every reported attack outcome comes from saved-file evaluation.
- All samples remain in the reported denominators. Quality screening never removes failures or damaged images from ASR.
- Fixed exploratory quality screen: PSNR >= 30 dB and LPIPS <= 0.15 on the unprocessed saved image; this is not semantic or human review.
- For transformed conditions the quality screen still describes the unprocessed PNG, not the transformed image.
- LAO attribution uses paired comparisons against the zero-step reconstruction at the same skip gate and saved decoder.
- Float/PNG retention is reported only for condition=none and uses the same saved-clean eligibility mask.
- Logged first/last-step scores precede their respective optimiser updates; final_after_all_updates records the actual final point when available.
- Diagnostic-enabled timings include additional scoring and reset-latent checks and are not production-runtime benchmarks.
- Detector outcomes on the same source image are correlated; do not treat images times detectors as independent samples.
- The reporter reads files only. It performs no model inference, optimisation, training, or source-file edits.

## Saved-image outcomes (all samples retained)

| run | detector | condition | n | n_clean_detected | n_success | cond_asr | real_fpr | n_quality_qualified | psnr | lpips | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | D | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 39.3482 | 0.0046 | 2.6800 |
| R0 | E | none | 10 | 8 | 0 | 0.0000 | 0.0000 | 10 | 39.3482 | 0.0046 | 2.6800 |
| R0 | R | none | 10 | 9 | 1 | 0.1111 | 0.0000 | 10 | 39.3482 | 0.0046 | 2.6800 |
| R0 | S | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 39.3482 | 0.0046 | 2.6800 |
| R1 | D | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 39.1240 | 0.0048 | 10.6770 |
| R1 | E | none | 10 | 8 | 0 | 0.0000 | 0.0000 | 10 | 39.1240 | 0.0048 | 10.6770 |
| R1 | R | none | 10 | 9 | 1 | 0.1111 | 0.0000 | 10 | 39.1240 | 0.0048 | 10.6770 |
| R1 | S | none | 10 | 10 | 2 | 0.2000 | 0.0000 | 10 | 39.1240 | 0.0048 | 10.6770 |
| R2 | D | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 35.7738 | 0.0092 | 11.2960 |
| R2 | E | none | 10 | 8 | 2 | 0.2500 | 0.0000 | 10 | 35.7738 | 0.0092 | 11.2960 |
| R2 | R | none | 10 | 9 | 1 | 0.1111 | 0.0000 | 10 | 35.7738 | 0.0092 | 11.2960 |
| R2 | S | none | 10 | 10 | 2 | 0.2000 | 0.0000 | 10 | 35.7738 | 0.0092 | 11.2960 |
| R3 | D | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 35.6442 | 0.0106 | 10.9180 |
| R3 | E | none | 10 | 8 | 2 | 0.2500 | 0.0000 | 10 | 35.6442 | 0.0106 | 10.9180 |
| R3 | R | none | 10 | 9 | 1 | 0.1111 | 0.0000 | 10 | 35.6442 | 0.0106 | 10.9180 |
| R3 | S | none | 10 | 10 | 1 | 0.1000 | 0.0000 | 10 | 35.6442 | 0.0106 | 10.9180 |

## Paired LAO contribution beyond reconstruction

Added/lost flips use the clean-detected population. The stricter subset also requires the zero-step baseline to remain detected. Quality-qualified new flips keep that full subset as the denominator.

| baseline | run | detector | condition | n_clean_detected | n_clean_and_baseline_detected | added_success | lost_success | net_added_success | quality_qualified_new_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | R1 | D | none | 10 | 10 | 0 | 0 | 0 | 0 |
| R0 | R1 | E | none | 8 | 8 | 0 | 0 | 0 | 0 |
| R0 | R1 | R | none | 9 | 8 | 0 | 0 | 0 | 0 |
| R0 | R1 | S | none | 10 | 10 | 2 | 0 | 2 | 2 |
| R0 | R2 | D | none | 10 | 10 | 0 | 0 | 0 | 0 |
| R0 | R2 | E | none | 8 | 8 | 2 | 0 | 2 | 2 |
| R0 | R2 | R | none | 9 | 8 | 0 | 0 | 0 | 0 |
| R0 | R2 | S | none | 10 | 10 | 2 | 0 | 2 | 2 |
| R0 | R3 | D | none | 10 | 10 | 0 | 0 | 0 | 0 |
| R0 | R3 | E | none | 8 | 8 | 2 | 0 | 2 | 2 |
| R0 | R3 | R | none | 9 | 8 | 0 | 0 | 0 | 0 |
| R0 | R3 | S | none | 10 | 10 | 1 | 0 | 1 | 1 |

## Reset-latent counterfactual integrity

When available, each optimised run's reset-latent PNG must match its paired R0/G0 final PNG byte for byte. A mismatch stops reporting and must be investigated before claiming attribution.

| baseline | run | image_id | available | identical |
| --- | --- | --- | --- | --- |
| R0 | R1 | 0000 | True | True |
| R0 | R1 | 0001 | True | True |
| R0 | R1 | 0002 | True | True |
| R0 | R1 | 0003 | True | True |
| R0 | R1 | 0004 | True | True |
| R0 | R1 | 0005 | True | True |
| R0 | R1 | 0006 | True | True |
| R0 | R1 | 0007 | True | True |
| R0 | R1 | 0008 | True | True |
| R0 | R1 | 0009 | True | True |
| R0 | R2 | 0000 | True | True |
| R0 | R2 | 0001 | True | True |
| R0 | R2 | 0002 | True | True |
| R0 | R2 | 0003 | True | True |
| R0 | R2 | 0004 | True | True |
| R0 | R2 | 0005 | True | True |
| R0 | R2 | 0006 | True | True |
| R0 | R2 | 0007 | True | True |
| R0 | R2 | 0008 | True | True |
| R0 | R2 | 0009 | True | True |
| R0 | R3 | 0000 | True | True |
| R0 | R3 | 0001 | True | True |
| R0 | R3 | 0002 | True | True |
| R0 | R3 | 0003 | True | True |
| R0 | R3 | 0004 | True | True |
| R0 | R3 | 0005 | True | True |
| R0 | R3 | 0006 | True | True |
| R0 | R3 | 0007 | True | True |
| R0 | R3 | 0008 | True | True |
| R0 | R3 | 0009 | True | True |

## Float-to-PNG retention (no post-processing)

| run | detector | n_clean_detected | float_predictions_available | float_success | png_success | retained_float_success | retention_rate | float_success_lost_on_png | png_only_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | D | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| R0 | E | 8 | True | 0 | 0 | 0 | n/a | 0 | 0 |
| R0 | R | 9 | False | n/a | 1 | n/a | n/a | n/a | n/a |
| R0 | S | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| R1 | D | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| R1 | E | 8 | True | 0 | 0 | 0 | n/a | 0 | 0 |
| R1 | R | 9 | False | n/a | 1 | n/a | n/a | n/a | n/a |
| R1 | S | 10 | False | n/a | 2 | n/a | n/a | n/a | n/a |
| R2 | D | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| R2 | E | 8 | True | 2 | 2 | 2 | 1.0000 | 0 | 0 |
| R2 | R | 9 | False | n/a | 1 | n/a | n/a | n/a | n/a |
| R2 | S | 10 | False | n/a | 2 | n/a | n/a | n/a | n/a |
| R3 | D | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| R3 | E | 8 | True | 2 | 2 | 2 | 1.0000 | 0 | 0 |
| R3 | R | 9 | False | n/a | 1 | n/a | n/a | n/a | n/a |
| R3 | S | 10 | False | n/a | 1 | n/a | n/a | n/a | n/a |

## Gradient and latent diagnostics

First/last-step scalar values and available per-image diagnostics are preserved in diagnostics.csv and report.json. Missing diagnostics are explicitly marked unavailable; missing gradients do not establish that the actual gradient was zero.

| run | image_id | history_steps | gradient_diagnostics_available |
| --- | --- | --- | --- |
| R0 | 0000 | 0 | False |
| R0 | 0001 | 0 | False |
| R0 | 0002 | 0 | False |
| R0 | 0003 | 0 | False |
| R0 | 0004 | 0 | False |
| R0 | 0005 | 0 | False |
| R0 | 0006 | 0 | False |
| R0 | 0007 | 0 | False |
| R0 | 0008 | 0 | False |
| R0 | 0009 | 0 | False |
| R1 | 0000 | 10 | True |
| R1 | 0001 | 10 | True |
| R1 | 0002 | 10 | True |
| R1 | 0003 | 10 | True |
| R1 | 0004 | 10 | True |
| R1 | 0005 | 10 | True |
| R1 | 0006 | 10 | True |
| R1 | 0007 | 10 | True |
| R1 | 0008 | 10 | True |
| R1 | 0009 | 10 | True |
| R2 | 0000 | 10 | True |
| R2 | 0001 | 10 | True |
| R2 | 0002 | 10 | True |
| R2 | 0003 | 10 | True |
| R2 | 0004 | 10 | True |
| R2 | 0005 | 10 | True |
| R2 | 0006 | 10 | True |
| R2 | 0007 | 10 | True |
| R2 | 0008 | 10 | True |
| R2 | 0009 | 10 | True |
| R3 | 0000 | 10 | True |
| R3 | 0001 | 10 | True |
| R3 | 0002 | 10 | True |
| R3 | 0003 | 10 | True |
| R3 | 0004 | 10 | True |
| R3 | 0005 | 10 | True |
| R3 | 0006 | 10 | True |
| R3 | 0007 | 10 | True |
| R3 | 0008 | 10 | True |
| R3 | 0009 | 10 | True |
