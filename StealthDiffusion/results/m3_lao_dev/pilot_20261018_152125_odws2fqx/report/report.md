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
- EXPERIMENTAL RESIDUAL RENDERING: anchored runs form x(z) = D_1(z0) + [D_g(z) - D_g(z0)] before the normal image conversion, clipping, and saved-file quantisation; z0 is the initial denoised latent and g is the fixed skip gate.
- An anchored output is a reconstruction plus a latent-controlled image residual. It is not the ordinary D_g(z) decoder output and is not guaranteed to lie in that decoder's image range.
- At z=z0 the residual cancels, so the matched zero-step anchor baseline should recover D_1(z0). Reset PNG hashes must still match the paired baseline; the formula alone is not evidence of a valid saved-file implementation.
- Success can be attributed to latent optimisation within this anchored rendering when the paired controls pass. It does not establish success for ordinary decoder-only LAO or repair the failed quality of raw gated decoding.
- Clipping and quantisation can alter the residual. Quality and perturbation measurements must describe the final saved image relative to the original, not only the latent change or the unclipped residual.

## Saved-image outcomes (all samples retained)

| run | rendering_mode | detector | condition | n | n_clean_detected | n_success | cond_asr | real_fpr | n_quality_qualified | psnr | lpips | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | experimental_residual_anchor | D | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 39.3482 | 0.0046 | 3.1000 |
| G0 | experimental_residual_anchor | E | none | 10 | 8 | 0 | 0.0000 | 0.0000 | 10 | 39.3482 | 0.0046 | 3.1000 |
| G0 | experimental_residual_anchor | R | none | 10 | 9 | 1 | 0.1111 | 0.0000 | 10 | 39.3482 | 0.0046 | 3.1000 |
| G0 | experimental_residual_anchor | S | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 39.3482 | 0.0046 | 3.1000 |
| G1 | experimental_residual_anchor | D | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 35.4965 | 0.0112 | 11.1030 |
| G1 | experimental_residual_anchor | E | none | 10 | 8 | 5 | 0.6250 | 0.0000 | 10 | 35.4965 | 0.0112 | 11.1030 |
| G1 | experimental_residual_anchor | R | none | 10 | 9 | 2 | 0.2222 | 0.0000 | 10 | 35.4965 | 0.0112 | 11.1030 |
| G1 | experimental_residual_anchor | S | none | 10 | 10 | 0 | 0.0000 | 0.0000 | 10 | 35.4965 | 0.0112 | 11.1030 |

## Paired LAO contribution beyond reconstruction

Added/lost flips use the clean-detected population. The stricter subset also requires the zero-step baseline to remain detected. Quality-qualified new flips keep that full subset as the denominator.

| baseline | run | rendering_mode | detector | condition | n_clean_detected | n_clean_and_baseline_detected | added_success | lost_success | net_added_success | quality_qualified_new_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | G1 | experimental_residual_anchor | D | none | 10 | 10 | 0 | 0 | 0 | 0 |
| G0 | G1 | experimental_residual_anchor | E | none | 8 | 8 | 5 | 0 | 5 | 5 |
| G0 | G1 | experimental_residual_anchor | R | none | 9 | 8 | 1 | 0 | 1 | 1 |
| G0 | G1 | experimental_residual_anchor | S | none | 10 | 10 | 0 | 0 | 0 | 0 |

## Reset-latent counterfactual integrity

When available, each optimised run's reset-latent PNG must match its paired R0/G0 final PNG byte for byte. A mismatch stops reporting and must be investigated before claiming attribution.

| baseline | run | image_id | available | identical |
| --- | --- | --- | --- | --- |
| G0 | G1 | 0000 | True | True |
| G0 | G1 | 0001 | True | True |
| G0 | G1 | 0002 | True | True |
| G0 | G1 | 0003 | True | True |
| G0 | G1 | 0004 | True | True |
| G0 | G1 | 0005 | True | True |
| G0 | G1 | 0006 | True | True |
| G0 | G1 | 0007 | True | True |
| G0 | G1 | 0008 | True | True |
| G0 | G1 | 0009 | True | True |

## Float-to-PNG retention (no post-processing)

| run | detector | n_clean_detected | float_predictions_available | float_success | png_success | retained_float_success | retention_rate | float_success_lost_on_png | png_only_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | D | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| G0 | E | 8 | True | 0 | 0 | 0 | n/a | 0 | 0 |
| G0 | R | 9 | False | n/a | 1 | n/a | n/a | n/a | n/a |
| G0 | S | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| G1 | D | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |
| G1 | E | 8 | True | 5 | 5 | 5 | 1.0000 | 0 | 0 |
| G1 | R | 9 | False | n/a | 2 | n/a | n/a | n/a | n/a |
| G1 | S | 10 | False | n/a | 0 | n/a | n/a | n/a | n/a |

## Gradient and latent diagnostics

First/last-step scalar values and available per-image diagnostics are preserved in diagnostics.csv and report.json. Missing diagnostics are explicitly marked unavailable; missing gradients do not establish that the actual gradient was zero.

| run | image_id | history_steps | gradient_diagnostics_available |
| --- | --- | --- | --- |
| G0 | 0000 | 0 | False |
| G0 | 0001 | 0 | False |
| G0 | 0002 | 0 | False |
| G0 | 0003 | 0 | False |
| G0 | 0004 | 0 | False |
| G0 | 0005 | 0 | False |
| G0 | 0006 | 0 | False |
| G0 | 0007 | 0 | False |
| G0 | 0008 | 0 | False |
| G0 | 0009 | 0 | False |
| G1 | 0000 | 10 | True |
| G1 | 0001 | 10 | True |
| G1 | 0002 | 10 | True |
| G1 | 0003 | 10 | True |
| G1 | 0004 | 10 | True |
| G1 | 0005 | 10 | True |
| G1 | 0006 | 10 | True |
| G1 | 0007 | 10 | True |
| G1 | 0008 | 10 | True |
| G1 | 0009 | 10 | True |
