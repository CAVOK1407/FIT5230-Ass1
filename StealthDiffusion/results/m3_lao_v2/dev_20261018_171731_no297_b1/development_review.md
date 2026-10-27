# LAO v2: old development-set selection

Only the original 10-image development set was used. Eight images were initially detected by E. This is parameter selection evidence, not an independent generalization result.

| Method | Low-distortion success | Mean LPIPS | Actual latent updates |
|---|---:|---:|---:|
| OLD_LAO8 | 5/8 | 0.01041809 | 100 |
| Q10 | 5/8 | 0.00962880 | 62 |
| Q40 | 6/8 | 0.01124162 | 123 |
| Q40_LR02 | 6/8 | 0.01048199 | 104 |

**Frozen:** Q40_LR02, at most 40 latent updates, learning rate 0.02; saved uint8 maximum pixel change 8; first feasible E-margin ≥0.1 and LPIPS≤0.03/PSNR≥30, followed by five measured latent bisection probes. All-sample mean-LPIPS guard ≤0.015.

The selected candidate adds one development success and loses none. Its mean LPIPS rises by about 0.61%. Q10 is the legal first-ten-update prefix of Q40, not an independent compute run. OLD_LAO8 export metadata contains an informational zero step count because the export omits optimization history; its raw source confirms 100 total updates. See development_step_count_note.json. This does not affect candidate ranking.

The independent core audit verifies the actual saved candidates, all gradient histories, resets and frozen source hashes. The previous-results preservation audit checks 1,873 archived and live files. No earlier validation or peer result was overwritten.

Second validation was launched once at results/m3_lao_v2/validate_20261018_172351_7bu0ig1v using freeze.json. No second-batch result may change this selection.
