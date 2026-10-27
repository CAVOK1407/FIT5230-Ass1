# Frozen LAO v2 — second validation batch

Completed on 2026-10-18. See `completion.json` for final status and `SHA256SUMS.json` for the sealed artifact inventory.

**Main result:** NEW_LAO8 40/41 versus OLD_LAO8 25/41 on E/original PNG under the predeclared low-distortion screen. Fifteen gains, no losses; paired difference +36.59 percentage points, bootstrap 95% CI [21.95, 51.22], exact McNemar p=0.0000610. All 50 new LAO images pass LPIPS≤0.03 and PSNR≥30; mean LPIPS 0.007919 versus old 0.007271. All saved output budgets are integer pixel difference≤8, with zero tolerance.

The standalone latent branch improves target-E success while retaining the low-distortion range. PGD10 still has higher classifier-only success (41/41). R/S transfer and JPEG robustness do not improve. This is not a claim of general superiority, equal-compute advantage, new-generator generalization, or dominance in the original hybrid pipeline.

- `plan.json`: original commands, frozen sources/checkpoints/protocol and candidate binding.
- `frozen_code/`, `protocol.md`: archived implementation and protocol before validation.
- `data/`: fixed second batch and provenance evidence; generated only after candidate freeze.
- `RAW_RECON/`, `RAW_LAO/`, `old/`, `new/`, `pgd/`: original outputs, histories, independent zero-step controls and budgeted PNGs.
- `eval/`: unchanged 3,600 per-image evaluation rows, real-image predictions and aggregate metrics.
- `report/report.md`, `report/report.json`: complete statistics, quality, transfer, integrity and limits.
- `independent_data_audit.json`, `independent_integrity_audit.json`, `independent_statistics_audit.json`, `independent_pgd_audit.json`: independent verification.

## Report-only numerical recovery

All generation, attacks and evaluation stages completed in the original run. The frozen reporter stopped on image0041: its stored single-image E margin differs from the independent batch16 margin by 1.017749309539795e-5, just above its 1e-5 consistency tolerance.

`report_recovery/verify_batching.py` re-scored the same 50 unchanged PNGs. Every single-image score exactly reproduced the stored score; every batch16 score exactly reproduced the evaluation CSV. All predictions, margin-buffer decisions and LPIPS values agree. No attack was rerun.

An isolated copy of the reporter raises only the E-margin absolute consistency tolerance to 2e-5. All 36 frozen runtime sources remain unchanged. The pixel budget remains zero-tolerance; statistical tests and quality thresholds are unchanged. The patch, original failed session, original failure log, numerical audit, amendment and recovery wrapper are preserved. The report explicitly records this amendment.

The second batch is now observed and must not select future parameters. A further development cycle needs a new validation batch.
