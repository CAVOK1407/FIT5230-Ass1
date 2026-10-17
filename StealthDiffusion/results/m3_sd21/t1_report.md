# T1 report — 20 images present in all runs: B00 (results/B00), A5 (results/m3_sd21/A5), A2 (results/m3_sd21/A2), X2 (results/m3_sd21/X2), X3 (results/m3_sd21/X3), X4 (results/m3_sd21/X4)
- B00: surrogates=E, decode_path=plain, loss=ce, iterations=10, pretrained_diffusion_path=Manojb/stable-diffusion-2-1-base, device=mps, seed=42
- A5: surrogates=E, decode_path=plain, loss=ce, iterations=10, encode_source=pgd1, control_source=pgd, pgd_impl=torchattacks, pgd_eot=none, lao_eot=none, pretrained_diffusion_path=Manojb/stable-diffusion-2-1-base, device=mps, seed=42
- A2: surrogates=E, decode_path=plain, loss=ce, iterations=0, encode_source=pgd1, control_source=pgd, pgd_impl=torchattacks, pgd_eot=none, lao_eot=none, pretrained_diffusion_path=Manojb/stable-diffusion-2-1-base, device=mps, seed=42
- X2: surrogates=E, decode_path=plain, loss=ce, iterations=10, encode_source=pgd1, control_source=clean, pgd_impl=torchattacks, pgd_eot=none, lao_eot=none, pretrained_diffusion_path=Manojb/stable-diffusion-2-1-base, device=mps, seed=42
- X3: surrogates=E, decode_path=plain, loss=ce, iterations=10, encode_source=pgd1, control_source=encoded, pgd_impl=torchattacks, pgd_eot=none, lao_eot=none, pretrained_diffusion_path=Manojb/stable-diffusion-2-1-base, device=mps, seed=42
- X4: surrogates=E, decode_path=plain, loss=ce, iterations=0, encode_source=clean, control_source=clean, pgd_impl=torchattacks, pgd_eot=none, lao_eot=none, pretrained_diffusion_path=Manojb/stable-diffusion-2-1-base, device=mps, seed=42

## 0. Clean image, all runs (must agree: the clean image is never modified)
| detector | caught (pred=AI) per run | max |Δ score| across runs | decisions that differ |
|---|---|---|---|
| D | B00:19, A5:19, A2:19, X2:19, X3:19, X4:19 | 0.0000 | 0 |
| E | B00:14, A5:14, A2:14, X2:14, X3:14, X4:14 | 0.0000 | 0 |
| R | B00:12, A5:12, A2:12, X2:12, X3:12, X4:12 | 0.0000 | 0 |
| S | B00:12, A5:12, A2:12, X2:12, X3:12, X4:12 | 0.0000 | 0 |

## 1. Stage table — of the caught images, how many read 'real' at each stage (mean score over all images)

### B00
| detector | caught | x1 (PGD 1) | x2 (control) | plain decode | saved image |
|---|---|---|---|---|---|
| D | 19 | 11/19 (+0.2) | — | 9/19 (+0.1) | 11/19 (+0.7) |
| E | 14 | 14/14 (+114.4) | — | 14/14 (+88.6) | 14/14 (+153.2) |
| R | 12 | 8/12 (+2.8) | — | 9/12 (+3.8) | 8/12 (+4.3) |
| S | 12 | 11/12 (+4.6) | — | 11/12 (+7.2) | 10/12 (+5.8) |
(mean 10.9 s/image)

### A5
| detector | caught | x1 (PGD 1) | x2 (control) | plain decode | saved image |
|---|---|---|---|---|---|
| D | 19 | 11/19 (+0.2) | 11/19 (+0.7) | 9/19 (+0.1) | 11/19 (+0.7) |
| E | 14 | 14/14 (+114.4) | 14/14 (+157.0) | 14/14 (+88.6) | 14/14 (+153.2) |
| R | 12 | 8/12 (+2.8) | 8/12 (+4.0) | 9/12 (+3.8) | 8/12 (+4.3) |
| S | 12 | 11/12 (+4.6) | 12/12 (+5.2) | 11/12 (+7.2) | 10/12 (+5.8) |
(mean 10.5 s/image; PGD steps pgd1: mean 10.0, pgd2: mean 10.0)

### A2
| detector | caught | x1 (PGD 1) | x2 (control) | plain decode | saved image |
|---|---|---|---|---|---|
| D | 19 | 11/19 (+0.2) | 11/19 (+0.7) | 9/19 (-0.8) | 11/19 (+0.7) |
| E | 14 | 14/14 (+114.4) | 14/14 (+157.0) | 7/14 (+1.8) | 14/14 (+153.2) |
| R | 12 | 8/12 (+2.8) | 8/12 (+4.0) | 7/12 (+1.8) | 8/12 (+4.2) |
| S | 12 | 11/12 (+4.6) | 12/12 (+5.2) | 11/12 (+6.4) | 10/12 (+5.8) |
(mean 3.4 s/image; PGD steps pgd1: mean 10.0, pgd2: mean 10.0)

### X2
| detector | caught | x1 (PGD 1) | x2 (control) | plain decode | saved image |
|---|---|---|---|---|---|
| D | 19 | 11/19 (+0.2) | 0/19 (-4.1) | 9/19 (+0.1) | 1/19 (-3.7) |
| E | 14 | 14/14 (+114.4) | 0/14 (-2.0) | 14/14 (+88.6) | 1/14 (-1.8) |
| R | 12 | 8/12 (+2.8) | 0/12 (-1.8) | 9/12 (+3.8) | 1/12 (-1.0) |
| S | 12 | 11/12 (+4.6) | 0/12 (-2.2) | 11/12 (+7.2) | 10/12 (+5.7) |
(mean 9.8 s/image; PGD steps pgd1: mean 10.0)

### X3
| detector | caught | x1 (PGD 1) | x2 (control) | plain decode | saved image |
|---|---|---|---|---|---|
| D | 19 | 11/19 (+0.2) | 11/19 (+0.2) | 9/19 (+0.1) | 12/19 (+0.1) |
| E | 14 | 14/14 (+114.4) | 14/14 (+114.4) | 14/14 (+88.6) | 14/14 (+107.9) |
| R | 12 | 8/12 (+2.8) | 8/12 (+2.9) | 9/12 (+3.8) | 8/12 (+3.1) |
| S | 12 | 11/12 (+4.6) | 11/12 (+4.7) | 11/12 (+7.2) | 10/12 (+6.1) |
(mean 9.7 s/image; PGD steps pgd1: mean 10.0)

### X4
| detector | caught | x1 (PGD 1) | x2 (control) | plain decode | saved image |
|---|---|---|---|---|---|
| D | 19 | — | 0/19 (-4.1) | 0/19 (-3.4) | 1/19 (-3.8) |
| E | 14 | — | 0/14 (-2.0) | 3/14 (-0.9) | 0/14 (-2.2) |
| R | 12 | — | 0/12 (-1.8) | 4/12 (+0.2) | 0/12 (-1.4) |
| S | 12 | — | 0/12 (-2.2) | 9/12 (+4.2) | 10/12 (+5.7) |
(mean 2.5 s/image)

## 2. Contrasts between runs (saved image)

### A5 vs A2
Saved files: RMS difference mean 0.881 grey levels (max over images 1.314); largest per-pixel difference mean 5.8; identical pixels 0.513; PSNR to input A5 34.02 dB, A2 34.12 dB
| detector | caught | real after A5 | real after A2 | decisions differ (all 20) | Wilson 95% of discordance | mean |Δ score| | max |Δ score| |
|---|---|---|---|---|---|---|---|
| D | 19 | 11 | 11 | 0 | [0.00, 0.16] | 0.08 | 0.16 |
| E | 14 | 14 | 14 | 0 | [0.00, 0.16] | 0.49 | 1.17 |
| R | 12 | 8 | 8 | 0 | [0.00, 0.16] | 0.09 | 0.29 |
| S | 12 | 10 | 10 | 0 | [0.00, 0.16] | 0.07 | 0.32 |
Decisions that differ, summed over detectors: 0 of 80

### A5 vs X3
Saved files: RMS difference mean 2.715 grey levels (max over images 2.853); largest per-pixel difference mean 8.9; identical pixels 0.132; PSNR to input A5 34.02 dB, X3 35.68 dB
| detector | caught | real after A5 | real after X3 | decisions differ (all 20) | Wilson 95% of discordance | mean |Δ score| | max |Δ score| |
|---|---|---|---|---|---|---|---|
| D | 19 | 11 | 12 | 1 | [0.01, 0.24] | 1.03 | 2.62 |
| E | 14 | 14 | 14 | 0 | [0.00, 0.16] | 45.26 | 128.06 |
| R | 12 | 8 | 8 | 0 | [0.00, 0.16] | 1.24 | 2.51 |
| S | 12 | 10 | 10 | 0 | [0.00, 0.16] | 0.90 | 3.29 |
Decisions that differ, summed over detectors: 1 of 80

### A5 vs X2
Saved files: RMS difference mean 4.051 grey levels (max over images 4.248); largest per-pixel difference mean 15.7; identical pixels 0.100; PSNR to input A5 34.02 dB, X2 38.64 dB
| detector | caught | real after A5 | real after X2 | decisions differ (all 20) | Wilson 95% of discordance | mean |Δ score| | max |Δ score| |
|---|---|---|---|---|---|---|---|
| D | 19 | 11 | 1 | 10 | [0.30, 0.70] | 4.38 | 9.60 |
| E | 14 | 14 | 1 | 14 | [0.48, 0.85] | 154.96 | 206.31 |
| R | 12 | 8 | 1 | 7 | [0.18, 0.57] | 5.20 | 8.31 |
| S | 12 | 10 | 10 | 0 | [0.00, 0.16] | 1.74 | 5.94 |
Decisions that differ, summed over detectors: 31 of 80

### X2 vs X4
Saved files: RMS difference mean 1.183 grey levels (max over images 1.493); largest per-pixel difference mean 8.2; identical pixels 0.379; PSNR to input X2 38.64 dB, X4 38.97 dB
| detector | caught | real after X2 | real after X4 | decisions differ (all 20) | Wilson 95% of discordance | mean |Δ score| | max |Δ score| |
|---|---|---|---|---|---|---|---|
| D | 19 | 1 | 1 | 0 | [0.00, 0.16] | 0.39 | 1.38 |
| E | 14 | 1 | 0 | 3 | [0.05, 0.36] | 0.77 | 1.77 |
| R | 12 | 1 | 0 | 1 | [0.01, 0.24] | 0.61 | 1.40 |
| S | 12 | 10 | 10 | 0 | [0.00, 0.16] | 0.51 | 2.00 |
Decisions that differ, summed over detectors: 4 of 80

### A5 vs B00
Saved files: RMS difference mean 0.000 grey levels (max over images 0.000); largest per-pixel difference mean 0.0; identical pixels 1.000; PSNR to input A5 34.02 dB, B00 34.02 dB
| detector | caught | real after A5 | real after B00 | decisions differ (all 20) | Wilson 95% of discordance | mean |Δ score| | max |Δ score| |
|---|---|---|---|---|---|---|---|
| D | 19 | 11 | 11 | 0 | [0.00, 0.16] | 0.00 | 0.00 |
| E | 14 | 14 | 14 | 0 | [0.00, 0.16] | 0.00 | 0.00 |
| R | 12 | 8 | 8 | 0 | [0.00, 0.16] | 0.00 | 0.00 |
| S | 12 | 10 | 10 | 0 | [0.00, 0.16] | 0.00 | 0.00 |
Decisions that differ, summed over detectors: 0 of 80

## 3. H1(b): success rate of X2 minus X4 (conditional on images caught before the attack)
| detector | caught | X4 real | X2 real | difference |
|---|---|---|---|---|
| D | 19 | 1 (0.05) | 1 (0.05) | +0.00 |
| E | 14 | 0 (0.00) | 1 (0.07) | +0.07 |
| R | 12 | 0 (0.00) | 1 (0.08) | +0.08 |
| S | 12 | 10 (0.83) | 10 (0.83) | +0.00 |
