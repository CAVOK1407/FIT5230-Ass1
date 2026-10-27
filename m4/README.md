# Milestone 4 — individual report demo (Yunjie Wang)

Everything referenced in the Milestone 4 individual report (submitted on Moodle) lives here or in `StealthDiffusion/`.

## Role reversal: defending against our own attack (report Section IV)

| File | What it does |
|---|---|
| `defense/m4_defense_twosided.py` | Defence 1 — two-sided score test (upper bound from real images), Table II |
| `defense/m4_defense_preprocess.py` | Defence 2 — JPEG / resize before detection, unchanged thresholds, Table III |
| `defense/m4_defense_calibrated.py` | Defence 3 — calibrated two-sided thresholds + 2-of-4 vote, Tables IV, V, VIII |
| `defense/make_tables.py` | Generates every LaTeX table in the report from the CSV outputs |
| `defense/results/*.csv` | Script outputs |
| `data/` | Archived Milestone 3 measurements the defences are computed from (100 SD 1.5 images, 100 Imagenette photographs, 4 detectors, 6 conditions) |

Run (a few seconds, needs only `pandas` and `numpy`):

```bash
cd m4/defense
python m4_defense_twosided.py   --data_dir ../data --out results/twosided_results.csv
python m4_defense_preprocess.py --data_dir ../data --out results/preprocess_results.csv
python m4_defense_calibrated.py --data_dir ../data --alpha 0.05 --beta 0 --out results/calibrated_a05_b0.csv
CVAE_DIR=../data_cvae python make_tables.py ../data . tables_defence.tex   # LaTeX tables of the report
```

## Unconstrained enhancement: fine-tuning the Control-VAE control branch (report Section III)

| File | What it does |
|---|---|
| `StealthDiffusion/m4_finetune_cvae.py` | Fine-tunes the control encoder against E/R/D with a margin loss + L1 + LPIPS; quality-gated checkpoint selection |
| `StealthDiffusion/m4_run.sh` | `smoke` / `train` / `x4` / `a5` / `ens` / `eval` stages (runs `m2_stealth.py` and `m2_evaluate.py` with the new weights) |
| `StealthDiffusion/results/m4_cvae/` | Per-image records (`records.jsonl`), run configs and the evaluation CSVs for X4_m4 / A5_m4 / ENS_m4 next to the Milestone 3 originals (PNG outputs and the `.pt` weights are not committed) |
| `data_cvae/train_log.jsonl` | Per-epoch dev metrics (Table VI) |

Run from `StealthDiffusion/` in the `fit5230m2` environment with the six checkpoints in `checkpoints/`:

```bash
bash m4_run.sh smoke     # ~2 min, checks loading
bash m4_run.sh all       # train (30 epochs, ~20 min on an M5 Max) + three 100-image runs + evaluation
```

The 150 training images are the three SD 1.5 batches generated for the Milestone 3 LAO validations (`results/m3_lao_*/.../data/gen/test`), disjoint from the 100-image test set; they are not committed either (see `.gitignore`), the generation plans and manifests are.
