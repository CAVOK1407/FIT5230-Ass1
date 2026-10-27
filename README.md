# FIT5230 Assignment 1 — Dark.ModelBreakers (Theme 1, Dark side)

Attack side of Theme 1: we reproduce, measure and improve **StealthDiffusion** (Zhou et al., ACM MM 2024, [arXiv:2408.05669](https://arxiv.org/abs/2408.05669)), an attack that hides Stable-Diffusion images from forensic detectors. The official code lives in `StealthDiffusion/` (fixed at commit `a156fb0` of [wyczzy/StealthDiffusion](https://github.com/wyczzy/StealthDiffusion)) together with our scripts.

## Where things are

| Milestone | Entry point |
| --- | --- |
| M1 — reproduction | `notebooks/M1_original_DO_NOT_EDIT.ipynb` |
| M2 — switches, variants, evaluation protocol | `notebooks/M2_Submission.ipynb`, scripts `StealthDiffusion/m2_*.py`, `StealthDiffusion/run_local_mac.sh`, [README_M2.md](README_M2.md) |
| M3 — hypothesis tests, capped / EOT variants, LAO experiments | `StealthDiffusion/m3_run.sh`, `StealthDiffusion/m3_*.py`, results under `StealthDiffusion/results/m3_*`, Colab bundle in [m3_colab/](m3_colab/README.md), notebook `notebooks/M3_Submission.ipynb` |
| M4 — defence demos, Control-VAE fine-tuning (individual) | [m4/README.md](m4/README.md), `StealthDiffusion/m4_finetune_cvae.py`, `StealthDiffusion/m4_run.sh` |

`m2_scripts/` holds the git-tracked source of every script; `StealthDiffusion/` holds the copy that runs inside the official code base. Edit in `StealthDiffusion/`, then sync to `m2_scripts/`.

## Setup

```bash
git clone https://github.com/CAVOK1407/FIT5230-Ass1.git
cd FIT5230-Ass1/StealthDiffusion
```

1. Environment: Python 3.10, `torch 2.2.2`, `diffusers 0.12.1`, `timm 0.6.12`, plus `StealthDiffusion/requirements.txt`. Step-by-step instructions for macOS (Apple Silicon, MPS) are in [docs/README_LOCAL_MAC_ZH.md](docs/README_LOCAL_MAC_ZH.md); the same packages work on CUDA (`--device cuda`). The device shim `m2_device.py` redirects the official `.cuda()` calls.
2. Weights (about 2.6 GB, not in git): the six files below from the official release (links in [StealthDiffusion/README.md](StealthDiffusion/README.md)) go into `StealthDiffusion/checkpoints/` with these exact names:

   ```text
   checkpoints/
   ├── efficientnet-b0.pth
   ├── resnet50.pth
   ├── deit.pth
   ├── swin-t.pth
   ├── Controlvae.pt
   └── dncnn_color_blind.pth
   ```

   The LPIPS VGG weights (`.cache/checkpoints/vgg16-397923af.pth`) and the Stable Diffusion 2.1-base pipeline are downloaded automatically on first run.
3. Data: generated images and real photographs are produced by the scripts (`bash run_local_mac.sh data` for the SD 2.1 set, `bash m3_run.sh data15` for the SD 1.5 set + Imagenette). They are not in git; the manifests and all per-image records and evaluation CSVs are.

## Run

```bash
conda activate fit5230m2            # or your own environment
bash run_local_mac.sh check         # verifies torch / device / the six checkpoints
bash m3_run.sh status               # what has been run
bash m3_run.sh t4                   # M3: all configurations on the SD 1.5 set (100 images, resumable)
bash m3_run.sh eval15               # evaluation -> results/m3_sd15/eval/summary.csv
bash m4_run.sh smoke && bash m4_run.sh all   # M4: fine-tune the control branch, re-run X4 / A5 / ENS, evaluate
```

The M4 defence demos need only the committed CSVs and `pandas`:

```bash
cd m4/defense && python m4_defense_calibrated.py --data_dir ../data --alpha 0.05 --beta 0 --out /tmp/out.csv
```

Reading the existing results (`StealthDiffusion/results/*/eval/summary.csv`, `records.jsonl`) needs no weights and no GPU.

## Conventions

- Every rate is recomputed from saved PNG files, never from in-memory tensors (`m2_evaluate.py`).
- Conditional ASR counts only images the detector caught before the attack; the denominator is reported next to every rate.
- Detectors are always in `eval()` mode (see the note on `Ensemble` in `m2_stealth.py`).
- Hypotheses for M3 were frozen before the runs: tag `m3-hypotheses-frozen`.
