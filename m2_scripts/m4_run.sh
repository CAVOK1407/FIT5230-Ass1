#!/usr/bin/env bash
# FIT5230 · Milestone 4 — unconstrained-enhancement demo: fine-tune the Control-VAE control
# branch so the reconstruction alone evades the detectors. Run from inside StealthDiffusion
# with the fit5230m2 environment active:   conda activate fit5230m2
#
#   bash m4_run.sh smoke    # 1 epoch on 4 images: checks that everything loads (~2 min)
#   bash m4_run.sh train    # fine-tune on the 150 held-out SD1.5 images + log dev metrics
#   bash m4_run.sh x4       # reconstruction only (no PGD, no LAO) with the new weights, 100 test images
#   bash m4_run.sh a5       # official pipeline with the new weights
#   bash m4_run.sh ens      # ensemble pipeline with the new weights
#   bash m4_run.sh eval     # evaluate new runs next to the M3 originals (X4 / A5 / ENS)
#   bash m4_run.sh all      # train + x4 + a5 + ens + eval
#   bash m4_run.sh status
#
# Nothing under results/m3_* or checkpoints/Controlvae.pt is modified. New weights go to
# checkpoints/Controlvae_m4.pt, new runs to results/m4_cvae/.
set -euo pipefail

DEVICE="${DEVICE:-mps}"
SD_MODEL="${SD_MODEL:-Manojb/stable-diffusion-2-1-base}"
PY="${PY:-python}"
EPOCHS="${EPOCHS:-30}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"   # e.g. TRAIN_EXTRA="--train_zero_convs_only --lr 1e-4"
N_TEST="${N_TEST:-100}"
R="results/m4_cvae"
LOG="$R/logs"
W="checkpoints/Controlvae_m4.pt"
mkdir -p "$R" "$LOG"
stage="${1:-status}"

# the three 50-image SD1.5 batches generated for the M3 LAO validations (disjoint from the
# 100-image test set and from each other, see docs/M3_LAO_V3_RESULTS_ZH.md)
TRAIN_DIRS=(results/m3_lao_validation/validation_20261018_154651_tm5zwx4p/data/gen/test
            results/m3_lao_v2/validate_20261018_172351_7bu0ig1v/data/gen/test
            results/m3_lao_v3/validate_20261018_195753_zxupemab/data/gen/test)

common=(--device "$DEVICE" --images_root data15/gen/test --max_images "$N_TEST"
        --pretrained_diffusion_path "$SD_MODEL" --encoder_weights "$W" --seed 42 --resume)

run_cfg() {  # run_cfg <name> <extra args...>
  local name="$1"; shift
  echo "== [$(date '+%F %T')] $name -> $R/$name" | tee -a "$LOG/$stage.log"
  $PY m2_stealth.py "${common[@]}" "$@" --save_dir "$R/$name" 2>&1 | tee -a "$LOG/$stage.log"
}

case "$stage" in
  smoke)
    $PY m4_finetune_cvae.py --device "$DEVICE" --train_dirs "${TRAIN_DIRS[@]}" --dev_dir data15/gen/dev \
        --real_dev_dir data15/real/dev --pretrained_diffusion_path "$SD_MODEL" \
        --epochs 1 --max_train 4 --out "$R/smoke/Controlvae_smoke.pt" --log_dir "$R/smoke" 2>&1 | tee -a "$LOG/$stage.log"
    ;;
  train)
    $PY m4_finetune_cvae.py --device "$DEVICE" --train_dirs "${TRAIN_DIRS[@]}" --dev_dir data15/gen/dev \
        --real_dev_dir data15/real/dev --pretrained_diffusion_path "$SD_MODEL" \
        --epochs "$EPOCHS" --out "$W" --log_dir "$R/train" $TRAIN_EXTRA 2>&1 | tee -a "$LOG/$stage.log"
    [ -f "$W" ] || { echo "no checkpoint passed the quality gate; not running attacks" >&2; exit 1; }
    ;;
  x4)  [ -f "$W" ] || { echo "missing $W: run train first" >&2; exit 1; }
       run_cfg X4_m4  --encode_source clean --control_source clean --iterations 0 ;;
  a5)  run_cfg A5_m4 ;;
  ens) run_cfg ENS_m4 --surrogates E,R,D ;;
  eval)
    $PY m2_evaluate.py --device "$DEVICE" --real_dir data15/real/test --out "$R/eval" \
        --conditions none,jpeg95,jpeg75,jpeg50,resize075,resize05 \
        --runs X4_m4=$R/X4_m4 A5_m4=$R/A5_m4 ENS_m4=$R/ENS_m4 \
               X4=results/m3_sd15/X4 A5=results/m3_sd15/A5 ENS=results/m3_sd15/ENS 2>&1 | tee -a "$LOG/$stage.log"
    ;;
  all)
    bash "$0" train && bash "$0" x4 && bash "$0" a5 && bash "$0" ens && bash "$0" eval
    ;;
  status)
    [ -f "$R/train/train_log.jsonl" ] && { echo "-- train log (last 3):"; tail -3 "$R/train/train_log.jsonl" | cut -c1-200; }
    for d in "$R"/*/; do [ -f "$d/records.jsonl" ] && printf "%-28s %4d images\n" "$d" "$(grep -c . "$d/records.jsonl")"; done
    [ -f "$R/eval/summary.csv" ] && { echo "-- eval summary:"; head -1 "$R/eval/summary.csv"; grep ",none," "$R/eval/summary.csv"; }
    true
    ;;
  *) echo "unknown stage $stage" >&2; exit 1 ;;
esac
