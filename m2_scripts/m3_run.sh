#!/usr/bin/env bash
# FIT5230 · Milestone 3 — local runs on macOS (MPS). Run from inside the StealthDiffusion
# folder with the m2_*.py / m3_*.py scripts copied in and checkpoints/ populated.
# Activate the conda env first:  conda activate fit5230m2
#
#   bash m3_run.sh t1        # T1: A5 / A2 / X2 / X3 / X4 on the local SD 2.1 test set (20 images), report + eval
#   bash m3_run.sh pgdcheck  # own PGD loop vs torchattacks: must be pixel-identical (3 images)
#   bash m3_run.sh data15    # SD 1.5 data set, dev 10 / test 100, + real photos (Imagenette) -> data15/
#   bash m3_run.sh t4        # T4: all T1 configurations + B10 / ENS / FULL on data15, 100 images (long, resumable)
#   bash m3_run.sh t2        # T2: capped variant (own PGD, stop at the real-dev median score)   -> CAP_E, CAP_ENS
#   bash m3_run.sh t2b       # T2b: capped variant with step bisection (lands at the cap)            -> CAPB_E, CAPB_ENS
#   bash m3_run.sh t3        # T3: EOT (JPEG 75 + 0.5x resize) inside PGD                        -> EOT_E, EOT_ENS
#   bash m3_run.sh t5        # T5: pixel PGD baseline, eps 8/255, step 2/255, 10 steps           -> PGD8
#   bash m3_run.sh eval15    # evaluation of everything that exists under results/m3_sd15
#   bash m3_run.sh status    # how many images each run has finished
#
# Every stage appends to results/m3_logs/<stage>.log. Every attack run uses --resume, so a stage
# can be interrupted and restarted. Nothing here modifies results/B00 .. results/FULL (M2).
set -euo pipefail

DEVICE="${DEVICE:-mps}"
SD_MODEL="${SD_MODEL:-Manojb/stable-diffusion-2-1-base}"          # the attack's diffusion model (official)
GEN_MODEL="${GEN_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}" # generator of the M3 data set
PY="${PY:-python}"
N_T1="${N_T1:-20}"
N_T4="${N_T4:-100}"
N_DEV15="${N_DEV15:-10}"
R21=results/m3_sd21      # runs on the local SD 2.1 data (data/) — comparable with results/B00 (M2)
R15=results/m3_sd15      # runs on the SD 1.5 data (data15/)   — comparable with the M2 Colab numbers
LOG=results/m3_logs
mkdir -p "$R21" "$R15" "$LOG"
stage="${1:-status}"

common21=(--device "$DEVICE" --images_root data/gen/test --max_images "$N_T1"
          --pretrained_diffusion_path "$SD_MODEL" --encoder_weights ./checkpoints/Controlvae.pt --seed 42 --resume)
common15=(--device "$DEVICE" --images_root data15/gen/test --max_images "$N_T4"
          --pretrained_diffusion_path "$SD_MODEL" --encoder_weights ./checkpoints/Controlvae.pt --seed 42 --resume)

# configuration -> extra switches (paper draft, Table "Pipeline configurations")
cfg_args() {
  case "$1" in
    A5|B00) echo "" ;;                                                   # released pipeline
    A2)  echo "--iterations 0" ;;                                        # no latent optimisation
    X2)  echo "--control_source clean" ;;                                # clean control image
    X3)  echo "--control_source encoded" ;;                              # second PGD call removed
    X4)  echo "--encode_source clean --control_source clean --iterations 0" ;;  # reconstruction floor
    A4a) echo "--encode_source clean --control_source clean" ;;
    A4b) echo "--encode_source clean --control_source pgd" ;;
    B10|X5) echo "--decode_path final" ;;                                # loss on the skip decoder
    ENS) echo "--surrogates E,R,D" ;;
    FULL) echo "--surrogates E,R,D --decode_path final --loss margin --lpips_weight 1.0 --di_prob 0.5" ;;
    *) echo "unknown configuration $1" >&2; exit 1 ;;
  esac
}

run_cfg() {  # run_cfg <results root> <config name> <common args...>
  local root="$1" name="$2"; shift 2
  echo "== [$(date '+%F %T')] $name -> $root/$name" | tee -a "$LOG/$stage.log"
  # shellcheck disable=SC2046
  $PY m2_stealth.py "$@" $(cfg_args "$name") --save_dir "$root/$name" 2>&1 | tee -a "$LOG/$stage.log"
}

alpha_cap=$($PY -c "print(4/255/10)")
real_dev21=data/real/dev
real_dev15=data15/real/dev

case "$stage" in
  t1)
    for c in A5 A2 X2 X3 X4; do run_cfg "$R21" "$c" "${common21[@]}"; done
    $PY m3_report.py --runs B00=results/B00 A5=$R21/A5 A2=$R21/A2 X2=$R21/X2 X3=$R21/X3 X4=$R21/X4 \
        --pairs A5:A2 A5:X3 A5:X2 X2:X4 A5:B00 --out $R21/t1_report.md 2>&1 | tee -a "$LOG/$stage.log"
    $PY m2_evaluate.py --device "$DEVICE" --real_dir data/real/test --out $R21/eval --conditions none,jpeg75,resize05 \
        --runs A5=$R21/A5 A2=$R21/A2 X2=$R21/X2 X3=$R21/X3 X4=$R21/X4 \
               A1=$R21/A5:_pgd1.png X1=$R21/A5:_control.png 2>&1 | tee -a "$LOG/$stage.log"
    ;;
  pgdcheck)
    $PY m2_stealth.py "${common21[@]}" --max_images 3 --pgd_impl own --save_dir "$R21/A5_ownpgd" 2>&1 | tee -a "$LOG/$stage.log"
    $PY - <<'EOF' 2>&1 | tee -a "$LOG/$stage.log"
import numpy as np, json
from PIL import Image
R="results/m3_sd21"
for i in ["0000","0001","0002"]:
    for suf in ["_pgd1.png","_control.png","_adv_image.png"]:
        a=np.asarray(Image.open(f"{R}/A5/{i}{suf}")).astype(int); b=np.asarray(Image.open(f"{R}/A5_ownpgd/{i}{suf}")).astype(int)
        d=np.abs(a-b); print(f"{i}{suf:<16} max diff {d.max():>3}  pixels differing {int((d>0).sum()):>6} / {d.size}")
ra=[json.loads(l) for l in open(f"{R}/A5/records.jsonl") if l.strip()][:3]
rb=[json.loads(l) for l in open(f"{R}/A5_ownpgd/records.jsonl") if l.strip()][:3]
for a,b in zip(ra,rb):
    print(a["image_id"], "E score after PGD1  torchattacks %.4f  own %.4f" % (a["pgd"]["E"]["real_margin"], b["pgd"]["E"]["real_margin"]))
print("PASS: own loop reproduces torchattacks" if all(np.array_equal(np.asarray(Image.open(f"{R}/A5/{i}_pgd1.png")),np.asarray(Image.open(f"{R}/A5_ownpgd/{i}_pgd1.png"))) for i in ["0000","0001","0002"]) else "FAIL: x1 differs")
EOF
    ;;
  data15)
    $PY m2_data.py real     --out data15 --n_dev "$N_DEV15" --n_test "$N_T4" 2>&1 | tee -a "$LOG/$stage.log"
    $PY m2_data.py generate --out data15 --n_dev "$N_DEV15" --n_test "$N_T4" --model "$GEN_MODEL" --device "$DEVICE" 2>&1 | tee -a "$LOG/$stage.log"
    $PY m2_data.py check    --out data15 --device "$DEVICE" 2>&1 | tee -a "$LOG/$stage.log"
    ;;
  t4)
    for c in A5 A2 X2 X3 X4 B10 ENS FULL; do run_cfg "$R15" "$c" "${common15[@]}"; done
    ;;
  t2)
    cap=(--pgd_impl own --cap auto --real_dev_dir "$real_dev15" --pgd_alpha "$alpha_cap" --pgd_steps 40 --loss margin --kappa cap)
    run_cfg "$R15" CAP_E   "${common15[@]}" "${cap[@]}"
    run_cfg "$R15" CAP_ENS "${common15[@]}" "${cap[@]}" --surrogates E,R,D
    ;;
  t2b)   # capped variant with step bisection: the score lands at the cap instead of overshooting it
    cap=(--pgd_impl own --cap auto --cap_bisect --real_dev_dir "$real_dev15" --pgd_alpha "$alpha_cap" --pgd_steps 40 --loss margin --kappa cap)
    run_cfg "$R15" CAPB_E   "${common15[@]}" "${cap[@]}"
    run_cfg "$R15" CAPB_ENS "${common15[@]}" "${cap[@]}" --surrogates E,R,D
    ;;
  t3)
    eot=(--pgd_impl own --pgd_eot both --eot_jpeg_q 75 --eot_resize 0.5)
    run_cfg "$R15" EOT_E   "${common15[@]}" "${eot[@]}"
    run_cfg "$R15" EOT_ENS "${common15[@]}" "${eot[@]}" --surrogates E,R,D
    ;;
  t5)
    run_cfg "$R15" PGD8 "${common15[@]}" --pgd_only --pgd_impl own --pgd_eps "$($PY -c 'print(8/255)')" --pgd_alpha "$($PY -c 'print(2/255)')" --pgd_steps 10
    ;;
  eval15)
    runs=()
    for d in "$R15"/*/; do n=$(basename "$d"); [ -f "$d/records.jsonl" ] && runs+=("$n=$d"); done
    [ -d "$R15/A5" ] && runs+=("A1=$R15/A5:_pgd1.png" "X1=$R15/A5:_control.png")
    $PY m2_evaluate.py --device "$DEVICE" --real_dir data15/real/test --out $R15/eval \
        --conditions none,jpeg95,jpeg75,jpeg50,resize075,resize05 --runs "${runs[@]}" 2>&1 | tee -a "$LOG/$stage.log"
    ;;
  status)
    for d in results/B00 results/B10 results/ENS results/FULL "$R21"/*/ "$R15"/*/; do
      [ -f "$d/records.jsonl" ] && printf "%-32s %4d images\n" "$d" "$(grep -c . "$d/records.jsonl")"
    done
    ls data15/gen/test 2>/dev/null | grep -c png | sed 's/^/data15\/gen\/test: /' || true
    ;;
  *) echo "usage: bash m3_run.sh [t1|pgdcheck|data15|t4|t2|t2b|t3|t5|eval15|status]"; exit 1 ;;
esac
echo "[m3_run] stage '$stage' done $(date '+%F %T')" | tee -a "$LOG/$stage.log"
