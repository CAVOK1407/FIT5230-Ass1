#!/usr/bin/env bash
# FIT5230 M2 - local run on macOS (MPS). Run from inside the official StealthDiffusion
# checkout, with the m2_*.py scripts copied in and checkpoints/ populated (see
# README_LOCAL_MAC_ZH.md steps 1-4). Activate the conda env first: conda activate fit5230m2
#
#   bash run_local_mac.sh all                       # small: dev 6 / test 20 / attack 6
#   N_TEST=100 N_ATTACK=50 bash run_local_mac.sh all
#   bash run_local_mac.sh data|attack|eval|challenge
set -euo pipefail

DEVICE="${DEVICE:-mps}"
N_DEV="${N_DEV:-6}"
N_TEST="${N_TEST:-20}"
N_ATTACK="${N_ATTACK:-6}"
REAL_SRC="${REAL_SRC:-}"                    # optional folder of real photos; empty -> Imagenette
GEN_MODEL="${GEN_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
SD_MODEL="${SD_MODEL:-Manojb/stable-diffusion-2-1-base}"
PY="${PY:-python}"

stage="${1:-all}"

check() {
  [ -f m2_stealth.py ] || { echo "run this from the StealthDiffusion folder with m2_*.py copied in"; exit 1; }
  local miss=0
  for f in efficientnet-b0.pth resnet50.pth deit.pth swin-t.pth Controlvae.pt dncnn_color_blind.pth; do
    [ -f "checkpoints/$f" ] || { echo "MISSING checkpoints/$f"; miss=1; }
  done
  [ "$miss" = 0 ] || { echo "put the 6 checkpoints in ./checkpoints (README step 4)"; exit 1; }
  $PY -c "import torch; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available())"
}

data() {
  local real_args=(--out data --n_dev "$N_DEV" --n_test "$N_TEST")
  [ -n "$REAL_SRC" ] && real_args+=(--real_src "$REAL_SRC")
  $PY m2_data.py real "${real_args[@]}"
  $PY m2_data.py generate --out data --n_dev "$N_DEV" --n_test "$N_TEST" --model "$GEN_MODEL" --device "$DEVICE"
  $PY m2_data.py check --out data --device "$DEVICE"
}

attack() {
  local common=(--device "$DEVICE" --images_root data/gen/test --max_images "$N_ATTACK" --iterations 10
                --pretrained_diffusion_path "$SD_MODEL" --encoder_weights ./checkpoints/Controlvae.pt --seed 42 --resume)
  $PY m2_stealth.py "${common[@]}" --save_dir results/B00
  $PY m2_stealth.py "${common[@]}" --decode_path final --save_dir results/B10
  $PY m2_stealth.py "${common[@]}" --surrogates E,R,D --save_dir results/ENS
  $PY m2_stealth.py "${common[@]}" --surrogates E,R,D --decode_path final \
      --loss margin --lpips_weight 1.0 --di_prob 0.5 --save_dir results/FULL
}

evaluate() {
  $PY m2_evaluate.py --device "$DEVICE" \
    --runs B00=results/B00 B10=results/B10 ENS=results/ENS FULL=results/FULL \
    --real_dir data/real/test --out results/eval --conditions none,jpeg75,resize05
  echo "== results/eval/summary.csv =="; cat results/eval/summary.csv
}

challenge() {
  local n=$(( N_ATTACK < 20 ? N_ATTACK : 20 ))
  $PY m2_challenge.py build --real_dir data/real/test --gen_dir data/gen/test \
    --attack_dir results/FULL --n_per_group "$n" --out challenge
}

case "$stage" in
  check) check ;;
  data) check; data ;;
  attack) check; attack ;;
  eval) evaluate ;;
  challenge) challenge ;;
  all) check; data; attack; evaluate; challenge ;;
  *) echo "usage: bash run_local_mac.sh [check|data|attack|eval|challenge|all]"; exit 1 ;;
esac
echo "[run_local_mac] stage '$stage' done."
