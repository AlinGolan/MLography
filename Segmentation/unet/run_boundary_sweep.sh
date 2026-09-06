#!/bin/bash
# Boundary-weight sweep: train the U-Net 10x with focal loss + distanceTransform
# boundary loss, varying the boundary weight from "favor focal" to "favor boundary".
set -u

cd /home/alingo/MLography/Segmentation/unet || exit 1
PY=/home/alingo/.conda/envs/mlography/bin/python
EPOCHS=${EPOCHS:-50}

RUNROOT=${RUNROOT:-boundary_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUNROOT/models" "$RUNROOT/logs"
SUMMARY="$RUNROOT/summary.txt"

{
  echo "host        : $(hostname)"
  echo "python      : $PY"
  echo "epochs/run  : $EPOCHS"
  echo "run root    : $(pwd)/$RUNROOT"
  echo "started     : $(date)"
  echo "GPU visible : $($PY -c 'import tensorflow as tf; print(tf.test.is_gpu_available())' 2>/dev/null)"
  echo
} | tee "$SUMMARY"

# boundary_weight values: 0.0 = pure focal ... 5.0 = boundary strongly favored
WEIGHTS=(0.0 0.1 0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0)

i=0
for w in "${WEIGHTS[@]}"; do
  i=$((i + 1))
  tag=$(printf "%02d_bw%s" "$i" "$w")
  model="$RUNROOT/models/model_${tag}.hdf5"
  log="$RUNROOT/logs/run_${tag}.log"

  if [ "$w" = "0.0" ]; then
    bl_flag="--nouse_boundary_loss"     # pure focal-loss baseline
  else
    bl_flag="--use_boundary_loss"
  fi

  echo "=== [$i/10] boundary_weight=$w -> $model ===" | tee -a "$SUMMARY"
  start=$(date +%s)
  stdbuf -oL -eL "$PY" main.py \
      --state=train \
      $bl_flag \
      --boundary_weight="$w" \
      --model_name="$model" \
      --epochs="$EPOCHS" \
      --keep_training=True \
      > "$log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "    exit=$rc  elapsed=$(( end - start ))s  finished=$(date)" | tee -a "$SUMMARY"
done

echo | tee -a "$SUMMARY"
echo "ALL DONE $(date)" | tee -a "$SUMMARY"
