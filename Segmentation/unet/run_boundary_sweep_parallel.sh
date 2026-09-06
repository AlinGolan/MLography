#!/bin/bash
# Boundary-weight sweep, PARALLEL: train the U-Net 10x at once (all CPU on gpu002),
# focal loss + distanceTransform boundary loss, boundary weight from focal-favored
# to boundary-favored. Each process is thread-capped so 10 fit on 64 cores.
set -u

cd /home/alingo/MLography/Segmentation/unet || exit 1
PY=/home/alingo/.conda/envs/mlography/bin/python
EPOCHS=${EPOCHS:-50}
THREADS=${THREADS:-6}

RUNROOT=${RUNROOT:-boundary_sweep_par_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUNROOT/models" "$RUNROOT/logs"
SUMMARY="$RUNROOT/summary.txt"

# thread caps so 10 procs * THREADS ~= 64 cores
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export TF_NUM_INTRAOP_THREADS=$THREADS
export TF_NUM_INTEROP_THREADS=2
export KMP_BLOCKTIME=0

{
  echo "host        : $(hostname)"
  echo "epochs/run  : $EPOCHS"
  echo "threads/run : $THREADS"
  echo "run root    : $(pwd)/$RUNROOT"
  echo "started     : $(date)"
  echo
} | tee "$SUMMARY"

WEIGHTS=(0.0 0.1 0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0)

pids=()
i=0
for w in "${WEIGHTS[@]}"; do
  i=$((i + 1))
  tag=$(printf "%02d_bw%s" "$i" "$w")
  model="$RUNROOT/models/model_${tag}.hdf5"
  log="$RUNROOT/logs/run_${tag}.log"

  if [ "$w" = "0.0" ]; then
    bl_flag="--nouse_boundary_loss"
  else
    bl_flag="--use_boundary_loss"
  fi

  stdbuf -oL -eL "$PY" main.py \
      --state=train \
      $bl_flag \
      --boundary_weight="$w" \
      --model_name="$model" \
      --epochs="$EPOCHS" \
      --keep_training=True \
      > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "launched [$i/10] boundary_weight=$w  pid=$pid  -> $log" | tee -a "$SUMMARY"
  sleep 5   # stagger imagenet-weight load / graph build
done

printf '%s\n' "${pids[@]}" > "$RUNROOT/pids.txt"
echo | tee -a "$SUMMARY"
echo "all 10 launched, waiting... $(date)" | tee -a "$SUMMARY"

i=0
for w in "${WEIGHTS[@]}"; do
  i=$((i + 1))
  wait "${pids[$((i - 1))]}"
  echo "run $(printf '%02d' $i) bw=$w exit=$?  $(date)" | tee -a "$SUMMARY"
done

echo | tee -a "$SUMMARY"
echo "ALL DONE $(date)" | tee -a "$SUMMARY"
