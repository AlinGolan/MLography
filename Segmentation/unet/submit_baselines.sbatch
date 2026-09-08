#!/bin/bash
#SBATCH --job-name=mlo_baseline
#SBATCH --partition=cluster
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/alingo/MLography/Segmentation/unet
#SBATCH --output=paper_baseline_slurm/slurm_%A_%a.out
#
# 4 focal-loss baselines (impurities/gb x 150-fixed/early-stop) on the 90% splits,
# one Slurm array task each, on CPU nodes of the 'cluster' partition, in parallel
# with the gpu002 boundary sweep.  focal loss = code default binary_focal_loss(alpha=0.2).
set -u

PY=/home/alingo/.conda/envs/mlography/bin/python
RUNROOT=paper_baseline_slurm
mkdir -p "$RUNROOT/models" "$RUNROOT/logs"

T=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T \
       NUMEXPR_NUM_THREADS=$T TF_NUM_INTRAOP_THREADS=$T TF_NUM_INTEROP_THREADS=2 \
       KMP_BLOCKTIME=0

case "${SLURM_ARRAY_TASK_ID:-x}" in
  0) name=impurities_150;   task=impurities; tdir=data/small_split/train;       extra="" ;;
  1) name=impurities_early; task=impurities; tdir=data/small_split/train;       extra="--early_stopping" ;;
  2) name=gb_150;           task=gb;         tdir=data/squares_128_split/train; extra="" ;;
  3) name=gb_early;         task=gb;         tdir=data/squares_128_split/train; extra="--early_stopping" ;;
  *) echo "bad/unset SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

model="$RUNROOT/models/${name}.hdf5"
log="$RUNROOT/logs/${name}.log"

echo "[$(date)] START $name  host=$(hostname)  cpus=$T  -> $model"
start=$(date +%s)
stdbuf -oL -eL "$PY" main.py \
    --state=train \
    --task="$task" \
    --train_dir="$tdir" \
    --epochs=150 \
    $extra \
    --keep_training=True \
    --model_name="$model" \
    > "$log" 2>&1
rc=$?
echo "[$(date)] END $name  rc=$rc  elapsed=$(( $(date +%s) - start ))s"
exit $rc
