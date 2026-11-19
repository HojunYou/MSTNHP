#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=biv1_eval
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/biv1/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
stage=train  # use the same PyTorch environment for CPU inference
source "${BIV1_ENV:?export the cluster job environment path}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -u train_mstnhp/replication/evaluate_biv1.py \
    --replicate "$SLURM_ARRAY_TASK_ID" --run "${EVAL_RUN:-v1}" --ns "${EVAL_NS:-40}"
