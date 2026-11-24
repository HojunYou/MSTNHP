#!/bin/bash
#SBATCH --job-name=biv1_centres
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
stage=train
source "${BIV1_ENV:-/project/jun/ccchukw2/envs/biv1-job-env.sh}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -u train_mstnhp/replication/biv1_centres.py --replicate "${1:-1}"
