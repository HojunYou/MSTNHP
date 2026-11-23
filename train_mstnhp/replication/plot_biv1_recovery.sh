#!/bin/bash
#SBATCH --job-name=biv1_plot
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
module load python
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
python train_mstnhp/replication/plot_biv1_recovery.py
