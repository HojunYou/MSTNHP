#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=pak_fig8
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
stage=train
source "${BIV1_ENV:-/project/jun/ccchukw2/envs/biv1-job-env.sh}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd train_mstnhp
for model in mstnhp tnhp; do
    case "$model" in
        mstnhp) run_dir=checkpoints/pakistan_rev/mstnhp/12534_23052896257856_261124-144405 ;;
        tnhp)   run_dir=checkpoints/pakistan_rev/tnhp/243500_22856777455424_261124-144403 ;;
    esac
    for year in 2014 2018; do
        for weights in best final; do
            python -u "viz/figure8_${model}.py" --year "$year" --run-dir "$run_dir" --weights "$weights"
        done
    done
done
