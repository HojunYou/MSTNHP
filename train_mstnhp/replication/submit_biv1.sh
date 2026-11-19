#!/usr/bin/env bash
# Run via bash from any directory in the cluster checkout.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
cd "$repo"
tasks=${1:-1}
mkdir -p outputs/biv1
# Array indices match across both jobs. Slurm opens the log before the job runs.
gen=$(sbatch --parsable --array="$tasks%${SIM_JOBS:-8}" --job-name=biv1_sim \
    --output='outputs/biv1/%x_%A_%a.out' --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem="${SIM_MEM:-8G}" --time="${SIM_TIME:-120:00:00}" \
    "$here/job_biv1.sh" generate)
gen=${gen%%;*}
echo "Simulation array: $gen"
train=$(sbatch --parsable --array="$tasks%${TRAIN_JOBS:-2}" --dependency="aftercorr:$gen" \
    --kill-on-invalid-dep=yes --job-name=biv1_train --output='outputs/biv1/%x_%A_%a.out' \
    --nodes=1 --ntasks=1 --cpus-per-task="${TRAIN_CPUS:-10}" --gpus=1 \
    --mem="${TRAIN_MEM:-300G}" --time="${TRAIN_TIME:-120:00:00}" \
    "$here/job_biv1.sh" train)
echo "Training array: $train"
