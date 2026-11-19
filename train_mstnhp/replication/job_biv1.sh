#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
stage=${1:?expected generate or train}
rep=${SLURM_ARRAY_TASK_ID:?expected an array task ID}
export MPLBACKEND=Agg
if [[ -n "${BIV1_ENV:-}" ]]; then
    source "$BIV1_ENV"
elif type module >/dev/null 2>&1; then
    module load python
    if [[ "$stage" == train ]]; then module load PyTorch; fi
fi
echo "stage=$stage replicate=$rep commit=$(git rev-parse HEAD) python=$(command -v python)"
# Prevent accidental overlapping submissions for the same stage and replicate.
exec 9>"outputs/biv1/${stage}_${rep}.lock"
flock -n 9 || { echo "Already running: $stage $rep" >&2; exit 1; }
python -u train_mstnhp/replication/biv1.py "$stage" --replicate "$rep"
