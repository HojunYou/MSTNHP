#!/bin/bash
#SBATCH -J bistpp_sim             # Job name
#SBATCH -o outputs/simulate_mstpp_%j.txt   # %j = job‑id
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task (no array)
#SBATCH -t 48:00:00
#SBATCH --mem=50GB
#SBATCH --begin=now

module load python
pip install --user arrow

SEED=2025

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/mstpp_bivariate_simulation.py" --seed "${SEED}"
