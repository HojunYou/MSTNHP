#!/bin/bash
#SBATCH -J zhang
#SBATCH -o outputs/simulate_mstpp_zhang_%A_%a.txt
#SBATCH -N 1 -n 20
#SBATCH -t 48:00:00
#SBATCH --mem=50GB
#SBATCH --begin=now
#SBATCH -a 0-19

module load python
pip install arrow

seed=($(seq 0 19)) # 700 - alpha softplus

python simulate_mstpp_parallel.py -s ${seed[$SLURM_ARRAY_TASK_ID]}
