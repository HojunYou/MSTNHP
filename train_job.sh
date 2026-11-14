#!/bin/bash
#SBATCH --job-name=train_nhp
#SBATCH --output=outputs/stpp_bra_bla_new_1000_epochs_%j.txt
#SBATCH -N 1 -n 10
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --gpus 1
#SBATCH --begin=now


module load python/3.10
module load PyTorch
pip install --user --quiet shapely
unset PYTHONNOUSERSITEm
#pip install --user shapely
#module load PyTorch
#module load pyyaml
#module load torch
#module load matplotlib
#module load tensorboard
#module load numpy

#pip install --user tensorboard

# Activate the virtual environment
#source easytpp_env/bin/activate

# Navigate to the directory containing the script
cd train_mstnhp
export PYTHONPATH=$PYTHONPATH:/project/jun/ccchukw2/MSTNHP
# Run the script
python train_mstnhp.py
