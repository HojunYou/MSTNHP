# -----------------------------------------------------------------------------
# Script to train STNHP Hawkes process model using MSTNHP library.
# Usage: python train_mstpp_hawkes.py --lr 0.001 -n 100 --seed 42

import torch
import numpy as np
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments for learning rate, number of epochs, and random seed
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default='learning rate')
parser.add_argument('-n', '--n_epochs', type=int, default='number of epochs')
parser.add_argument('-s', '--seed', type=int, default=0)
args = parser.parse_args()

# Display parsed hyperparameters for verification
print(args)

# Define dataset name and experimental year range
dataname = 'gtd_pakistan'
year_list = list(range(2008, 2021))
config_suffix = dataname.split('_')[0]
model_name = 'STNHP' 
experiment_name = model_name+'_train'

# Load experiment configuration from YAML file
config = Config.build_from_config_yaml_file('configs/experiment_config_'+dataname+'.yaml', experiment_id=experiment_name)

# Override default training parameters with CLI inputs
config.trainer_config.max_epoch = args.n_epochs
config.trainer_config.learning_rate = args.lr
config.trainer_config.seed = args.seed

# Initialize model runner and execute training loop
model_runner = Runner.build_from_config(config)
model_runner.run()

# Output model directory where checkpoints and logs are saved
print(model_runner.get_model_dir())