# MSTNHP
This repo stands for Multivariate spatio-temporal Neural Hawkes Processes

## stpp-simulator
This repo is the simulator of spatio-temporal point processes
- simulate_mstpp_parallel.py: simulate multivariate spatio-temporal point processes in parallel.
- aggregate_dataset.py: aggregate the simulated point processes from simulate_mstpp_parallel.py
- simulate_mstpp.sh: sbatch script for simulate_mstpp_parallel.py

## train_mstnhp (previously a folder named "examples")
This repo is the trainer of spatio-temporal point processes

- configs: a directory where configuration files are stored
- train_mstpp_hawkes.py: train STNHP Hawkes process model

## easy_tpp
This repo contains updated library for training spatio-temporal point processes.

- torch_wrapper.py: torch wrapper for spatio-temporal point processes
- model/torch_model/torch_stnhp.py: torch model for spatio-temporal neural Hawkes processes.
- preprocess: contains updated library for handling spatio-temporal point process dataset.
- runner: contains updated library for training spatio-temporal point processes.

## visualization_prediction.py

- Visualize intensity maps from trained models and evaluate prediction results from MSTNHP and MLE.

## process_realdata.py

- Process real data and save it as pickle files. (GTD dataset preprocessing starts from L297.)

## easy_tpp.zip

- Contains the whole updated library for training spatio-temporal point processes from easy_tpp.

## Quickstart

```bash
python train_mstnhp/train_mstpp_hawkes.py
```

### Configuration

- train_mstnhp/configs/experiment_config_gtd_pakistan.yaml