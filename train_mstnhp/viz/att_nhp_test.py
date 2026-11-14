
# Use the shared training directory for imports and relative paths.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
from easy_tpp.utils import load_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                    help='Dir of configuration yaml to train and evaluate the model.')

parser.add_argument('--experiment_id', type=str, required=False, default='AttNHP_gen',
                    help='Experiment id in the config file.')

parser.add_argument('--ckpt_dir', type=str, required=False,
                    default='./checkpoints/111377_22967005582336_240801-214731/models/saved_model',
                    help='Path to model')

parser.add_argument('--data_path', type=str, required=False, default='./data/exercise/test.pkl',
                    help='Path to dataset file')

parser.add_argument('--sample_index', type=int, required=False, default=21, help='Index of the sample to evaluate')

args = parser.parse_args()

# #generate non-trained model
config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)
model = model_runner.model_wrapper.model

#model.load_state_dict(torch.load(args.ckpt_dir), strict=False)
state_dict = torch.load(args.ckpt_dir, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)

# #generate dataset: test dataset
data = load_pickle(args.data_path)

# # A list containing 100 event sequences
source_datasets = data['test']

# # A random event event sequences from source_datasets
source_dataset = source_datasets[args.sample_index]
# #print(f"source_dataset: {source_dataset}")

# # Input processing
time_seqs = [event['time_since_start'] for event in source_dataset]
type_seqs = [event['type_event'] for event in source_dataset]
time_delta_seqs = [event['time_since_last_event'] for event in source_dataset]
#print(f"time delta seqs: {time_delta_seqs}")
#print(f"time  seqs: {time_seqs}")

# # Converting inputs to tensors
time_seqs_tensor = torch.tensor(time_seqs, dtype=torch.float32).unsqueeze(0)
time_delta_seqs_tensor = torch.tensor(time_delta_seqs, dtype=torch.float32).unsqueeze(0)
type_seqs_tensor = torch.tensor(type_seqs, dtype=torch.long).unsqueeze(0)

# # Mask for padding
seq_mask = torch.ones_like(type_seqs_tensor, dtype=torch.bool)
type_mask = torch.nn.functional.one_hot(type_seqs_tensor, num_classes=model.num_event_types_pad).float()

# # Batch processing
batch = (time_seqs_tensor, time_delta_seqs_tensor, type_seqs_tensor, seq_mask, None, type_mask)
#batch1 = (time_seqs_tensor, time_delta_seqs_tensor, type_seqs_tensor, None, None, None)

temp_time = model.make_dtime_loss_samples(time_delta_seqs_tensor[:, 1:])
sample_dtimes = temp_time + time_seqs_tensor[:, :-1].unsqueeze(-1)
print(f"Shape sample_dtimes:{sample_dtimes.shape}")

with torch.no_grad():
    intensities = model.compute_intensities_at_sample_times(time_seqs_tensor[:, :-1], time_delta_seqs_tensor[:, :-1], type_seqs_tensor[:, :-1], sample_dtimes)

print(f"Shape Intensity:{intensities.shape}")

#sample_dtimes = sample_dtimes + time_seqs_tensor[:, :-1, None]
sample_dtimes = sample_dtimes.numpy().reshape(1, -1)[0, :1500]
#print(sample_dtimes)

intensities = intensities.squeeze(0)
print(f"Shape Intensity:{intensities.shape}")
intensities = intensities.numpy().reshape(1, -1, 2)


plt.figure(figsize=(10, 6))
plt.plot(sample_dtimes, intensities[0, :1500, 0], color="red", linestyle='-', label=f'Type 0')
plt.plot(sample_dtimes, intensities[0, :1500, 1], color="blue", linestyle='-', label=f'Type 1')


filtered_events = [[], []]
for time, event_type in zip(time_seqs, type_seqs):
    if time <= sample_dtimes[-1]:
        filtered_events[event_type].append(time)

colors = ['red', 'blue']
line_length = 0.003
line_width = 2
line_offset = 0.1

for i in range(2):
    plt.eventplot(filtered_events[i], lineoffsets=line_offset, colors=colors[i], linelengths=line_length, linewidths=line_width)


plt.xlabel('Time')
plt.ylabel('Conditional Intensity')
#plt.ylim(0.5, None)
plt.legend()
plt.show()
