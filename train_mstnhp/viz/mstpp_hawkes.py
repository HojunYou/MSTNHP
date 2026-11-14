
# Use the shared training directory for imports and relative paths.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import torch
import numpy as np
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

import matplotlib.pyplot as plt
import os, pickle

## HJ: Define model and data configurations
dataname = 'bistpp_nonseparable_short' # dataname = 'gtd_pakistan'
# year_list = list(range(2008, 2020))
config_suffix = dataname.split('_')[0]
model_name = 'STNHP'
experiment_name = model_name+'_train'

# config = Config.build_from_yaml_file('easytpp/examples/configs/experiment_config_'+dataname+'.yaml', experiment_id=experiment_name)
## HJ: Create a configuration object from the yaml file
config = Config.build_from_yaml_file('configs/experiment_config_'+dataname+'.yaml', experiment_id=experiment_name)

## HJ: Create model runner from the configuration and train the model
model_runner = Runner.build_from_config(config)
model_runner.run()

## HJ: Load model from the checkpoint
n_comp = model_runner.model_wrapper.model.n_comps
n_epochs = 100
model_name = f'{model_name}_{n_comp}_{n_epochs}'
model_runner.model_wrapper.model.load_state_dict(torch.load('checkpoints/'+dataname+'/'+model_name, map_location='cpu'))
model_runner.model_wrapper.model.eval()

## HJ: get data and model configurations
data_config = model_runner.runner_config.data_config
model_backend = model_runner.runner_config.base_config.backend

## HJ: Load train dataset
split = 'train'
data_dir = data_config.get_data_dir(split)
data_source_type = data_dir.split('.')[-1]
data_loader = STPPDataLoader(data_config, model_backend) # data_loader = TPPDataLoader(data_config, model_backend)
data = data_loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)  # dataset = TPPDataset(data)
sample_size = len(dataset)
tokenizer = STEventTokenizer(data_config.data_specs) # tokenizer = EventTokenizer(data_config.data_specs)
batch_size = 5

## HJ: Load train dataloader
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size=batch_size, shuffle=False, space=True)

## HJ: get batch data
batch = next(iter(train_loader))
batch = batch.values()
time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch

## HJ: evaluate conditional intensity at equally-spaced space-time grid points
ns = 40
num_sample_t = 1000
time_domain = 100
model_runner.model_wrapper.model.spatial_npoints = ns
st_sample_list = model_runner.model_wrapper.model.make_multiple_equal_dtimespace_samples(time_seqs, [0,time_domain], space_seqs, num_sample_t = num_sample_t)
st_sample_location = model_runner.model_wrapper.model.make_equal_dtimespace_samples(time_seqs, [0,time_domain], torch.zeros_like(space_seqs), num_sample_t = num_sample_t)

input_ = time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, None, None, None
hiddens_ti, decays, output_states = model_runner.model_wrapper.model.forward(input_)

## HJ: Select data index for evaluation
data_idx = 0
# year = year_list[data_idx]
interval_t_samples = [st_sample_list[data_idx][i][:, 0, 0]-time_seqs[data_idx, i] for i in range(len(st_sample_list[data_idx]))]
interval_s_samples = [torch.norm(st_sample_list[data_idx][i][:, :, 1:], p=2, dim=-1) for i in range(len(st_sample_list[data_idx]))]

## HJ: evaluate conditional intensity at interval_t_samples and interval_s_samples
state_t_samples = [model_runner.model_wrapper.model.compute_states_at_sample_points(decays[data_idx:(data_idx+1), i:(i+1)], output_states[data_idx:(data_idx+1), i:(i+1)], interval_t_sample, interval_s_sample)
                   for i, (interval_t_sample, interval_s_sample) in enumerate(zip(interval_t_samples, interval_s_samples))]

## HJ: get intensity samples
lambda_t_samples = [model_runner.model_wrapper.model.layer_intensity(state_t_sample) for state_t_sample in state_t_samples]
lambda_t_samples = [lambda_t_sample.detach().cpu().numpy()[0, 0] for lambda_t_sample in lambda_t_samples]
lambda_t_samples = np.concatenate(lambda_t_samples, axis=0)
lambda_t_samples.shape

sample_intensity = lambda_t_samples.copy()
st_sample = torch.cat(st_sample_location[data_idx], dim=0).cpu().numpy()

## HJ: Compare the number of fitted points and the number of actual points
fitted_points = sample_intensity.sum(axis=0)*time_domain/num_sample_t*(2/ns)**2
expected_points = time_seqs[data_idx].argmax().item()+1
actual_points = np.unique(type_seqs[data_idx, :expected_points].numpy(), return_counts=True)[1]
(fitted_points, actual_points)

np.quantile(sample_intensity, [.001, 0.01, 0.9, 0.99, 0.999], axis=0)
np.min(sample_intensity, axis=0), np.max(sample_intensity, axis=0)

import pandas as pd

## Visualization
cmap = plt.get_cmap('coolwarm')
ns = model_runner.model_wrapper.model.spatial_npoints
ns2 = ns**2
df = pd.DataFrame({'t': st_sample[:,0], 'x': st_sample[:,1], 'y': st_sample[:,2]})
intensity_df = pd.DataFrame(sample_intensity, columns = [f'intensity{i}' for i in range(sample_intensity.shape[1])])
df = pd.concat([df, intensity_df], axis=1)
# df = df[df['t']> 2]
df_np = df.to_numpy()
savepath = f'../Meeting/2024/240808/{dataname}/{split}/{n_comp}_{n_epochs}'
os.makedirs(savepath, exist_ok=True)
max_frame = len(df_np)//ns2


seq_t = time_seqs[data_idx].numpy()
seq_s = space_seqs[data_idx].numpy()
point_types = type_seqs[data_idx].numpy()

current_time = [seq_t[point_types==0][0], seq_t[point_types==1][0]]
next_time = [seq_t[point_types==0][1], seq_t[point_types==1][1]]
current_loc = [seq_s[point_types==0][0], seq_s[point_types==1][0]]
start_is = [int(np.where(df_np[:,0]>=current_time[0])[0][0]/ns2), int(np.where(df_np[:,0]>=current_time[1])[0][0]/ns2)]

for i in range(0, 2*max_frame//3):
    fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(10, 4)) # (11,4)
    p0=ax[0].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -2].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.02, vmax = .14)
    p1=ax[1].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -1].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.02, vmax = .14)

    if df_np[i*ns2,0] > next_time[0]:
        temp_time = next_time[0]
        current_time[0] = temp_time
        current_idx = np.where(seq_t==current_time[0])[0][0]
        next_time[0] = seq_t[point_types==0][np.where(seq_t[point_types==0]>temp_time)[0][0]]
        current_loc[0] = seq_s[current_idx]
    if df_np[i*ns2,0] > next_time[1]:
        temp_time = next_time[1]
        current_time[1] = temp_time
        current_idx = np.where(seq_t==current_time[1])[0][0]
        next_time[1] = seq_t[point_types==1][np.where(seq_t[point_types==1]>temp_time)[0][0]]
        current_loc[1] = seq_s[current_idx]

    if i>=start_is[0]:
        ax[0].scatter(current_loc[0][0], current_loc[0][1], c='c', s=20)
        ax[1].scatter(current_loc[0][0], current_loc[0][1], c='c', s=20)
    if i>=start_is[1]:
        ax[0].scatter(current_loc[1][0], current_loc[1][1], c='m', s=20)
        ax[1].scatter(current_loc[1][0], current_loc[1][1], c='m', s=20)
    plt.colorbar(p1, ax=[ax[0], ax[1]], location='right', format='%.2f')
    plt.suptitle(f'time: {df_np[i*ns2, 0]:.2f}', fontsize=12)
    # plt.show()
    if i<10:
        savename = f'000{i}'
    elif i<100:
        savename = f'00{i}'
    elif i<1000:
        savename = f'0{i}'
    else:
        savename = f'{i}'
    plt.savefig(f'{savepath}/{data_idx}_{savename}.jpg', bbox_inches='tight')
    plt.close()



### Temporal component comparison

time_points = df.t.to_numpy().reshape(-1, ns2)
intensity_0 = df.iloc[:, 3].to_numpy().reshape(-1, ns2)
intensity_1 = df.iloc[:, 4].to_numpy().reshape(-1, ns2)

unit_area = 2**2/(ns2)

temporal_intensity_0 = intensity_0.sum(axis=1)*unit_area
temporal_intensity_1 = intensity_1.sum(axis=1)*unit_area

# true_intensity= np.load('stpp_simulator/bistpp_third_intensity0.npy')
# true_intensity_0 = true_intensity[:,0].reshape(-1, 50**2).sum(axis=1)*(2**2/50**2)
# true_intensity_1 = true_intensity[:,1].reshape(-1, 50**2).sum(axis=1)*(2**2/50**2)
# true_intensity_0 = true_intensity_0[:988]
# true_intensity_1 = true_intensity_1[:988]

time_df = pd.DataFrame({'t': time_points[:, 0], 'intensity_0': temporal_intensity_0, 'intensity_1': temporal_intensity_1})
time_df.describe()
# true_df = pd.DataFrame({'t': time_points[:,0], 'intensity_0': true_intensity_0, 'intensity_1': true_intensity_1})
# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))

# ax[0].plot(time_df.t, time_df.intensity_0, label='fitted')
# ax[0].plot(time_df.t, true_df.intensity_0, 'r-', label='true')
# ax[0].set_title('Type 0')
# ax[0].legend()
# ax[0].set_ylim([0., 1.4])  # Set the y-axis range of the plot
# ax[1].plot(time_df.t, time_df.intensity_1, label='fitted')
# ax[1].plot(time_df.t, true_df.intensity_1, 'r-', label='true')
# ax[1].set_title('Type 1')
# ax[1].legend()
# ax[1].set_ylim([.37, 1.4])  # Set the y-axis range of the plot
# plt.show()
plt.figure(figsize=(15, 6))
plt.scatter(seq_t[point_types == 0], np.zeros_like(seq_t[point_types == 0]), s=10)
plt.scatter(seq_t[point_types == 1], np.zeros_like(seq_t[point_types == 1]), s=10)
plt.plot(time_df.t, time_df.intensity_0, label='Type 0')  # label='Taliban')
plt.plot(time_df.t, time_df.intensity_1, label='Type 1')  # label='Baloch')
plt.legend()
# plt.show()
# plt.savefig(f'{savepath}/../{year}_temporal.jpg', bbox_inches='tight')
plt.savefig(f'{savepath}/../../{data_idx}_{n_comp}_{n_epochs}_time_intensity.jpg', bbox_inches='tight')
plt.close()
