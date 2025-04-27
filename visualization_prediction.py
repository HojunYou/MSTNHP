import torch
import numpy as np
from easy_stpp.config_factory import Config
from easy_stpp.runner import STPPRunner as Runner
from easy_stpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_stpp.preprocess.dataset import get_data_loader

import matplotlib.pyplot as plt
import os, pickle


# with open('./data/gtd_pakistan/pakistan_polygon.pkl', 'rb') as f:
#     polygon = pickle.load(f)

## HJ: Define model and data configurations
# dataname = 'bistpp_separable_zhang'
dataname = 'gtd_pakistan'
year_list = list(range(2008, 2021))
config_suffix = dataname.split('_')[0]
model_name = 'STNHP' 
experiment_name = model_name+'_train'

config = Config.build_from_yaml_file('configs/experiment_config_'+dataname+'.yaml', experiment_id=experiment_name)
## HJ: Create a model runner from the configuration and train the model
model_runner = Runner.build_from_config(config)

## HJ: Load model from the checkpoint
n_comp = model_runner.model_wrapper.model.n_comps
n_epochs = 200
n_times = 50
model_name = f'{model_name}_{n_comp}_{n_times}_{n_epochs}'
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
dataset = STPPDataset(data) # dataset = TPPDataset(data)
sample_size = len(dataset)
tokenizer = STEventTokenizer(data_config.data_specs) # tokenizer = EventTokenizer(data_config.data_specs)
batch_size = 13 # 13

## HJ: Load train dataloader
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size = batch_size, shuffle=False, space=True)

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
# st_sample_location[0][28][6400:8000,0] = 26.60
input_ = time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, None, None, None
hiddens_ti, decays, output_states = model_runner.model_wrapper.model.forward(input_)

## HJ: Select data index for evaluation
data_idx = 7
# for data_idx in range(batch_size):
# year = year_list[data_idx]
interval_t_samples = [st_sample_list[data_idx][i][:,0,0]-time_seqs[data_idx,i] for i in range(len(st_sample_list[data_idx]))]
interval_s_samples = [torch.norm(st_sample_list[data_idx][i][:,:,1:], p=2, dim=-1) for i in range(len(st_sample_list[data_idx]))]

## HJ: Add minimum threshold to interval_t_samples to avoid peaks.
min_thres = .01
interval_t_samples_copy = interval_t_samples.copy()
for i, interval_t_sample in enumerate(interval_t_samples_copy):
    if not interval_t_sample.any():
        continue
    first_t_sample = interval_t_sample[0]
    if first_t_sample<=min_thres:
        interval_t_sample[:ns**2]+=min_thres
        interval_t_samples[i] = interval_t_sample
        print(f"Min threshold is added to {i} th interval_t_sample.")
    
## HJ: evaluate conditional intensity at interval_t_samples and interval_s_samples
state_t_samples = [model_runner.model_wrapper.model.compute_states_at_sample_points(decays[data_idx:(data_idx+1), i:(i+1)], output_states[data_idx:(data_idx+1), i:(i+1)], interval_t_sample, interval_s_sample)
                for i, (interval_t_sample, interval_s_sample) in enumerate(zip(interval_t_samples, interval_s_samples))]

## HJ: get intensity samples
lambda_t_samples = [model_runner.model_wrapper.model.layer_intensity(state_t_sample) for state_t_sample in state_t_samples]
lambda_t_samples = [lambda_t_sample.detach().cpu().numpy()[0,0] for lambda_t_sample in lambda_t_samples]
lambda_t_samples = np.concatenate(lambda_t_samples, axis=0)
lambda_t_samples.shape

sample_intensity = lambda_t_samples.copy()
st_sample = torch.cat(st_sample_location[data_idx], dim=0).cpu().numpy()


## HJ: Compare the number of fitted points and the number of actual points
fitted_points = sample_intensity.sum(axis=0)*time_domain/num_sample_t*(2/ns)**2
expected_points = time_seqs[data_idx].argmax().item()+1
actual_points = np.unique(type_seqs[data_idx, :expected_points].numpy(), return_counts=True)[1]
print(fitted_points, actual_points)

## HJ: Get quantiles of sample intensity to determine ranges for intensity plots.
np.quantile(sample_intensity, [.001, 0.01, 0.9, 0.99, 0.999], axis=0)
np.min(sample_intensity, axis=0), np.max(sample_intensity, axis=0)

## HJ: Load polygon
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

with open('./data/gtd_pakistan/pakistan_polygon.pkl', 'rb') as f:
    pakistan_polygon = pickle.load(f)
pakistan_gdf = gpd.GeoSeries([pakistan_polygon])

## Visualization
# cmap = plt.get_cmap('coolwarm')
cmap = plt.get_cmap('viridis')
ns = model_runner.model_wrapper.model.spatial_npoints
ns2 = ns**2
# ns2 = model_runner.model_wrapper.model.inside_polygon_index.shape[0]
df = pd.DataFrame({'t': st_sample[:,0], 'x': st_sample[:,1], 'y': st_sample[:,2]})
intensity_df = pd.DataFrame(sample_intensity, columns = [f'intensity{i}' for i in range(sample_intensity.shape[1])])
df = pd.concat([df, intensity_df], axis=1)
df_np = df.to_numpy()
savepath = f'../Meeting/2025/0101/{dataname}/{split}/{n_comp}_{n_epochs}'
if n_times !=50:
    savepath = f'{savepath}_{n_times}'
os.makedirs(savepath, exist_ok=True)
max_frame = len(df_np)//ns2

seq_t = time_seqs[data_idx].numpy()
seq_s = space_seqs[data_idx].numpy()
point_types = type_seqs[data_idx].numpy()

current_time = [seq_t[point_types==0][0], seq_t[point_types==1][0]]
next_time = [seq_t[point_types==0][1], seq_t[point_types==1][1]]
current_loc = [seq_s[point_types==0][0], seq_s[point_types==1][0]]
start_is = [int(np.where(df_np[:,0]>=current_time[0])[0][0]/ns2), int(np.where(df_np[:,0]>=current_time[1])[0][0]/ns2)]

type1_color = 'black'
type2_color = 'yellow'

for i in range(0, 2*max_frame//3):
    fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(11, 4)) # (10,4)
    xy_points = [Point(x, y) for x, y in zip(df_np[i*ns2:(i+1)*ns2,1], df_np[i*ns2:(i+1)*ns2,2])]
    gdf_points = gpd.GeoDataFrame(geometry=xy_points, crs="EPSG:4326")
    gdf_within = gdf_points.within(pakistan_gdf.iloc[0])
    pakistan_mask = gdf_within.values.reshape(df_np[i*ns2:(i+1)*ns2,1].shape)
    masked_intensity0 = np.ma.masked_where(~pakistan_mask, df_np[i*ns2:(i+1)*ns2, -2])
    masked_intensity1 = np.ma.masked_where(~pakistan_mask, df_np[i*ns2:(i+1)*ns2, -1])
    p0 = ax[0].pcolormesh(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), masked_intensity0.reshape(ns, -1), shading='auto', cmap=plt.cm.coolwarm, vmin=0.0, vmax = 3.0)
    p1 = ax[1].pcolormesh(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), masked_intensity1.reshape(ns, -1), shading='auto', cmap=plt.cm.coolwarm, vmin=0.0, vmax = 2.3)
    pakistan_gdf.boundary.plot(ax=ax[0], linewidth=1)
    pakistan_gdf.boundary.plot(ax=ax[1], linewidth=1)

    # p0=ax[0].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -2].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.05, vmax = 0.6)
    # p1=ax[1].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -1].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.05, vmax = 0.6)
    
    ## HJ: Update current time & loc and next time & loc for plotting actual events on intensity map
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
    
    ## HJ: Plot actual events on intensity map
    if i>=start_is[0]:
        ax[0].scatter(current_loc[0][0], current_loc[0][1], color=type1_color, s=20)
        ax[1].scatter(current_loc[0][0], current_loc[0][1], color=type1_color, s=20)
    if i>=start_is[1]:
        ax[0].scatter(current_loc[1][0], current_loc[1][1], color=type2_color, s=20)
        ax[1].scatter(current_loc[1][0], current_loc[1][1], color=type2_color, s=20)
    
    ## two separate colorbars for ax[0] and ax[1]
    plt.colorbar(p0, ax=ax[0], location='right', format='%.2f')
    plt.colorbar(p1, ax=ax[1], location='right', format='%.2f')
    # plt.colorbar(p1, ax = [ax[0], ax[1]], location='right', format='%.2f')
    plt.suptitle(f'time: {df_np[i*ns2, 0]:.2f}', fontsize=12)

    ## HJ: Rename plots for proper temporal order.
    if i<10:
        savename = f'000{i}'
    elif i<100:
        savename = f'00{i}'
    elif i<1000:
        savename = f'0{i}'
    else:
        savename = f'{i}'
    # plt.savefig(f'{savepath}/{data_idx}_{savename}.jpg', bbox_inches='tight')
    plt.savefig(f'{savepath}/{year_list[data_idx]}_{savename}.jpg', bbox_inches='tight')
    plt.close()


## HJ: Temporal conditional intensity
### Temporal component comparison
time_points = df.t.to_numpy().reshape(-1, ns2)
intensity_0 = df.iloc[:,3].to_numpy().reshape(-1, ns2)
intensity_1 = df.iloc[:,4].to_numpy().reshape(-1, ns2)

unit_area = 2**2/(ns2)

temporal_intensity_0 = intensity_0.sum(axis=1)*unit_area
temporal_intensity_1 = intensity_1.sum(axis=1)*unit_area

time_df = pd.DataFrame({'t': time_points[:,0], 'intensity_0': temporal_intensity_0, 'intensity_1': temporal_intensity_1})
time_df.describe()

min_value = .5
plt.figure(figsize=(15,6))
plt.scatter(seq_t[point_types==0], np.zeros_like(seq_t[point_types==0])+min_value, s=10)
plt.scatter(seq_t[point_types==1], np.zeros_like(seq_t[point_types==1])+min_value, s=10)
plt.plot(time_df.t, time_df.intensity_0, label='Type 0') #label='Taliban')
plt.plot(time_df.t, time_df.intensity_1, label = 'Type 1') # label='Baloch')
plt.ylim(min_value-0.1, 5.0)
plt.legend()
plt.savefig(f'{savepath}/../{year_list[data_idx]}_time_intensity_{n_times}_{n_epochs}.jpg', bbox_inches='tight')
# plt.savefig(f'{savepath}/../../{data_idx}_{n_comp}_{n_epochs}_time_intensity.jpg', bbox_inches='tight')
plt.close()







##### Prediction #####
def get_upperbound_stnhp(model, batch):
    time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch
    dtimespace_for_bound = model.make_multiple_dtimespace_loss_samples(time_delta_seqs[:, 1:], space_seqs[:, :-1])
    dtime_for_bound, dspace_for_bound = dtimespace_for_bound[...,0,0], dtimespace_for_bound[...,:,:,1:]
    dspace_for_bound = torch.norm(dspace_for_bound, dim=-1, p=2)
    intensities_for_bound = model.compute_intensities_at_sample_spacetimes(
        time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, dtime_for_bound, dspace_for_bound, compute_last_step_only=False
    )
    intensity_upperbound = intensities_for_bound.sum(dim=-1).max(dim=-1)[0]*model.event_sampler.over_sample_rate
    return intensity_upperbound.detach().cpu().numpy()

def generate_homogeneous_points(time_delta_seq, intensity_upperbound, seed_i=0):

    B, L = time_delta_seq.shape
    T=time_delta_seq+5
    T = T.tolist()
    S = [[-1, 1], [-1, 1]]
    npoints      = np.array([[4*t_i for t_i in T_i] for T_i in T ])
    poisson_lam = intensity_upperbound*npoints
    np.random.seed(seed_i)
    N      = np.random.poisson(lam=poisson_lam)
    # simulate spatial sequence and temporal sequence separately.
    homo_points = [[np.random.uniform([0, -1,-1], [T[n_i][i], 1, 1], (N[n_i][i], 3)) for i in range(L)] for n_i in range(B)]
    homo_points = [[point[point[:,0].argsort()] for point in points] for points in homo_points]
    return homo_points

def get_space_components(space_seq, homo_points_s):

    B, L, _ = space_seq.shape
    space_samples = [
        [homo_seq_s[:,None,:].repeat(1, 3, 1) for homo_seq_s in homo_point_s] for homo_point_s in homo_points_s
    ]
    # space_samples = space_samples.repeat(1, 1, 1, 3, 1)
    for i in range(B):
        for j in range(len(space_samples[i])):
            if j==0:
                space_samples[i][j][:,:2,:]=0
                space_samples[i][j][:,2,:] -= space_seq[i,j,:][None,:]
            elif j==1:
                space_samples[i][j][:,0,:] = 0
                space_samples[i][j][:,1:,:] -= space_seq[i,j-1:j+1,...][None,:,:]
            else:
                space_samples[i][j] -= space_seq[i,j-2:j+1, ...][None,:,:]

    space_samples = [
        [space_seq.norm(p=2, dim=-1) for space_seq in space_sample] for space_sample in space_samples
    ]

    return space_samples

def get_prediction(predicted_intensity, intensity_upperbound, homo_points_t, homo_points_s, type_seqs):

    predicted_intensity_sum = predicted_intensity.sum(-1)
    B, L, _ = predicted_intensity_sum.shape
    end_points = np.argmax(type_seqs[:,1:]==2, axis=1)
    zero_idx = np.where(end_points==0)[0]
    end_points[zero_idx] = L
    pred_type_seqs = np.zeros((B, L))
    pred_time_seqs = np.zeros((B, L))
    pred_space_seqs = np.zeros((B, L, 2))

    prob = np.random.rand(*predicted_intensity_sum.shape)
    retained_idx = np.where(predicted_intensity_sum > prob*intensity_upperbound[...,None])
    retained_idx = [idx.tolist() for idx in retained_idx]
    retained_idx = list(zip(*retained_idx))

    retained_idx_dict = {(k[0], k[1]):k[2] for k in retained_idx[::-1]}

    ## get an idx from retained_idx_dict and get values from time_samples with the corresponding index.
    ## then insert the value to pred_time_seqs
    for k, v in retained_idx_dict.items():
        pred_time = homo_points_t[k[0]][k[1]][v]
        pred_space = homo_points_s[k[0]][k[1]][v]
        pred_type = predicted_intensity[k[0], k[1], v].argmax().item()
        pred_type_seqs[k[0], k[1]] = pred_type
        pred_time_seqs[k[0], k[1]] = pred_time
        pred_space_seqs[k[0], k[1]] = pred_space

    all_retained = (pred_time_seqs!=0).all().item()
    if not all_retained:
        not_retained = np.array(np.where(pred_time_seqs==0)).transpose()
        for i in range(B):
            not_retained_i_cols = not_retained[not_retained[:,0]==i,1]
            if not not_retained_i_cols.any():
                continue
            if (not_retained_i_cols>=end_points[i]).all():
                continue
            for c in not_retained_i_cols:
                if c < end_points[i]:
                    pred_time_seqs[i,c] = homo_points_t[i][c][-1]
    # assert all_retained, 'There are some points not retained.'
    pad_idx = np.where(type_seqs[:,1:]==2.0)
    pad_idx = [idx.tolist() for idx in pad_idx]
    pad_idx = list(zip(*pad_idx))
    for pad_i in pad_idx:
        pred_time_seqs[pad_i] = 2.0
        pred_type_seqs[pad_i] = 2
        pred_space_seqs[pad_i] = 0

    total_len = len(type_seqs[:,1:].flatten())-len(pad_idx)

    return pred_time_seqs, pred_space_seqs, pred_type_seqs, total_len

def prediction_stnhp(model, batch, seed=0):
    
    time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, _, _, _ = batch
    time_seq, time_delta_seq, type_seq, space_seq, space_delta_seq = time_seqs[:, :-1], time_delta_seqs[:, 1:], type_seqs[:, :-1], space_seqs[:, :-1], space_delta_seqs[:, 1:]
    B, L = time_seq.size()
    nt, ns = model.loss_integral_num_sample_per_step, model.spatial_npoints
    model.loss_integral_num_sample_per_step = 5
    model.spatial_npoints = 5
    intensity_upperbound = get_upperbound_stnhp(model, batch)
    homo_points = generate_homogeneous_points(time_delta_seq, intensity_upperbound, seed_i=seed)
    homo_points_t = [
        [torch.Tensor(homo_seq[:,0]) for j, homo_seq in enumerate(homo_point)] for i, homo_point in enumerate(homo_points)
    ]
    homo_points_s = [
        [torch.Tensor(homo_seq[:,1:]) for homo_seq in homo_point] for homo_point in homo_points
    ]
    space_samples = get_space_components(space_seq, homo_points_s)
    max_len = max([len(homo_points_t[i][j]) for i in range(B) for j in range(L)])
    pred_sample_intensity = np.zeros((B,L,max_len,2))
    for i in range(B):
        for j in range(L):
            len_points = homo_points_t[i][j].shape[0]
            pred_sample_intensity[i,j,:len_points] = model.compute_intensities_at_sample_spacetimes(
                time_seqs[i][None,...], 
                time_delta_seqs[i][None,...], 
                space_seqs[i][None,...], 
                space_delta_seqs[i][None,...], 
                type_seqs[i][None,...], 
                homo_points_t[i][j][None,None,...], space_samples[i][j][None, None,...]
            )[0,j].detach().cpu().numpy()

    pred_time_delta_seqs, pred_space_seqs, pred_type_seqs, total_len =\
        get_prediction(pred_sample_intensity, intensity_upperbound, homo_points_t, homo_points_s, type_seqs.numpy())

    time_pred_diff = (time_delta_seqs[:,1:].numpy()-pred_time_delta_seqs)**2
    space_pred_diff = np.sum((space_seqs[:,1:].numpy()-pred_space_seqs)**2, -1)
    type_pred_diff = type_seqs[:,1:].numpy()!=pred_type_seqs

    return time_pred_diff, space_pred_diff, type_pred_diff, total_len


from easy_stpp.preprocess.dataset import get_data_loader

batch_size = 13
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size = batch_size, shuffle=False, space=True)
import time
from datetime import datetime
start_time = time.time()
## start_time to hour:min:seconds
print(f"Start time: {datetime.now()}")

time_errors = []
type_errors = []
space_time_errors = []
space_errors = []

n_rep = 10

for i in range(n_rep):
    total_len = 0
    time_pred_diff = 0
    total_time_pred_diff = []
    space_pred_diff = 0
    type_pred_diff = 0
    torch.manual_seed(i)
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.values()
        batch_time_pred_diff, batch_space_pred_diff, batch_type_pred_diff, batch_len = prediction_stnhp(model_runner.model_wrapper.model, batch, seed=i)
        time_pred_diff+=batch_time_pred_diff.sum()
        space_pred_diff+=batch_space_pred_diff.sum()
        type_pred_diff+=batch_type_pred_diff.sum()
        total_len+=batch_len
        print(f"Batch time prediction diff: {np.sqrt(batch_time_pred_diff.sum()/batch_len).item()}")
        print(f"Batch space prediction diff: {np.sqrt(batch_space_pred_diff.sum()/batch_len).item()}")
        print(f"Batch type prediction diff: {batch_type_pred_diff.sum()/batch_len}")
        print(f"Batch {batch_idx}/{len(train_loader)} done.")
        if np.sqrt(batch_time_pred_diff.sum()/batch_len).item()>20:
            break

    end_time = time.time()
    print(f"Time prediction diff: {np.sqrt(time_pred_diff/total_len).item()}")
    print(f"Space prediction diff: {np.sqrt(space_pred_diff/total_len).item()}")
    print(f"Type prediction diff: {type_pred_diff/total_len}")
    print(f"Space-time prediction diff: {np.sqrt((time_pred_diff+space_pred_diff).sum()/total_len).item()}")
    print(f"End time: {datetime.now()}")
    print(f"Time elapsed: {end_time-start_time}")

    type_errors.append(type_pred_diff/total_len)
    time_errors.append(np.sqrt(time_pred_diff/total_len).item())
    space_errors.append(np.sqrt(space_pred_diff/total_len).item())
    space_time_errors.append(np.sqrt((time_pred_diff+space_pred_diff).sum()/total_len).item())

type_errors = np.array(type_errors)
time_errors = np.array(time_errors)
space_time_errors = np.array(space_time_errors)
space_errors = np.array(space_errors)
print(f"Distribution of type errors: {np.mean(type_errors), np.std(type_errors)}")
print(f"Distribution of time errors: {np.mean(time_errors), np.std(time_errors)}")
print(f"Distribution of space errors: {np.mean(space_errors), np.std(space_errors)}")
print(f"Distribution of space-time errors: {np.mean(space_time_errors), np.std(space_time_errors)}")

pred_dict = {}
pred_dict['type_errors'] = type_errors
pred_dict['time_errors'] = time_errors
pred_dict['space_errors'] = space_errors
pred_dict['space_time_errors'] = space_time_errors



with open(f'./{model_name}_pred_dict.pkl', 'wb') as f:
    pickle.dump(pred_dict, f)

n_epochs = 200
model_name = model_name[:-3]+str(n_epochs)
with open(f'./{model_name}_pred_dict.pkl', 'rb') as f:
    pred_dict = pickle.load(f)

round(pred_dict['type_errors'].mean(), 4)
round(pred_dict['time_errors'].mean(), 4)
round(pred_dict['space_time_errors'].mean(), 4)

# time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch

# event_sampler = model_runner.model_wrapper.model.event_sampler

### MLE prediction
from stpp_simulator.mstppg import BivariateHawkesLam, BivariateSpatialTemporalPointProcess
from stpp_simulator.stppg import StdDiffusionKernel, SeparableExponentialKernel, NonseparableKernel
import itertools
import numpy as np

seed_value = 0
np.random.seed(seed_value)
np.set_printoptions(suppress=True)

def get_upperbound_mle(lam_mle, batch_list, num_t_for_bound =5, num_s_for_bound = 5):
    time_seqs, time_delta_seqs, space_seqs, _, type_seqs = batch_list
    dtimes_ratio_sampled = np.linspace(
        start=0.0,
        stop=1.0,
        num=num_t_for_bound,
    )[None, None, :]
    B, L = time_delta_seqs[:,1:].shape
    time_samples = time_delta_seqs[:, 1:, None] * dtimes_ratio_sampled
    space_grid = np.linspace(-1, 1, num_s_for_bound, endpoint=False)+1/num_s_for_bound
    end_points = np.argmax(type_seqs[:,1:]==2, axis=1)
    zero_idx = np.where(end_points==0)[0]
    end_points[zero_idx] = L
    lam_values = np.zeros((B,L, num_t_for_bound*num_s_for_bound**2, 2))
    for i in range(B):
        for j in range(end_points[i]):
            t = time_samples[i,j]
            st_grid = np.array(list(itertools.product(t, space_grid, space_grid)))
            t = st_grid[:,0]
            s = st_grid[:,1:]
            sub_seq_t = time_seqs[i,:j+1]
            sub_seq_s = space_seqs[i,:j+1]
            sub_seq_types = type_seqs[i,:j+1]
            t=t+sub_seq_t[-1]
            lam_value_seq = []
            for k in range(len(t)):
                sample_t = t[k]
                sample_s = s[k]
                while j>=1 and sample_t<=sub_seq_t[-1]:
                    sample_t+=0.01
                lam_value = np.array([
                    lam_mle.value(sample_t, sub_seq_t, sample_s, sub_seq_s, 0, sub_seq_types),
                    lam_mle.value(sample_t, sub_seq_t, sample_s, sub_seq_s, 1, sub_seq_types)
                ])
                lam_value_seq.append(lam_value)
            lam_values[i,j] = np.array(lam_value_seq)

    intensities_for_bound_mle = 10*lam_values
    intensity_upperbound_mle = intensities_for_bound_mle.sum(-1).max(-1)

    return intensity_upperbound_mle

# def generate_homogeneous_points_mle(time_delta_seq, intensity_upperbound_mle):

#     B, L = time_delta_seq.shape
#     T=time_delta_seq+5
#     T = T.tolist()
#     S = [[-1, 1], [-1, 1]]
#     # sample the number of events from S
#     npoints      = np.array([[4*t_i for t_i in T_i] for T_i in T ])
#     poisson_lam_mle = intensity_upperbound_mle*npoints
#     N_mle = np.random.poisson(lam=poisson_lam_mle)
#     # poisson_lam_mle[0]
#     # simulate spatial sequence and temporal sequence separately.
#     homo_points_mle = [[np.random.uniform([0, -1,-1], [T[n_i][i], 1, 1], (N_mle[n_i][i], 3)) for i in range(L)] for n_i in range(B)]
#     homo_points_mle = [[point[point[:,0].argsort()] for point in points] for points in homo_points_mle]
#     # homo_points_mle = torch.Tensor(homo_points_mle)

#     return homo_points_mle

def prediction_mle(lam_mle, batch_list, seed):
    time_seqs, time_delta_seqs, space_seqs, _, type_seqs = batch_list
    end_points = np.argmax(type_seqs[:,1:]==2, axis=1)
    B, L = time_seqs[:,1:].shape
    zero_idx = np.where(end_points==0)[0]
    end_points[zero_idx] = L
    intensity_upperbound_mle = get_upperbound_mle(lam_mle, batch_list)
    homo_points_mle = generate_homogeneous_points(time_delta_seqs[:,1:], intensity_upperbound_mle, seed_i=seed)
    homo_points_t_mle = [
        [homo_seq_mle[:,0]+time_seqs[i,j,None] for j, homo_seq_mle in enumerate(homo_point_mle)] for i, homo_point_mle in enumerate(homo_points_mle)
    ]
    homo_points_s_mle = [
        [homo_seq_mle[:,1:] for homo_seq_mle in homo_point_mle] for homo_point_mle in homo_points_mle
    ]
    max_len_points = max([len(homo_seq_t) for homo_point_t_mle in homo_points_t_mle for homo_seq_t in homo_point_t_mle])
    pred_sample_intensity_mle = np.zeros((B, L, max_len_points, 2))

    for i in range(B):
        for j in range(end_points[i]):
            t_mle = homo_points_t_mle[i][j]
            s_mle = homo_points_s_mle[i][j]
            sub_seq_t_mle = time_seqs[i,:j+1]
            sub_seq_s_mle = space_seqs[i,:j+1]
            sub_seq_type_mle = type_seqs[i,:j+1]
            pred_sample_intensity_seq = []
            for k in range(len(t_mle)):
                sample_t = t_mle[k]
                sample_s = s_mle[k]
                while j>=1 and sample_t<=sub_seq_t_mle[-1]:
                    sample_t+=0.01
                pred_sample_intensity_point = np.array([
                    lam_mle.value(sample_t, sub_seq_t_mle, sample_s, sub_seq_s_mle, 0, sub_seq_type_mle),
                    lam_mle.value(sample_t, sub_seq_t_mle, sample_s, sub_seq_s_mle, 1, sub_seq_type_mle)
                ])
                pred_sample_intensity_seq.append(pred_sample_intensity_point)
            pred_sample_intensity_mle[i,j,:len(t_mle)]=np.array(pred_sample_intensity_seq)

    pred_time_seqs_mle, pred_space_seqs_mle, pred_type_seqs_mle, total_len = \
        get_prediction(pred_sample_intensity_mle, intensity_upperbound_mle, homo_points_t_mle, homo_points_s_mle, type_seqs)

    time_pred_diff_mle = (time_seqs[:,1:]-pred_time_seqs_mle)**2
    space_pred_diff_mle = np.sum((space_seqs[:,1:]-pred_space_seqs_mle)**2, -1)
    type_pred_diff_mle = type_seqs[:,1:]!=pred_type_seqs_mle

    return time_pred_diff_mle, space_pred_diff_mle, type_pred_diff_mle, total_len



# parameters initialization
mu_mle = [0.111, 0.044]
A_mles = [1.217, -0.020, 1.722, 0.768]
beta_t_mles = [0.041, 0.629, 1.089, 0.123]
beta_s_mles = [13.323, 0.078, 0.016, 11.078]
C_mles = [round(A_mle*beta_t_mle*beta_s_mle, 4) for A_mle, beta_t_mle, beta_s_mle in zip(A_mles, beta_t_mles, beta_s_mles)]
C_mles

# mu_mle = [0.1104, 0.0995]
# # A_mles = [1.217, -0.020, 1.722, 0.768]
# beta_t_mles = [0.85280453, 0.11055560, 0.49395569, 0.35623536]
# beta_s_mles = [2.00465827, 1.75411099, 1.97511932, 1.82365556]
# # C_mles = [round(A_mle*beta_t_mle*beta_s_mle, 4) for A_mle, beta_t_mle, beta_s_mle in zip(A_mles, beta_t_mles, beta_s_mles)]
# C_mles = [0.24841706, 0.01721968, 0.19884980, 0.09055001]



kernels_mle = [SeparableExponentialKernel(C=C_mle, beta_t = beta_t_mle, beta_s=beta_s_mle) 
               for C_mle, beta_t_mle, beta_s_mle in zip(C_mles, beta_t_mles, beta_s_mles)]
lam_mle = BivariateHawkesLam(mu_mle, kernels_mle)

import time
from datetime import datetime
start_time = time.time()
## start_time to hour:min:seconds
print(f"Start time: {datetime.now()}")

time_errors_mle = []
type_errors_mle = []
space_time_errors_mle = []
space_errors_mle = []
for j in range(n_rep):
    time_pred_diff_mle = 0
    space_pred_diff_mle = 0
    type_pred_diff_mle = 0
    total_len =0
    for i, batch in enumerate(train_loader):
        batch = batch.values()
        time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch
        batch_list = [time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs]
        batch_list = list(map(lambda x: x.numpy(), batch_list))

        batch_time_pred_diff_mle, batch_space_pred_diff_mle, batch_type_pred_diff_mle, batch_len = prediction_mle(lam_mle, batch_list, seed=j+i)
        time_pred_diff_mle+=batch_time_pred_diff_mle.sum()
        space_pred_diff_mle+=batch_space_pred_diff_mle.sum()
        type_pred_diff_mle+=batch_type_pred_diff_mle.sum()
        total_len+=batch_len
        print(f"Batch time prediction diff: {np.sqrt(batch_time_pred_diff_mle.sum()/batch_len).item()}")
        print(f"Batch space prediction diff: {np.sqrt(batch_space_pred_diff_mle.sum()/batch_len).item()}")
        print(f"Batch type prediction diff: {batch_type_pred_diff_mle.sum()/batch_len}")
        print(f"Batch {i}/{len(train_loader)} done.")
        if np.sqrt(batch_time_pred_diff_mle.sum()/batch_len).item()>20:
            break
    
    type_errors_mle.append(type_pred_diff_mle/total_len)
    time_errors_mle.append(np.sqrt(time_pred_diff_mle/total_len).item())
    space_errors_mle.append(np.sqrt(space_pred_diff_mle/total_len).item())
    space_time_errors_mle.append(np.sqrt((time_pred_diff_mle+space_pred_diff_mle).sum()/total_len).item())
    
type_errors_mle = np.array(type_errors_mle)
time_errors_mle = np.array(time_errors_mle)
space_time_errors_mle = np.array(space_time_errors_mle)
space_errors_mle = np.array(space_errors_mle)
print(f"Distribution of type errors: {np.mean(type_errors_mle), np.std(type_errors_mle)}")
print(f"Distribution of time errors: {np.mean(time_errors_mle), np.std(time_errors_mle)}")
print(f"Distribution of space errors: {np.mean(space_errors_mle), np.std(space_errors_mle)}")
print(f"Distribution of space-time errors: {np.mean(space_time_errors_mle), np.std(space_time_errors_mle)}")

end_time = time.time()
print(f"End time: {datetime.now()}")
print(f"Time elapsed: {end_time-start_time}")


pred_dict_mle = {}
pred_dict_mle['type_errors'] = type_errors_mle
pred_dict_mle['time_errors'] = time_errors_mle
pred_dict_mle['space_errors'] = space_errors_mle
pred_dict_mle['space_time_errors'] = space_time_errors_mle

with open(f'./{model_name}_pred_dict_mle.pkl', 'wb') as f:
    pickle.dump(pred_dict_mle, f)
