
# Resolve project imports and legacy paths before loading dependencies.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import argparse, os, sys, time, logging, pickle, numpy as np, shapely.geometry as sg
from shapely.geometry import Point, Polygon, MultiPolygon
import torch

# ───────────────────────────────────────── CLI ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml')
parser.add_argument('--experiment_id', default='STNHP_eval')

parser.add_argument('--polygon',  default='./data/gtd_pakistan/pakistan_polygon_norm.pkl')
parser.add_argument('--device',   default='cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# ─────────────────────────── easy_tpp initialisation (unchanged) ───────────────────────
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

config       = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)


data_cfg   = model_runner.runner_config.data_config
backend    = model_runner.runner_config.base_config.backend
split      = 'test'
data_dir   = data_cfg.get_data_dir(split)
loader     = STPPDataLoader(data_cfg, backend)
data       = loader.build_input_from_pkl(data_dir, split)
dataset    = STPPDataset(data)
tokenizer  = STEventTokenizer(data_cfg.data_specs)
train_loader = get_data_loader(dataset, backend, tokenizer,
                               batch_size=1, shuffle=False, space=True)

with open(args.polygon, 'rb') as f:
    polygon = pickle.load(f)

area = polygon.area if isinstance(polygon, sg.Polygon)           \
       else sum(p.area for p in polygon.geoms)


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
    T = time_delta_seq+5
    T = T.tolist()
    S = [[-1, 1], [-1, 1]]
    npoints = np.array([[4*t_i for t_i in T_i] for T_i in T])
    poisson_lam = intensity_upperbound*npoints
    np.random.seed(seed_i)
    N = np.random.poisson(lam=poisson_lam)
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



### MLE prediction
from stpp_simulator.mstppg import BivariateHawkesLam, BivariateSpatialTemporalPointProcess
from stpp_simulator.stppg import StdDiffusionKernel, SeparableExponentialKernel, NonseparableKernel
import itertools
import numpy as np

seed_value = 0
np.random.seed(seed_value)
np.set_printoptions(suppress=True)


def get_upperbound_mle(lam_mle, batch_list, num_t_for_bound=5, num_s_for_bound=5):
    time_seqs, time_delta_seqs, space_seqs, _, type_seqs = batch_list
    dtimes_ratio_sampled = np.linspace(
        start=0.0,
        stop=1.0,
        num=num_t_for_bound,
    )[None, None, :]
    B, L = time_delta_seqs[:, 1:].shape
    time_samples = time_delta_seqs[:, 1:, None] * dtimes_ratio_sampled
    space_grid = np.linspace(-1, 1, num_s_for_bound, endpoint=False) + 1 / num_s_for_bound
    end_points = np.argmax(type_seqs[:, 1:] == 2, axis=1)
    zero_idx = np.where(end_points == 0)[0]
    end_points[zero_idx] = L
    lam_values = np.zeros((B, L, num_t_for_bound * num_s_for_bound ** 2, 2))
    for i in range(B):
        for j in range(end_points[i]):
            t = time_samples[i, j]
            st_grid = np.array(list(itertools.product(t, space_grid, space_grid)))
            t = st_grid[:, 0]
            s = st_grid[:, 1:]
            sub_seq_t = time_seqs[i, :j + 1]
            sub_seq_s = space_seqs[i, :j + 1]
            sub_seq_types = type_seqs[i, :j + 1]
            t = t + sub_seq_t[-1]
            lam_value_seq = []
            for k in range(len(t)):
                sample_t = t[k]
                sample_s = s[k]
                while j >= 1 and sample_t <= sub_seq_t[-1]:
                    sample_t += 0.01
                lam_value = np.array([
                    lam_mle.value(sample_t, sub_seq_t, sample_s, sub_seq_s, 0, sub_seq_types),
                    lam_mle.value(sample_t, sub_seq_t, sample_s, sub_seq_s, 1, sub_seq_types)
                ])
                lam_value_seq.append(lam_value)
            lam_values[i, j] = np.array(lam_value_seq)

    intensities_for_bound_mle = 10 * lam_values
    intensity_upperbound_mle = intensities_for_bound_mle.sum(-1).max(-1)

    return intensity_upperbound_mle


# def generate_homogeneous_points_mle(time_delta_seq, intensity_upperbound_mle):
#
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
#
#     return homo_points_mle




def prediction_mle(lam_mle, batch_list, seed):
    time_seqs, time_delta_seqs, space_seqs, _, type_seqs = batch_list
    end_points = np.argmax(type_seqs[:, 1:] == 2, axis=1)
    B, L = time_seqs[:, 1:].shape
    zero_idx = np.where(end_points == 0)[0]
    end_points[zero_idx] = L
    intensity_upperbound_mle = get_upperbound_mle(lam_mle, batch_list)
    homo_points_mle = generate_homogeneous_points(time_delta_seqs[:, 1:], intensity_upperbound_mle, seed_i=seed)
    homo_points_t_mle = [
        [homo_seq_mle[:, 0] + time_seqs[i, j, None] for j, homo_seq_mle in enumerate(homo_point_mle)] for
        i, homo_point_mle in enumerate(homo_points_mle)
    ]
    homo_points_s_mle = [
        [homo_seq_mle[:, 1:] for homo_seq_mle in homo_point_mle] for homo_point_mle in homo_points_mle
    ]
    max_len_points = max([len(homo_seq_t) for homo_point_t_mle in homo_points_t_mle for homo_seq_t in homo_point_t_mle])
    pred_sample_intensity_mle = np.zeros((B, L, max_len_points, 2))

    for i in range(B):
        for j in range(end_points[i]):
            t_mle = homo_points_t_mle[i][j]
            s_mle = homo_points_s_mle[i][j]
            sub_seq_t_mle = time_seqs[i, :j + 1]
            sub_seq_s_mle = space_seqs[i, :j + 1]
            sub_seq_type_mle = type_seqs[i, :j + 1]
            pred_sample_intensity_seq = []
            for k in range(len(t_mle)):
                sample_t = t_mle[k]
                sample_s = s_mle[k]
                while j >= 1 and sample_t <= sub_seq_t_mle[-1]:
                    sample_t += 0.01
                pred_sample_intensity_point = np.array([
                    lam_mle.value(sample_t, sub_seq_t_mle, sample_s, sub_seq_s_mle, 0, sub_seq_type_mle),
                    lam_mle.value(sample_t, sub_seq_t_mle, sample_s, sub_seq_s_mle, 1, sub_seq_type_mle)
                ])
                pred_sample_intensity_seq.append(pred_sample_intensity_point)
            pred_sample_intensity_mle[i, j, :len(t_mle)] = np.array(pred_sample_intensity_seq)

    pred_time_seqs_mle, pred_space_seqs_mle, pred_type_seqs_mle, total_len = \
        get_prediction(pred_sample_intensity_mle, intensity_upperbound_mle, homo_points_t_mle, homo_points_s_mle,
                       type_seqs)

    time_pred_diff_mle = (time_seqs[:, 1:] - pred_time_seqs_mle) ** 2
    space_pred_diff_mle = np.sum((space_seqs[:, 1:] - pred_space_seqs_mle) ** 2, -1)
    type_pred_diff_mle = type_seqs[:, 1:] != pred_type_seqs_mle

    return time_pred_diff_mle, space_pred_diff_mle, type_pred_diff_mle, total_len


# parameters initialization
mu_mle = [1.244, 1.032]
A_mles = [1.793, -0.015, 0.185, 1.203]
beta_t_mles = [0.137,  0.121, 1.800, 0.265]
beta_s_mles = [8.768,  1.751, 0.030, 9.004]
C_mles = [round(A_mle * beta_t_mle * beta_s_mle, 4) for A_mle, beta_t_mle, beta_s_mle in
          zip(A_mles, beta_t_mles, beta_s_mles)]


kernels_mle = [SeparableExponentialKernel(C=C_mle, beta_t=beta_t_mle, beta_s=beta_s_mle)
               for C_mle, beta_t_mle, beta_s_mle in zip(C_mles, beta_t_mles, beta_s_mles)]
lam_mle = BivariateHawkesLam(mu_mle, kernels_mle)

import time
from datetime import datetime

n_rep = 2

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
    total_len = 0
    for i, batch in enumerate(train_loader):
        batch = batch.values()
        time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch
        batch_list = [time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs]
        batch_list = list(map(lambda x: x.numpy(), batch_list))

        batch_time_pred_diff_mle, batch_space_pred_diff_mle, batch_type_pred_diff_mle, batch_len = prediction_mle(
            lam_mle, batch_list, seed=j + i)
        time_pred_diff_mle += batch_time_pred_diff_mle.sum()
        space_pred_diff_mle += batch_space_pred_diff_mle.sum()
        type_pred_diff_mle += batch_type_pred_diff_mle.sum()
        total_len += batch_len
        print(f"Batch time prediction diff: {np.sqrt(batch_time_pred_diff_mle.sum() / batch_len).item()}")
        print(f"Batch space prediction diff: {np.sqrt(batch_space_pred_diff_mle.sum() / batch_len).item()}")
        print(f"Batch type prediction diff: {batch_type_pred_diff_mle.sum() / batch_len}")
        print(f"Batch {i}/{len(train_loader)} done.")
        if np.sqrt(batch_time_pred_diff_mle.sum() / batch_len).item() > 20:
            break

    type_errors_mle.append(type_pred_diff_mle / total_len)
    time_errors_mle.append(np.sqrt(time_pred_diff_mle / total_len).item())
    space_errors_mle.append(np.sqrt(space_pred_diff_mle / total_len).item())
    space_time_errors_mle.append(np.sqrt((time_pred_diff_mle + space_pred_diff_mle).sum() / total_len).item())

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
print(f"Time elapsed: {end_time - start_time}")
