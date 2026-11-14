
# Use the shared training directory for imports and relative paths.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import os, pickle
import matplotlib as mpl
from matplotlib import colors

from matplotlib.collections import LineCollection
import matplotlib.transforms as mtransforms

from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

dataname = 'bistpp_bivariate_3'

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config_stnhp.yaml',
                    help='Dir of configuration yaml to train and evaluate the model.')

parser.add_argument('--experiment_id', type=str, required=False, default='STNHP_eval',
                    help='Experiment id in the config file.')

parser.add_argument('--ckpt_dir', type=str, required=False, default='./checkpoints/131227_22617460385280_251015-131052/models/saved_model',
                        help='Path to model')

parser.add_argument('--data_path', type=str, required=False, default='./data/bistpp_bivariate_3_1/train.pkl',
                        help='Path to dataset file')

parser.add_argument('--sample_index', type=int, required=False, default=0, help='Index of the sample to evaluate')

args = parser.parse_args()


# #generate non-trained model
config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)

n_comp = model_runner.model_wrapper.model.n_comps
n_epochs = 1000


#saved_model_dir = model_runner.runner_config.base_config.specs['saved_model_dir']
# print(f"saved_model_dir: {saved_model_dir}")

state_dict = torch.load(args.ckpt_dir, map_location=torch.device('cpu'))
model_runner.model_wrapper.model.load_state_dict(state_dict, strict=False)
model_runner.model_wrapper.model.eval()


## HJ: get data and model configurations
data_config = model_runner.runner_config.data_config
#print(data_config)
model_backend = model_runner.runner_config.base_config.backend
#print(model_backend)


## HJ: Load train dataset
split = 'train'
data_dir = data_config.get_data_dir(split)
data_source_type = data_dir.split('.')[-1]
data_loader = STPPDataLoader(data_config, model_backend) # data_loader = TPPDataLoader(data_config, model_backend)
data = data_loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)  # dataset = TPPDataset(data)
sample_size = len(dataset)
tokenizer = STEventTokenizer(data_config.data_specs) # tokenizer = EventTokenizer(data_config.data_specs)
batch_size = 1

## HJ: Load train dataloader
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size=batch_size, shuffle=False, space=True)

## HJ: get batch data
batch = next(iter(train_loader))
batch = batch.values()

time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch
print(f"time_seq :{time_seqs.shape}")
print(f"space_seqs :{space_seqs.shape}")
print(f"type_seqs :{type_seqs.shape}")

## HJ: evaluate conditional intensity at equally-spaced space-time grid points
ns = 40
num_sample_t = 1000
time_domain = 100
model_runner.model_wrapper.model.spatial_npoints = ns
st_sample_list = model_runner.model_wrapper.model.make_multiple_equal_dtimespace_samples(time_seqs, [0, time_domain], space_seqs, num_sample_t=num_sample_t)
st_sample_location = model_runner.model_wrapper.model.make_equal_dtimespace_samples(time_seqs, [0, time_domain], torch.zeros_like(space_seqs), num_sample_t=num_sample_t)

print(f"st_sample_list :{len(st_sample_list)}")
print(f"st_sample_location :{st_sample_location[0][0].shape}")

input_ = time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, None, None, None
hiddens_ti, decays, output_states = model_runner.model_wrapper.model.forward(input_)

print(f"hiddens_ti :{hiddens_ti.shape}")
print(f"decays :{decays.shape}")
print(f"output_states :{output_states.shape}")

## HJ: Select data index for evaluation
data_idx = args.sample_index
# year = year_list[data_idx]
interval_t_samples = [st_sample_list[data_idx][i][:, 0, 0]-time_seqs[data_idx, i] for i in range(len(st_sample_list[data_idx]))]
interval_s_samples = [torch.norm(st_sample_list[data_idx][i][:, :, 1:], p=2, dim=-1) for i in range(len(st_sample_list[data_idx]))]
print(f"interval_t_samples length :{len(interval_t_samples)}")
print(f"interval_s_samples length :{len(interval_s_samples)}")

## HJ: evaluate conditional intensity at interval_t_samples and interval_s_samples
state_t_samples = [model_runner.model_wrapper.model.compute_states_at_sample_points(decays[data_idx:(data_idx+1), i:(i+1)], output_states[data_idx:(data_idx+1), i:(i+1)], interval_t_sample, interval_s_sample)
                   for i, (interval_t_sample, interval_s_sample) in enumerate(zip(interval_t_samples, interval_s_samples))]

print(f"state_t_samples length :{len(state_t_samples)}")


## HJ: get intensity samples
lambda_t_samples = [model_runner.model_wrapper.model.layer_intensity(state_t_sample) for state_t_sample in state_t_samples]
lambda_t_samples = [lambda_t_sample.detach().cpu().numpy()[0, 0] for lambda_t_sample in lambda_t_samples]
lambda_t_samples = np.concatenate(lambda_t_samples, axis=0)
print(f"lambda_t_samples: {lambda_t_samples.shape}")

sample_intensity = lambda_t_samples.copy()
st_sample = torch.cat(st_sample_location[data_idx], dim=0).cpu().numpy()


## HJ: Compare the number of fitted points and the number of actual points
fitted_points = sample_intensity.sum(axis=0)*time_domain/num_sample_t*(2/ns)**2
expected_points = time_seqs[data_idx].argmax().item()+1
actual_points = np.unique(type_seqs[data_idx, :expected_points].numpy(), return_counts=True)[1]
print(f"points: {(fitted_points, actual_points)}")

np.quantile(sample_intensity, [.001, 0.01, 0.9, 0.99, 0.999], axis=0)
print(f"quantiles: {np.quantile(sample_intensity, [.001, 0.01, 0.9, 0.99, 0.999], axis=0)}")
np.min(sample_intensity, axis=0), np.max(sample_intensity, axis=0)
print(f"min, max : {np.min(sample_intensity, axis=0), np.max(sample_intensity, axis=0)}")


import pandas as pd

## Visualization
#cmap = plt.get_cmap('coolwarm')
ns = model_runner.model_wrapper.model.spatial_npoints
ns2 = ns**2
print(f"ns2: {ns2}")
df = pd.DataFrame({'t': st_sample[:, 0], 'x': st_sample[:, 1], 'y': st_sample[:, 2]})
intensity_df = pd.DataFrame(sample_intensity, columns=[f'intensity{i}' for i in range(sample_intensity.shape[1])])
df = pd.concat([df, intensity_df], axis=1)
# df = df[df['t']> 2]
df_np = df.to_numpy()
print(f"len of df_np: {len(df_np)}")
savepath = f'../Meeting/2025/251208/{dataname}/{split}/{n_comp}_{n_epochs}'
os.makedirs(savepath, exist_ok=True)
max_frame = len(df_np)//ns2
#print(f"df_np_len: {df_np}, ns2: {ns2}, max_frame :{max_frame}")


print("Min/Max Type0:", df['intensity0'].min(), df['intensity0'].max())
print("Min/Max Type1:", df['intensity1'].min(), df['intensity1'].max())


seq_t = time_seqs[data_idx].numpy()
seq_s = space_seqs[data_idx].numpy()
point_types = type_seqs[data_idx].numpy()

current_time = [seq_t[point_types == 0][0], seq_t[point_types == 1][0]]
next_time = [seq_t[point_types == 0][1], seq_t[point_types == 1][1]]
current_loc = [seq_s[point_types == 0][0], seq_s[point_types == 1][0]]
start_is = [int(np.where(df_np[:, 0] >= current_time[0])[0][0] / ns2),
            int(np.where(df_np[:, 0] >= current_time[1])[0][0] / ns2)]


# frame‑wise spatial‑intensity

# ------------------------------------------------------------------
# frame-wise spatial-intensity movie (MATCHED STYLE VERSION)
# ------------------------------------------------------------------

print("start building frames")

# Spatial map
# spatial_types = (0, 1)
# type1, type2 = spatial_types
#
# type1_color = 'red'
# type2_color = 'red'
#
# # choose shapes
# type1_marker = 'o'   # 'o' = tpp, "X" = bra
# type2_marker = '^'   # "D" = bla, '^' = blf
#
# # style
# t1_style = dict(color=type1_color, marker=type1_marker, s=30,
#                 edgecolors='white', linewidths=0.6, zorder=4)
# t2_style = dict(color=type2_color, marker=type2_marker, s=30,
#                 edgecolors='white', linewidths=0.6, zorder=4)
#
# # --- (optional) filter out padding/zeros if present ---
# # You can keep this as-is even if there is no padding.
# valid_mask = seq_t > 0
# seq_t_valid = seq_t[valid_mask]
# seq_s_valid = seq_s[valid_mask]
# point_types_valid = point_types[valid_mask]
#
#
# # trackers
# def init_tracker(k):
#     t_k = seq_t_valid[point_types_valid == k]
#     s_k = seq_s_valid[point_types_valid == k]
#
#     if t_k.size > 0:
#         cur_t = float(t_k[0])
#         nxt_t = float(t_k[1]) if t_k.size > 1 else np.inf
#         cur_loc = s_k[0]
#
#         # first frame index where grid time >= cur_t
#         # df_np time for each frame sits at df_np[frame*ns2, 0]
#         frame_times = df_np[:, 0][::ns2]
#         start_i = int(np.searchsorted(frame_times, cur_t, side='left'))
#     else:
#         cur_t = nxt_t = np.inf
#         cur_loc = None
#         start_i = max_frame + 1
#
#     return {
#         "times": t_k,
#         "cur_t": cur_t,
#         "nxt_t": nxt_t,
#         "cur_loc": cur_loc,
#         "start_i": start_i,
#         "idx_ptr": 0
#     }
#
#
# trkA = init_tracker(type1)
# trkB = init_tracker(type2)
#
# # columns for intensities in df_np:
# # df_np columns are [t, x, y, intensity0, intensity1]
# colA = 3 + type1
# colB = 3 + type2
#
#
# frame_factor = 1.0  # set <1.0 to render fewer frames
# n_frames_to_draw = int(max_frame * max(0.0, min(1.0, frame_factor)))
#
# # colors choice
# CMAP_NAME = "viridis"   # "cividis", "magma", "YlOrRd", "Greys", "viridis"
# USE_LOG = False         # True -> log color scale
# GAMMA = 0.7             # <1 boosts low values; >1 compresses lows
#
# cmap = mpl.colormaps[CMAP_NAME]

print("Begin saving frames")


def save_spatial_frames_fitted(df_np, ns, ns2,
                               type_idx,
                               vmin, vmax,
                               save_root,
                               frame_factor=1.0,
                               CMAP_NAME="viridis",
                               USE_LOG=False,
                               GAMMA=0.7,
                               dpi=200):
    """
    Save fitted spatial intensity frames for a single type (type_idx in {0,1}).

    - IMAGE: per-frame vmin/vmax for good contrast
    - COLORBAR: fixed global vmin/vmax (from TRUE run) so that
      true vs fitted share the same bar for that type.

    df_np columns: [t, x, y, intensity0, intensity1]
    """

    vmin_global = float(vmin)
    vmax_global = float(vmax)

    type_label = f"type{type_idx + 1}"  # type1 for index 0, type2 for index 1

    save_root = Path(save_root)
    #save_root = f'../Meeting/2025/251208/{dataname}'
    save_dir = save_root / type_label
    save_dir.mkdir(parents=True, exist_ok=True)

    max_frame = int(df_np.shape[0] // ns2)
    frame_factor = float(max(0.0, min(1.0, frame_factor)))
    n_frames_to_draw = int(max_frame * frame_factor)

    cmap = mpl.colormaps[CMAP_NAME]

    # global norm: used ONLY for the colorbar
    if USE_LOG:
        norm_global = colors.LogNorm(vmin=max(vmin_global, 1e-6),
                                     vmax=max(vmax_global, 1e-6))
    else:
        # linear mapping on the bar so ticks are equally spaced in value
        norm_global = colors.Normalize(vmin=vmin_global, vmax=vmax_global)

    # intensity column for this type
    col = 3 + type_idx

    print(f"[FITTED] {type_label}: global z-scale vmin={vmin_global:.6g}, vmax={vmax_global:.6g}")
    print(f"[FITTED] saving {n_frames_to_draw} frames to {save_dir}")

    # ScalarMappable for the global colorbar
    sm = mpl.cm.ScalarMappable(norm=norm_global, cmap=cmap)
    sm.set_array([])

    # equally spaced ticks on the global bar
    n_ticks = 5
    ticks = np.linspace(vmin_global, vmax_global, n_ticks)

    for i in range(n_frames_to_draw):
        start, stop = i * ns2, (i + 1) * ns2
        if stop > df_np.shape[0]:
            break

        x_block = df_np[start:stop, 1].reshape(ns, ns)
        y_block = df_np[start:stop, 2].reshape(ns, ns)
        frame = df_np[start:stop, col].reshape(ns, ns)

        # per-frame limits for the IMAGE
        vmin_frame = float(frame.min())
        vmax_frame = float(frame.max())
        if vmax_frame <= vmin_frame:
            vmax_frame = vmin_frame + 1e-12

        if USE_LOG:
            norm_frame = colors.LogNorm(vmin=max(vmin_frame, 1e-6),
                                        vmax=max(vmax_frame, 1e-6))
        else:
            norm_frame = colors.PowerNorm(gamma=GAMMA,
                                          vmin=vmin_frame, vmax=vmax_frame)

        t_cur = float(df_np[start, 0])
        t_label_int = int(round(t_cur))  # integer time in the title

        fig, ax = plt.subplots(figsize=(6, 5))

        # IMAGE uses per-frame norm (high contrast)
        p = ax.pcolormesh(x_block, y_block, frame,
                          shading='auto', cmap=cmap, norm=norm_frame)

        ax.set_title(f"{type_label}  t={t_label_int}", fontsize=20)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        # COLORBAR uses global norm + shared ticks
        cbar = plt.colorbar(sm, ax=ax, location='right', format='%.3f')
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=18)

        savename = f"{i:04d}.jpg"
        plt.savefig(save_dir / savename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    print(f"[FITTED] finished saving frames for {type_label} → {save_dir}")


print("finished building frames")


# # global scales from the true run
# #type1
# vmin0_true = 0.1   # example
# vmax0_true = 0.91216   # example
#
# #type2
# vmin1_true = 0.1   # example
# vmax1_true = 0.892937   # example

# base directory for fitted frames
# save_root_fitted = f'../Meeting/2025/260111/{dataname}'
#
# frame_factor = 1.0  # or <1.0 for fewer frames

# type 1 (index 0)
# save_spatial_frames_fitted(
#     df_np, ns, ns2,
#     type_idx=0,
#     vmin=vmin0_true, vmax=vmax0_true,
#     save_root=save_root_fitted,
#     frame_factor=frame_factor,
#     CMAP_NAME="viridis",
#     USE_LOG=False,
#     GAMMA=0.7,
#     dpi=200
# )

# type 2 (index 1)
# save_spatial_frames_fitted(
#     df_np, ns, ns2,
#     type_idx=1,
#     vmin=vmin1_true, vmax=vmax1_true,
#     save_root=save_root_fitted,
#     frame_factor=frame_factor,
#     CMAP_NAME="viridis",
#     USE_LOG=False,
#     GAMMA=0.7,
#     dpi=200
# )

######################################################################

print("begin saving averaged frames")

def save_cumulative_timeavg_maps_fitted(df_np, ns, ns2,
                                        type_idx,
                                        vmin, vmax,
                                        n_slices=10,
                                        save_root="figs/spatial_timeavg_fitted",
                                        cmap_name="viridis",
                                        dpi=300,
                                        label_size=20,
                                        tick_size=18):
    """
    FITTED cumulative time-averaged spatial intensity maps for a single type.

    For type_idx ∈ {0,1}, we:
      - reconstruct λ̂_k(t_i, s_j) from df_np,
      - build cumulative time-averages up to a sequence of times τ_1, ..., τ_{n_slices},
      - save one figure per τ_m (so n_slices figures),
      - use a *global* vmin/vmax passed in from the TRUE run,
        so TRUE vs FITTED share the same color scale for this type.

    df_np columns: [t, x, y, intensity0, intensity1]
    """

    # infer nt and time grid from df_np
    nt = int(df_np.shape[0] // ns2)
    assert df_np.shape[0] == nt * ns2, "df_np shape inconsistent with ns and ns2"

    # time grid: one time per frame (every ns2 rows)
    tgrid = df_np[0::ns2, 0]  # shape (nt,)

    # extract intensities for this type as (nt, ns2)
    col = 3 + type_idx
    int_flat = df_np[:, col].reshape(nt, ns2)

    # slice times (cumulative endpoints), consistent with TRUE convention
    t_max = float(tgrid[-1])
    slice_times = np.linspace(t_max / n_slices, t_max, n_slices)

    # determine indices into tgrid for each slice time
    slice_indices = [int(np.searchsorted(tgrid, tau, side='right') - 1)
                     for tau in slice_times]
    slice_indices = [max(0, min(nt - 1, idx)) for idx in slice_indices]

    # compute cumulative averages for each slice (for plotting)
    cum_maps = []  # list of (idx, tau, map_ns_ns)
    for idx, tau in zip(slice_indices, slice_times):
        cum_mean_flat = int_flat[:idx+1, :].mean(axis=0)  # shape (ns2,)
        cum_map = cum_mean_flat.reshape(ns, ns)
        cum_maps.append((idx, tau, cum_map))

    # use vmin/vmax from TRUE cumulative run
    vmin_global = float(vmin)
    vmax_global = float(vmax)
    if vmax_global <= vmin_global:
        vmax_global = vmin_global + 1e-12

    type_label = f"type{type_idx + 1}"

    print(f"[FITTED cum-avg] {type_label}: using TRUE global vmin={vmin_global:.6g}, vmax={vmax_global:.6g}")
    print(f"[FITTED cum-avg] slice times (approx): {', '.join(f'{tau:.1f}' for tau in slice_times)}")

    # global norm for both image and colorbar (consistent across slices)
    cmap = mpl.colormaps[cmap_name]
    norm_global = colors.Normalize(vmin=vmin_global, vmax=vmax_global)

    # grid coordinates from first frame
    x = df_np[0:ns2, 1].reshape(ns, ns)
    y = df_np[0:ns2, 2].reshape(ns, ns)

    # output directory
    save_root = Path(save_root)
    save_dir = save_root / type_label
    save_dir.mkdir(parents=True, exist_ok=True)

    # ticks on bar
    n_ticks = 5
    ticks = np.linspace(vmin_global, vmax_global, n_ticks)

    print(f"[FITTED cum-avg] saving {len(cum_maps)} frames → {save_dir}")

    for m, (idx, tau, cum_map) in enumerate(cum_maps):
        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.pcolormesh(x, y, cum_map,
                           shading='auto', cmap=cmap, norm=norm_global)

        tau_int = int(round(tau))
        ax.set_title(f"{type_label}  t={tau_int}", fontsize=label_size)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        cbar = plt.colorbar(im, ax=ax, location='right', format="%.3f")
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=tick_size)

        savename = f"{m:02d}.jpg"   # 00, 01, ..., 09 for n_slices=10
        plt.savefig(save_dir / savename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"[FITTED cum-avg] finished cumulative maps for {type_label} → {save_dir}")



# global scales from the true run
#type1
vmin0_true = 0.100743   # example
vmax0_true = 0.131663   # example

#type2
vmin1_true = 0.100912   # example
vmax1_true = 0.126615   # example


# FITTED cumulative time-averaged spatial maps (10 slices per type)

save_root_cum_fitted = f'../Meeting/2026/260111/{dataname}_cum_timeavg'

# type 1 (index 0)
save_cumulative_timeavg_maps_fitted(
    df_np, ns, ns2,
    type_idx=0,
    vmin=vmin0_true, vmax=vmax0_true,
    n_slices=10,
    save_root=save_root_cum_fitted,
    cmap_name="viridis",
    dpi=300,
    label_size=20,
    tick_size=18
)

# type 2 (index 1)
save_cumulative_timeavg_maps_fitted(
    df_np, ns, ns2,
    type_idx=1,
    vmin=vmin1_true, vmax=vmax1_true,
    n_slices=10,
    save_root=save_root_cum_fitted,
    cmap_name="viridis",
    dpi=300,
    label_size=20,
    tick_size=18
)


print("finished saving averaged frames")

##########################################################################


print("old plot")

# for i in range(n_frames_to_draw):
#     start, stop = i * ns2, (i + 1) * ns2
#     if stop > len(df_np):
#         break
#
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
#
#     x_block = df_np[start:stop, 1].reshape(ns, ns)
#     y_block = df_np[start:stop, 2].reshape(ns, ns)
#
#     frameA = df_np[start:stop, colA].reshape(ns, ns)
#     frameB = df_np[start:stop, colB].reshape(ns, ns)
#
#     # per-frame limits
#     vminA, vmaxA = float(np.min(frameA)), float(np.max(frameA))
#     vminB, vmaxB = float(np.min(frameB)), float(np.max(frameB))
#
#     # Avoid degenerate norms if a frame is constant
#     if vmaxA <= vminA:
#         vmaxA = vminA + 1e-12
#     if vmaxB <= vminB:
#         vmaxB = vminB + 1e-12
#
#     if USE_LOG:
#         normA = colors.LogNorm(vmin=max(vminA, 1e-6), vmax=max(vmaxA, 1e-6))
#         normB = colors.LogNorm(vmin=max(vminB, 1e-6), vmax=max(vmaxB, 1e-6))
#     else:
#         normA = colors.PowerNorm(gamma=GAMMA, vmin=vminA, vmax=vmaxA)
#         normB = colors.PowerNorm(gamma=GAMMA, vmin=vminB, vmax=vmaxB)
#
#     p0 = ax[0].pcolormesh(x_block, y_block, frameA, shading='auto', cmap=cmap, norm=normA)
#     p1 = ax[1].pcolormesh(x_block, y_block, frameB, shading='auto', cmap=cmap, norm=normB)
#
#     # current model time
#     t_cur = float(df_np[start, 0])
#
#     # --- update trackers and draw markers ---
#     # Type 1 tracker
#     if t_cur > trkA["nxt_t"]:
#         trkA["cur_t"] = trkA["nxt_t"]
#         trkA["idx_ptr"] += 1
#
#         if trkA["idx_ptr"] < trkA["times"].size:
#             # update next time
#             if trkA["idx_ptr"] + 1 < trkA["times"].size:
#                 trkA["nxt_t"] = float(trkA["times"][trkA["idx_ptr"] + 1])
#             else:
#                 trkA["nxt_t"] = np.inf
#
#             # update location based on exact time match
#             cur_idx = np.where(seq_t_valid == trkA["cur_t"])[0]
#             if cur_idx.size > 0:
#                 trkA["cur_loc"] = seq_s_valid[cur_idx[0]]
#         else:
#             trkA["nxt_t"] = np.inf
#
#     # Type 2 tracker
#     if t_cur > trkB["nxt_t"]:
#         trkB["cur_t"] = trkB["nxt_t"]
#         trkB["idx_ptr"] += 1
#
#         if trkB["idx_ptr"] < trkB["times"].size:
#             if trkB["idx_ptr"] + 1 < trkB["times"].size:
#                 trkB["nxt_t"] = float(trkB["times"][trkB["idx_ptr"] + 1])
#             else:
#                 trkB["nxt_t"] = np.inf
#
#             cur_idx = np.where(seq_t_valid == trkB["cur_t"])[0]
#             if cur_idx.size > 0:
#                 trkB["cur_loc"] = seq_s_valid[cur_idx[0]]
#         else:
#             trkB["nxt_t"] = np.inf
#
#     # draw latest-event markers once we reach the frame where that type begins
#     if i >= trkA["start_i"] and trkA["cur_loc"] is not None:
#         ax[0].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], **t1_style)
#         ax[1].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], **t1_style)
#
#     if i >= trkB["start_i"] and trkB["cur_loc"] is not None:
#         ax[0].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], **t2_style)
#         ax[1].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], **t2_style)
#
#     # titles: dimensionless time
#     # ax[0].set_title(f"type 1  t={t_cur:.1f}")
#     # ax[1].set_title(f"type 2  t={t_cur:.1f}")
#
#     ax[0].set_title(f"type1  t={t_cur:.2f}", fontsize=20)
#     ax[1].set_title(f"type2  t={t_cur:.2f}", fontsize=20)
#
#     # clean axes
#     for a in ax:
#         a.set_xticks([])
#         a.set_yticks([])
#         a.tick_params(axis='both', which='both', length=0)
#         for s in a.spines.values():
#             s.set_visible(False)
#
#     # colorbars
#     cbar0 = plt.colorbar(p0, ax=ax[0], shrink=.85, format="%.3f")
#     cbar1 = plt.colorbar(p1, ax=ax[1], shrink=.85, format="%.3f")
#     cbar0.ax.tick_params(labelsize=18)
#     cbar1.ax.tick_params(labelsize=18)
#
#     savename = f"{i:04d}"
#     plt.savefig(f"{savepath}/{data_idx}_{savename}.jpg", bbox_inches="tight")
#     plt.close(fig)

print("old plot")

# for i in range(0, 2 * max_frame ):
#
#     # --- pick out the ns×ns block of λ₀ and λ₁ for this frame -----------
#     block0 = df_np[i*ns2:(i+1)*ns2, -2].reshape(ns, ns)   # intensity 0
#     block1 = df_np[i*ns2:(i+1)*ns2, -1].reshape(ns, ns)   # intensity 1
#
#     # --- per‑frame colour limits  --------------------------------------
#     # if you want a tiny cushion so a single extreme pixel isn't all yellow:
#     # eps = 1e‑10
#     vmin = min(block0.min(), block1.min())
#     vmax = max(block0.max(), block1.max())
#     # (use separate vmin/vmax per type if you prefer: vmin0,vmax0 …)
#
#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#
#     p0 = ax[0].pcolor(df_np[i*ns2:(i+1)*ns2, 1].reshape(ns, ns),
#                       df_np[i*ns2:(i+1)*ns2, 2].reshape(ns, ns),
#                       block0, cmap="viridis",
#                       vmin=vmin, vmax=vmax)
#
#     p1 = ax[1].pcolor(df_np[i*ns2:(i+1)*ns2, 1].reshape(ns, ns),
#                       df_np[i*ns2:(i+1)*ns2, 2].reshape(ns, ns),
#                       block1, cmap="viridis",
#                       vmin=vmin, vmax=vmax)
#
#     # -------- colour‑bars (one bar shared or one per axis) -------------
#     plt.colorbar(p0, ax=ax[0], shrink=.85, format="%.3f")   # left plot
#     plt.colorbar(p1, ax=ax[1], shrink=.85, format="%.3f")   # right plot
#
#     # ------------------------------------------------------------------
#     # (unchanged) update moving event marker, titles, save jpg …
#     # ------------------------------------------------------------------
#     if df_np[i*ns2, 0] > next_time[0]:
#         temp_time = next_time[0]
#         current_time[0] = temp_time
#         current_idx = np.where(seq_t == current_time[0])[0][0]
#         next_time[0] = seq_t[point_types == 0][np.where(seq_t[point_types == 0] > temp_time)[0][0]]
#         current_loc[0] = seq_s[current_idx]
#     if df_np[i*ns2, 0] > next_time[1]:
#         temp_time = next_time[1]
#         current_time[1] = temp_time
#         current_idx = np.where(seq_t == current_time[1])[0][0]
#         next_time[1] = seq_t[point_types == 1][np.where(seq_t[point_types == 1] > temp_time)[0][0]]
#         current_loc[1] = seq_s[current_idx]
#
#     if i >= start_is[0]:
#         ax[0].scatter(*current_loc[0], c='c', s=20)
#         ax[1].scatter(*current_loc[0], c='c', s=20)
#     if i >= start_is[1]:
#         ax[0].scatter(*current_loc[1], c='m', s=20)
#         ax[1].scatter(*current_loc[1], c='m', s=20)
#
#     plt.suptitle(f"time = {df_np[i*ns2,0]:.2f}", fontsize=12)
#
#     savename = f"{i:04d}"
#     plt.savefig(f"{savepath}/{data_idx}_{savename}.jpg", bbox_inches="tight")
#     plt.close()

print("old plot")

# for i in range(0, 2 * max_frame // 3):
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # (11,4)
#     p0 = ax[0].pcolor(df_np[i * ns2:(i + 1) * ns2, 1].reshape(ns, -1).round(4),
#                       df_np[i * ns2:(i + 1) * ns2, 2].reshape(ns, -1).round(4),
#                       df_np[i * ns2:(i + 1) * ns2, -2].reshape(ns, -1).round(4), cmap=plt.cm.viridis, vmin=0.04,
#                       vmax=.2)
#     p1 = ax[1].pcolor(df_np[i * ns2:(i + 1) * ns2, 1].reshape(ns, -1).round(4),
#                       df_np[i * ns2:(i + 1) * ns2, 2].reshape(ns, -1).round(4),
#                       df_np[i * ns2:(i + 1) * ns2, -1].reshape(ns, -1).round(4), cmap=plt.cm.viridis, vmin=0.03,
#                       vmax=.2)
#
#     if df_np[i * ns2, 0] > next_time[0]:
#         temp_time = next_time[0]
#         current_time[0] = temp_time
#         current_idx = np.where(seq_t == current_time[0])[0][0]
#         next_time[0] = seq_t[point_types == 0][np.where(seq_t[point_types == 0] > temp_time)[0][0]]
#         current_loc[0] = seq_s[current_idx]
#     if df_np[i * ns2, 0] > next_time[1]:
#         temp_time = next_time[1]
#         current_time[1] = temp_time
#         current_idx = np.where(seq_t == current_time[1])[0][0]
#         next_time[1] = seq_t[point_types == 1][np.where(seq_t[point_types == 1] > temp_time)[0][0]]
#         current_loc[1] = seq_s[current_idx]
#
#     if i >= start_is[0]:
#         ax[0].scatter(current_loc[0][0], current_loc[0][1], c='c', s=20)
#         ax[1].scatter(current_loc[0][0], current_loc[0][1], c='c', s=20)
#     if i >= start_is[1]:
#         ax[0].scatter(current_loc[1][0], current_loc[1][1], c='m', s=20)
#         ax[1].scatter(current_loc[1][0], current_loc[1][1], c='m', s=20)
#     plt.colorbar(p0, ax=[ax[0], ax[1]], location='left', format='%.3f')
#     plt.colorbar(p1, ax=[ax[0], ax[1]], location='right', format='%.3f')
#     plt.suptitle(f'time: {df_np[i * ns2, 0]:.2f}', fontsize=12)
#     # plt.show()
#     if i < 10:
#         savename = f'000{i}'
#     elif i < 100:
#         savename = f'00{i}'
#     elif i < 1000:
#         savename = f'0{i}'
#     else:
#         savename = f'{i}'
#     plt.savefig(f'{savepath}/{data_idx}_{savename}.jpg', bbox_inches='tight')
#     plt.close()


### Turn into video ###

#  ffmpeg -framerate 5 -i 4_%04d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p obj.mp4

### Temporal component comparison

time_points = df.t.to_numpy().reshape(-1, ns2)
intensity_0 = df.iloc[:, 3].to_numpy().reshape(-1, ns2)
intensity_1 = df.iloc[:, 4].to_numpy().reshape(-1, ns2)

## CCC###


#ffmpeg -framerate 1 -start_number 0 -i it%d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p obj.mp4


unit_area = 2**2/ns2

#print(f"unit_area:{unit_area}, cell_area:{cell_area_deg2}")

temporal_intensity_0 = intensity_0.sum(axis=1)*unit_area
temporal_intensity_1 = intensity_1.sum(axis=1)*unit_area


time_df = pd.DataFrame({'t': time_points[:, 0], 'intensity_0': temporal_intensity_0, 'intensity_1': temporal_intensity_1})
time_df.describe()


print(f"int0: {len(time_df.intensity_0)}, int1: {len(time_df.intensity_1)}, time: {len(time_df.t)}")
print(f"event0: {len(seq_t[point_types == 0])}, event1: {len(seq_t[point_types == 1])}")



# ffmpeg -framerate 5 -i 0_%04d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p Non_Separable_stnhp_1000_epochs.mp4

# fig, ax = plt.subplots(figsize=(10, 4))
#
# ax.plot(time_df.t, time_df.intensity_0, label='Type 0')
# ax.plot(time_df.t, time_df.intensity_1, label='Type 1')
#
# ymin, ymax = 0.4, 2.4
# ax.set_ylim(ymin, ymax)
#
# baseline = 0.02  # 2% above axes bottom
# x0 = seq_t[point_types == 0]
# x1 = seq_t[point_types == 1]
# ax.scatter(x0, np.full_like(x0, baseline, dtype=float), s=10,
#            transform=ax.get_xaxis_transform(), clip_on=False)
# ax.scatter(x1, np.full_like(x1, baseline, dtype=float), s=10,
#            transform=ax.get_xaxis_transform(), clip_on=False)
#
# ax.legend()
# fig.tight_layout()
# plt.savefig(f'{savepath}/bistpp_bivariate_4.png', dpi=150, bbox_inches='tight')
# plt.show()
# plt.close()


def add_event_ticks_axesfrac(ax, times, color, base=0.02, height=0.018,
                             lw=1.0, alpha=0.95, zorder=3, capstyle='butt'):
    """Short vertical 'rug' lines with height in axes fraction (y is independent of y-lims)."""
    times = np.asarray(times)
    if times.size == 0:
        return
    y0, y1 = base, base + height
    segs = [((t, y0), (t, y1)) for t in times]
    lc = LineCollection(segs, colors=[color], linewidths=lw, alpha=alpha,
                        capstyle=capstyle, zorder=zorder)
    lc.set_transform(mtransforms.blended_transform_factory(ax.transData, ax.transAxes))
    lc.set_clip_on(True)
    ax.add_collection(lc)


# styling knobs
LABEL_SIZE = 20
TICK_SIZE = 18
LEGEND_SIZE = 18
LINE_LW = 1.6

# event “rug” appearance
RUG_BASE = 0.025   # distance from bottom of axes (0..1)
RUG_HEIGHT = 0.020  # vertical length in axes fraction
RUG_LW = 1.0

# choose y-limits
Y_LIM = (0.4, 2.4)
Y_QUANT = 0.99

# fig, ax = plt.subplots(figsize=(10, 4))
#
# # lines
# line0, = ax.plot(time_df.t, time_df.intensity_0, color='C0', lw=LINE_LW)  # label='Type 0',
# line1, = ax.plot(time_df.t, time_df.intensity_1,  color='C1', lw=LINE_LW)  # label='Type 1',
#
# # y-limits
# if Y_LIM is None:
#     all_vals = np.r_[time_df.intensity_0.values, time_df.intensity_1.values]
#     ymin = 0.0
#     if np.isfinite(all_vals).any():
#         ymax = float(np.nanquantile(all_vals[~np.isnan(all_vals)], Y_QUANT))
#         if not np.isfinite(ymax) or ymax <= 0:
#             ymax = float(np.nanmax(all_vals)) if np.isfinite(np.nanmax(all_vals)) else 1.0
#     else:
#         ymax = 1.0
# else:
#     ymin, ymax = Y_LIM
# ax.set_ylim(ymin, ymax)
#
#
# # event rugs
# x0 = seq_t[point_types == 0]
# x1 = seq_t[point_types == 1]
# add_event_ticks_axesfrac(ax, x0, color=line0.get_color(), base=RUG_BASE, height=RUG_HEIGHT, lw=RUG_LW)
# add_event_ticks_axesfrac(ax, x1, color=line1.get_color(), base=RUG_BASE, height=RUG_HEIGHT, lw=RUG_LW)
#
#
# # labels/legend/fonts
# #ax.set_xlabel("Time (days)", fontsize=LABEL_SIZE)
# #ax.set_ylabel("Temporal intensity", fontsize=LABEL_SIZE)
# ax.tick_params(axis='both', labelsize=TICK_SIZE)
# #ax.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=False)
#
# #fig.tight_layout()
# fig.subplots_adjust(left=0.12, bottom=0.20, right=0.98, top=0.95)
# plt.savefig(f'{savepath}/{dataname}.pdf', dpi=300, )
# #plt.show()
# plt.close()
