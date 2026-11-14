import os, argparse, pickle, numpy as np, torch, matplotlib.pyplot as plt
import pandas as pd, geopandas as gpd
import shapely.geometry as sg
from shapely.geometry import Point
from datetime import datetime, timedelta

from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader
import datetime as dt

#  CLI
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml')
parser.add_argument('--experiment_id', default='STNHP_eval')
parser.add_argument('--ckpt_dir', default='./checkpoints/100296_23058698261504_250913-204729/models/saved_model')
parser.add_argument('--data_path', default='./data/chicago_crime_2001-2022_downtown/train.pkl')
parser.add_argument('--polygon', default='./data/gtd_pakistan/chicago_polygon_norm.pkl')
parser.add_argument('--polygon_aoi', default='./data/gtd_pakistan/chicago_aoi_south_norm.pkl')
parser.add_argument('--sample_index', type=int, default=1)
parser.add_argument('--save_root', default='../Meeting/2025/250913')
parser.add_argument('--dataname', default='chicago_crime_2001-2022_downtown')
args = parser.parse_args()

#  Run setup
dataname = args.dataname
year_list = list(range(2001, 2023))   # inclusive 2001–2022
year_cur = year_list[argparse.Namespace(sample_index=args.sample_index).sample_index]

config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)

n_comp, n_epochs, n_times = model_runner.model_wrapper.model.n_comps, 500, 50
model_runner.model_wrapper.model.load_state_dict(torch.load(args.ckpt_dir, map_location='cpu'), strict=False)
model_runner.model_wrapper.model.eval()

data_config = model_runner.runner_config.data_config
model_backend = model_runner.runner_config.base_config.backend
split = 'train'
data_dir = data_config.get_data_dir(split)
print(f"data_dir: {data_dir}")

data_loader = STPPDataLoader(data_config, model_backend)
data = data_loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)
tokenizer = STEventTokenizer(data_config.data_specs)
batch_size = 2
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size=batch_size, shuffle=False, space=True)

# one batch (year) to visualize
batch = next(iter(train_loader))
time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch.values()

#  Sampling grid
ns = 40
num_sample_t = 1000
MONTH_KEEP = 7          # our data were filtered to July
time_domain = 7 * 24    # hours in first week of July
model_runner.model_wrapper.model.spatial_npoints = ns

st_sample_list = model_runner.model_wrapper.model.make_multiple_equal_dtimespace_samples(
    time_seqs, [0, time_domain], space_seqs, num_sample_t=num_sample_t
)
st_sample_location = model_runner.model_wrapper.model.make_equal_dtimespace_samples(
    time_seqs, [0, time_domain], torch.zeros_like(space_seqs), num_sample_t=num_sample_t
)

# forward pass
input_ = time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, None, None, None
hiddens_ti, decays, output_states = model_runner.model_wrapper.model.forward(input_)

# pick sample index (year)
data_idx = args.sample_index

interval_t_samples = [
    st_sample_list[data_idx][i][:, 0, 0] - time_seqs[data_idx, i]
    for i in range(len(st_sample_list[data_idx]))
]
interval_s_samples = [
    torch.norm(st_sample_list[data_idx][i][:, :, 1:], p=2, dim=-1)
    for i in range(len(st_sample_list[data_idx]))
]

# threshold first delta to avoid peaks
min_thres = 0.01
for i, interval_t_sample in enumerate(interval_t_samples):
    if interval_t_sample.any() and interval_t_sample[0] <= min_thres:
        interval_t_sample[:ns**2] += min_thres

# states -- intensity
state_t_samples = [
    model_runner.model_wrapper.model.compute_states_at_sample_points(
        decays[data_idx:(data_idx+1), i:(i+1)],
        output_states[data_idx:(data_idx+1), i:(i+1)],
        t_sample, s_sample
    )
    for i, (t_sample, s_sample) in enumerate(zip(interval_t_samples, interval_s_samples))
]
lambda_t_samples = [model_runner.model_wrapper.model.layer_intensity(s) for s in state_t_samples]
lambda_t_samples = [x.detach().cpu().numpy()[0, 0] for x in lambda_t_samples]
lambda_t_samples = np.concatenate(lambda_t_samples, axis=0)

sample_intensity = lambda_t_samples.copy()
st_sample = torch.cat(st_sample_location[data_idx], dim=0).cpu().numpy()

# quick counts
seq_t = time_seqs[data_idx].numpy()
seq_s = space_seqs[data_idx].numpy()
point_types = type_seqs[data_idx].numpy()
fitted_points = sample_intensity.sum(axis=0) * time_domain / num_sample_t * (2 / ns) ** 2
expected_points = time_seqs[data_idx].argmax().item() + 1
actual_points = np.unique(type_seqs[data_idx, :expected_points].numpy(), return_counts=True)[1]
#print(f"Fitted_points:{fitted_points} \nActual_points:{actual_points}")

#  Paths & polygon
ns2 = ns ** 2
savepath = f'{args.save_root}/{dataname}/{split}/{n_comp}_{n_epochs}'
if n_times != 50:
    savepath = f'{savepath}_{n_times}'
os.makedirs(savepath, exist_ok=True)

with open(args.polygon_aoi, 'rb') as f:
    pakistan_polygon = pickle.load(f)
pakistan_gdf = gpd.GeoSeries([pakistan_polygon])

if isinstance(pakistan_polygon, sg.Polygon):
    area = pakistan_polygon.area
else:
    area = sum(p.area for p in pakistan_polygon.geoms)
print('polygon bounds:', pakistan_polygon.bounds)


#  Build per-sample DataFrame
K = sample_intensity.shape[1]
df = pd.DataFrame({"t": st_sample[:, 0], "x": st_sample[:, 1], "y": st_sample[:, 2]})
df = pd.concat([df, pd.DataFrame(sample_intensity, columns=[f"intensity{k}" for k in range(K)])], axis=1)
df_np = df.to_numpy()
max_frame = len(df_np) // ns2
print("Grid:", ns, "×", ns, "| frames:", max_frame, "| K =", K)

####################
xy0 = df_np[:ns2, 1:3]                                # (x,y) of first frame
pts = gpd.GeoDataFrame(geometry=[Point(x,y) for x,y in xy0], crs="EPSG:4326")
inside = pts.within(pakistan_gdf.iloc[0]).values           # shape (ns2,)
cell_area = (2.0 / ns) ** 2

I = df_np[:, 3:3+K].reshape(max_frame, ns2, K)

#integrate in space over AOI, then over time
space_sum = I[:, inside, :].sum(axis=1) * cell_area          # [frames, K]
fitted_points = space_sum.sum(axis=0) * (time_domain / num_sample_t)  # [K]

print(f"Fitted_points:{fitted_points} \nActual_points:{actual_points}")
print("AOI-masked fitted points:", fitted_points)
#####################





# labels (trim to K)
TYPE_LABELS = [
    "Assault", "BATTERY", "Burglary", "Criminal Damage",
    "Deceptive Practice", "Motor Vehicle Theft", "Robbery", "THEFT",
][:K]

# polygon mask (fixed grid)
xy0 = df_np[:ns2, 1:3]
poly_mask = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy0], crs="EPSG:4326") \
              .within(pakistan_gdf.iloc[0]).values.reshape(ns, ns)

# pre-split events by type (for plotting event markers)
events_by_type = []
for k in range(K):
    m = (point_types == k)
    events_by_type.append({"t": seq_t[m], "xy": seq_s[m]})


#  Utility functions for plots
def _norm(s): return str(s).strip().lower()


# to choose types to plot
def resolve_types(chosen_list, labels, K):
    label_to_idx = {_norm(lbl): i for i, lbl in enumerate(labels)}
    out = []
    for item in chosen_list:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if not (0 <= idx < K):
                raise ValueError(f"type index {idx} out of range [0, {K-1}]")
            out.append(idx)
        else:
            key = _norm(item)
            if key in label_to_idx:
                out.append(label_to_idx[key])
            else:
                matches = [i for i, lbl in enumerate(labels) if key in _norm(lbl)]
                if len(matches) == 1:
                    out.append(matches[0])
                elif len(matches) > 1:
                    raise ValueError(f"Name '{item}' is ambiguous; matches {[labels[i] for i in matches]}")
                else:
                    raise ValueError(f"Name '{item}' not found in TYPE_LABELS.")
    if len(out) != 2:
        raise ValueError(f"Please pick exactly 2 types; got {out}")
    return out


# to plot for a specific time window
def temporal_window_prep(df, ns, ns2, seq_t, point_types, data_idx, T0, T1):
    K = sample_intensity.shape[1]
    L = int(batch_non_pad_mask[data_idx].sum().item())
    seq_t_valid = seq_t[:L]
    types_valid = point_types[:L]
    obs_counts = np.bincount(types_valid, minlength=K)
    time_points = df["t"].to_numpy().reshape(-1, ns2)[:, 0]
    mask = (time_points >= T0) & (time_points <= T1)
    if not mask.any():
        print(f"[warn] no frames in [{T0},{T1}]h; plotting full range.")
        mask = slice(None)
    tp_win = time_points[mask] if isinstance(mask, np.ndarray) else time_points
    unit_area = (2.0 / ns) ** 2
    return tp_win, mask, unit_area, seq_t_valid, types_valid, obs_counts


# for event markers on spatial frames
def init_tracker(k):
    mask = (point_types == k)
    t_k, s_k = seq_t[mask], seq_s[mask]
    if t_k.size > 0:
        cur_t = t_k[0]
        nxt_t = t_k[1] if t_k.size > 1 else np.inf
        cur_loc = s_k[0]
        start_i = int(np.searchsorted(df_np[:, 0][::ns2], cur_t, side='left'))
    else:
        cur_t, nxt_t, cur_loc, start_i = np.inf, np.inf, None, max_frame + 1
    return {"times": t_k, "locs": s_k, "cur_t": cur_t, "nxt_t": nxt_t,
            "cur_loc": cur_loc, "start_i": start_i, "idx_ptr": 0}


# TYPE_LABELS = [
#     "Assault", "BATTERY", "Burglary", "Criminal Damage",
#     "Deceptive Practice", "Motor Vehicle Theft", "Robbery", "THEFT",
# ]

#  Spatial frames (two selected groups)
chosen = ["Deceptive Practice", "Motor Vehicle Theft"]   # change if you want
typeA, typeB = resolve_types(chosen, TYPE_LABELS, K)
labelA = TYPE_LABELS[typeA] if typeA < len(TYPE_LABELS) else f"type {typeA}"
labelB = TYPE_LABELS[typeB] if typeB < len(TYPE_LABELS) else f"type {typeB}"

trkA, trkB = init_tracker(typeA), init_tracker(typeB)

print(f"Min/Max {labelA}:", df[f'intensity{typeA}'].min(), df[f'intensity{typeA}'].max())
print(f"Min/Max {labelB}:", df[f'intensity{typeB}'].min(), df[f'intensity{typeB}'].max())

type1_color, type2_color = 'cyan', 'magenta'
colA, colB = 3 + typeA, 3 + typeB
day0 = datetime(year_list[args.sample_index], MONTH_KEEP, 1, 0, 0)

print("Begin saving frames")
N = df_np.shape[0]
max_frame = N // ns2

# robust global fallbacks
pA_lo, pA_hi = np.nanpercentile(df_np[:, colA], [1, 99])
pB_lo, pB_hi = np.nanpercentile(df_np[:, colB], [1, 99])


def safe_limits(masked_frame, raw_frame, p_low, p_high):
    try:
        vmin, vmax = masked_frame.min(), masked_frame.max()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            raise ValueError
    except Exception:
        vmin, vmax = np.nanpercentile(raw_frame, [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = p_low, p_high
    return float(vmin), float(vmax)


# for i in range(max_frame):
#     start, stop = i * ns2, (i + 1) * ns2
#     if stop > N:
#         break
#
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
#
#     x_block = df_np[start:stop, 1].reshape(ns, ns)
#     y_block = df_np[start:stop, 2].reshape(ns, ns)
#
#     inside_mask = gpd.GeoDataFrame(
#         geometry=[Point(x, y) for x, y in zip(df_np[start:stop, 1], df_np[start:stop, 2])],
#         crs="EPSG:4326"
#     ).within(pakistan_gdf.iloc[0]).values.reshape(ns, ns)
#
#     frameA = df_np[start:stop, colA].reshape(ns, ns)
#     frameB = df_np[start:stop, colB].reshape(ns, ns)
#
#     mA = np.ma.masked_where(~inside_mask, frameA)
#     mB = np.ma.masked_where(~inside_mask, frameB)
#
#     vminA, vmaxA = safe_limits(mA, frameA, pA_lo, pA_hi)
#     vminB, vmaxB = safe_limits(mB, frameB, pB_lo, pB_hi)
#
#     p0 = ax[0].pcolormesh(x_block, y_block, mA, shading='auto', cmap='coolwarm', vmin=vminA, vmax=vmaxA)
#     p1 = ax[1].pcolormesh(x_block, y_block, mB, shading='auto', cmap='coolwarm', vmin=vminB, vmax=vmaxB)
#
#     pakistan_gdf.boundary.plot(ax=ax[0], linewidth=1)
#     pakistan_gdf.boundary.plot(ax=ax[1], linewidth=1)
#
#     # model time in HOURS
#     t_cur = float(df_np[start, 0])
#     cal_dt = day0 + timedelta(hours=t_cur)
#     date_label = cal_dt.strftime("%b %d, %H:%M")
#
#     # update trackers and event markers
#     if t_cur > trkA["nxt_t"]:
#         trkA["cur_t"] = trkA["nxt_t"]; trkA["idx_ptr"] += 1
#         if trkA["idx_ptr"] < trkA["times"].size:
#             trkA["nxt_t"] = trkA["times"][trkA["idx_ptr"]] if trkA["idx_ptr"] + 1 >= trkA["times"].size else trkA["times"][trkA["idx_ptr"] + 1]
#             cur_idx = np.where(seq_t == trkA["cur_t"])[0]
#             if cur_idx.size > 0: trkA["cur_loc"] = seq_s[cur_idx[0]]
#         else:
#             trkA["nxt_t"] = np.inf
#
#     if t_cur > trkB["nxt_t"]:
#         trkB["cur_t"] = trkB["nxt_t"]; trkB["idx_ptr"] += 1
#         if trkB["idx_ptr"] < trkB["times"].size:
#             trkB["nxt_t"] = trkB["times"][trkB["idx_ptr"]] if trkB["idx_ptr"] + 1 >= trkB["times"].size else trkB["times"][trkB["idx_ptr"] + 1]
#             cur_idx = np.where(seq_t == trkB["cur_t"])[0]
#             if cur_idx.size > 0: trkB["cur_loc"] = seq_s[cur_idx[0]]
#         else:
#             trkB["nxt_t"] = np.inf
#
#     if i >= trkA["start_i"] and trkA["cur_loc"] is not None:
#         ax[0].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], color=type1_color, s=20)
#         ax[1].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], color=type1_color, s=20)
#     if i >= trkB["start_i"] and trkB["cur_loc"] is not None:
#         ax[0].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], color=type2_color, s=20)
#         ax[1].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], color=type2_color, s=20)
#
#     plt.colorbar(p0, ax=ax[0], location='right', format='%.2f')
#     plt.colorbar(p1, ax=ax[1], location='right', format='%.2f')
#     plt.suptitle(f"{date_label} — {labelA}  |  {labelB}", fontsize=12)
#
#     savename = f'{i:04d}'
#     plt.savefig(f'{savepath}/{year_list[args.sample_index]}_{savename}.jpg', bbox_inches='tight')
#     plt.close()

print("All frames saved")


#  Temporal plots (windowed)
# user window in HOURS since month start
T0, T1 = 0.0, time_domain  #50.0  # time_domain  # 100.0 #


def plot_temporal(types_to_draw, title_tag, filename_tag):
    tp_win, mask, unit_area, seq_t_valid, types_valid, obs_counts = \
        temporal_window_prep(df, ns, ns2, seq_t, point_types, args.sample_index, T0, T1)

    colors = ['#6AAED6', '#F28E2B']

    plt.figure(figsize=(15, 6))
    for j, k in enumerate(types_to_draw):
        Ik_full = df[f"intensity{k}"].to_numpy().reshape(-1, ns2)
        temporal_k = (Ik_full.sum(axis=1) * unit_area)[mask]

        t_k = seq_t_valid[types_valid == k]
        t_kw = t_k[(t_k >= T0) & (t_k <= T1)]

        label = TYPE_LABELS[k] if k < len(TYPE_LABELS) else f"type {k}"
        if np.sum(types_valid == k) == 0:
            print(f"[warn] No observed events for {label} in {year_cur}")
        plt.scatter(t_kw, np.zeros_like(t_kw), s=10, )  # color=colors[j]
        plt.plot(tp_win, temporal_k,
                 label=f"{label} λ(t) (n={int(t_kw.size)}, y={year_cur})")

    plt.xlim(T0, T1)
    #plt.ylim(-0.01, 5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"{year_cur}_{filename_tag}_{int(T0)}-{int(T1)}h.png"), dpi=300)
    plt.close()


# Top-2 observed types
L = int(batch_non_pad_mask[args.sample_index].sum().item())
seq_t_valid = seq_t[:L]
types_valid = point_types[:L]
obs_counts = np.bincount(types_valid, minlength=K)
order = np.argsort(obs_counts)[::-1]
top_types = [k for k in order if obs_counts[k] > 0][:2]
if top_types:
    plot_temporal(top_types, "Top-2", "temporal_top2")
else:
    print("No events for this sequence; skipping top-2 temporal plot.")


# Two selected types (same 'chosen' list as spatial)
selected_types = [typeA, typeB]
plot_temporal(selected_types, "Selected", "temporal_selected")
