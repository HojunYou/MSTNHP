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
parser.add_argument('--ckpt_dir', default='./checkpoints/45442_23280759170048_250923-061811/models/saved_model')
parser.add_argument('--data_path', default='./data/chicago_crime_2001-2022_downtown/train.pkl')
parser.add_argument('--polygon', default='./data/gtd_pakistan/chicago_polygon_norm.pkl')
parser.add_argument('--polygon_aoi', default='./data/gtd_pakistan/chicago_aoi_south_norm.pkl')
parser.add_argument('--sample_index', type=int, default=2)
parser.add_argument('--save_root', default='../Meeting/2025/250924')
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
batch_size = 3
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

def apply_min_threshold(interval_t_samples, ns, eps=1e-2):
    """
    Shift the first within-interval time slice by +eps (in your time units,
    hours now) to avoid evaluating intensities at t=0, which can cause
    huge peaks.
    """
    ns2 = ns * ns
    for i, ts in enumerate(interval_t_samples):
        # guard empty
        if ts is None or ts.numel() == 0:
            continue
        # clamp only the first time-slice (the first ns^2 entries)
        head = ts[:ns2]
        need = head < eps
        if need.any():
            ts[:ns2][need] = eps
            print(f"[thresh] added eps={eps} to first slice of interval {i}")


interval_t_samples = [
    st_sample_list[data_idx][i][:, 0, 0] - time_seqs[data_idx, i]
    for i in range(len(st_sample_list[data_idx]))
]
interval_s_samples = [
    torch.norm(st_sample_list[data_idx][i][:, :, 1:], p=2, dim=-1)
    for i in range(len(st_sample_list[data_idx]))
]

# nudge the first time-slice away from 0 to avoid singular spikes
apply_min_threshold(interval_t_samples, ns, eps=1e-2)  # 0.01 hours ≈ 36 sec

# threshold first delta to avoid peaks
# min_thres = 0.01
# for i, interval_t_sample in enumerate(interval_t_samples):
#     if interval_t_sample.any() and interval_t_sample[0] <= min_thres:
#         interval_t_sample[:ns**2] += min_thres

# states  intensity
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

# with open(args.polygon_aoi, 'rb') as f:
#     pakistan_polygon = pickle.load(f)
# pakistan_gdf = gpd.GeoSeries([pakistan_polygon])
#
# if isinstance(pakistan_polygon, sg.Polygon):
#     area = pakistan_polygon.area
# else:
#     area = sum(p.area for p in pakistan_polygon.geoms)
# print('polygon bounds:', pakistan_polygon.bounds)


#  Load polygons
# Full Chicago (context outline)
# with open(args.polygon, 'rb') as f:
#     chicago_polygon = pickle.load(f)
# chicago_gdf = gpd.GeoSeries([chicago_polygon])

# downtown chicago (mask + area)
with open(args.polygon_aoi, 'rb') as f:
    aoi_polygon = pickle.load(f)
aoi_gdf = gpd.GeoSeries([aoi_polygon])

#  area (downtown chicago)
if isinstance(aoi_polygon, sg.Polygon):
    area = aoi_polygon.area
else:
    area = sum(p.area for p in aoi_polygon.geoms)
print('AOI bounds:', aoi_polygon.bounds)



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
inside = pts.within(aoi_gdf.iloc[0]).values           # shape (ns2,)
cell_area = (2.0 / ns) ** 2

I = df_np[:, 3:3+K].reshape(max_frame, ns2, K)

#integrate in space over AOI, then over time
space_sum = I[:, inside, :].sum(axis=1) * cell_area          # [frames, K]
fitted_points = space_sum.sum(axis=0) * (time_domain / num_sample_t)  # [K]

print(f"Actual_points:{actual_points}")
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
              .within(aoi_gdf.iloc[0]).values.reshape(ns, ns)

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
chosen = ["THEFT", "BATTERY"]   # change if you want
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
# pA_lo, pA_hi = np.nanpercentile(df_np[:, colA], [1, 99])
# pB_lo, pB_hi = np.nanpercentile(df_np[:, colB], [1, 99])


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

print("old block")
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
#     ).within(aoi_gdf.iloc[0]).values.reshape(ns, ns)
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
#
#
#     p0 = ax[0].pcolormesh(x_block, y_block, mA, shading='auto', cmap='coolwarm', vmin=vminA, vmax=vmaxA)
#     p1 = ax[1].pcolormesh(x_block, y_block, mB, shading='auto', cmap='coolwarm', vmin=vminB, vmax=vmaxB)
#
#     aoi_gdf.boundary.plot(ax=ax[0], linewidth=1)
#     aoi_gdf.boundary.plot(ax=ax[1], linewidth=1)
#
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

# zoom window (in normalized coords)
minx, miny, maxx, maxy = aoi_polygon.bounds
pad = 0.03  # small margin around the AOI
xlim = (minx - pad, maxx + pad)
ylim = (miny - pad, maxy + pad)

X0 = df_np[:ns2, 1].reshape(ns, ns)
Y0 = df_np[:ns2, 2].reshape(ns, ns)

gdf_points0 = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in zip(X0.ravel(), Y0.ravel())]
)
inside_mask = gdf_points0.within(aoi_gdf.iloc[0]).values.reshape(ns, ns)


# robust, stable color limits PER TYPE across all frames
pA_lo, pA_hi = np.nanpercentile(df_np[:, 3 + typeA], [1, 99])
pB_lo, pB_hi = np.nanpercentile(df_np[:, 3 + typeB], [1, 99])

#  simple bilinear upsampling on a rect. grid
def _interp_along_cols(M, new_ncols):
    """Interpolate each row of M from ncols -> new_ncols."""
    nrows, ncols = M.shape
    x_old = np.arange(ncols)
    x_new = np.linspace(0, ncols - 1, new_ncols)
    out = np.empty((nrows, new_ncols), dtype=float)
    for r in range(nrows):
        out[r] = np.interp(x_new, x_old, M[r])
    return out

def _interp_along_rows(M, new_nrows):
    """Interpolate each column of M from nrows -> new_nrows."""
    nrows, ncols = M.shape
    y_old = np.arange(nrows)
    y_new = np.linspace(0, nrows - 1, new_nrows)
    out = np.empty((new_nrows, ncols), dtype=float)
    for c in range(ncols):
        out[:, c] = np.interp(y_new, y_old, M[:, c])
    return out

def upsample_grid(X, Y, Z, factor=3):
    """
    Bilinear-ish upsampling of a rectilinear grid.
    X, Y, Z are ns×ns. Returns Xh, Yh, Zh of shape (ns*factor, ns*factor).
    """
    nr, nc = Z.shape
    nc_h = nc * factor
    nr_h = nr * factor
    # cols
    Zc = _interp_along_cols(Z, nc_h)
    Xc = _interp_along_cols(X, nc_h)
    Yc = _interp_along_cols(Y, nc_h)
    # rows
    Zh = _interp_along_rows(Zc, nr_h)
    Xh = _interp_along_rows(Xc, nr_h)
    Yh = _interp_along_rows(Yc, nr_h)
    return Xh, Yh, Zh

def per_frame_limits(masked_frame, lo=2, hi=98):
    """
    Robust per-frame limits from the *AOI-masked* values.
    Falls back to min/max if percentiles collapse.
    """
    if masked_frame.count() == 0:
        return 0.0, 1e-6
    vals = masked_frame.compressed()
    vmin, vmax = np.percentile(vals, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            vmax = vmin + 1e-6
    return float(vmin), float(vmax)


print("new block")

UPSAMPLE = 3            # 1 = no upsample; try 2–4 for smoother look
PCT_LOW, PCT_HIGH = 0, 100   # per-frame robust limits (in percentiles)

for i in range(max_frame):
    start, stop = i * ns2, (i + 1) * ns2
    if stop > N:
        break

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=180)

    # base ns×ns grids
    X = df_np[start:stop, 1].reshape(ns, ns)
    Y = df_np[start:stop, 2].reshape(ns, ns)

    # AOI mask (ns×ns)
    inside_mask = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(df_np[start:stop, 1], df_np[start:stop, 2])],
        crs="EPSG:4326"
    ).within(aoi_gdf.iloc[0]).values.reshape(ns, ns)

    # intensities (ns×ns)
    ZA = df_np[start:stop, colA].reshape(ns, ns)
    ZB = df_np[start:stop, colB].reshape(ns, ns)

    # mask out-of-AOI
    mA = np.ma.masked_where(~inside_mask, ZA)
    mB = np.ma.masked_where(~inside_mask, ZB)

    # global limits (precomputed)
    vminA, vmaxA = float(pA_lo), float(pA_hi)
    vminB, vmaxB = float(pB_lo), float(pB_hi)

    # ---- PER-FRAME limits from AOI only (no global scaling) ----
    # vminA, vmaxA = per_frame_limits(mA, lo=PCT_LOW, hi=PCT_HIGH)
    # vminB, vmaxB = per_frame_limits(mB, lo=PCT_LOW, hi=PCT_HIGH)

    # ---- OPTIONAL upsample for a smoother look ----
    if UPSAMPLE > 1:
        Xh, Yh, ZAh = upsample_grid(X, Y, ZA, factor=UPSAMPLE)
        _,  _,  ZBh = upsample_grid(X, Y, ZB, factor=UPSAMPLE)
        # upsample the boolean mask by nearest-neighbor
        mask_h = inside_mask.repeat(UPSAMPLE, axis=0).repeat(UPSAMPLE, axis=1)
        mAh = np.ma.masked_where(~mask_h, ZAh)
        mBh = np.ma.masked_where(~mask_h, ZBh)

        # draw
        p0 = ax[0].pcolormesh(Xh, Yh, mAh, shading='auto', cmap='coolwarm',
                              vmin=vminA, vmax=vmaxA)
        p1 = ax[1].pcolormesh(Xh, Yh, mBh, shading='auto', cmap='coolwarm',
                              vmin=vminB, vmax=vmaxB)
    else:
        p0 = ax[0].pcolormesh(X, Y, mA, shading='auto', cmap='coolwarm',
                              vmin=vminA, vmax=vmaxA)
        p1 = ax[1].pcolormesh(X, Y, mB, shading='auto', cmap='coolwarm',
                              vmin=vminB, vmax=vmaxB)

    # outline AOI and set axes to AOI bounds (zooms in)
    aoi_gdf.boundary.plot(ax=ax[0], linewidth=1, color='k')
    aoi_gdf.boundary.plot(ax=ax[1], linewidth=1, color='k')
    xmin, ymin, xmax, ymax = aoi_polygon.bounds
    ax[0].set_xlim(xmin, xmax); ax[0].set_ylim(ymin, ymax)
    ax[1].set_xlim(xmin, xmax); ax[1].set_ylim(ymin, ymax)
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')

    # model time in HOURS
    t_cur = float(df_np[start, 0])
    cal_dt = day0 + timedelta(hours=t_cur)
    date_label = cal_dt.strftime("%b %d, %H:%M")
    # update trackers + event markers (unchanged)
    if t_cur > trkA["nxt_t"]:
        trkA["cur_t"] = trkA["nxt_t"]; trkA["idx_ptr"] += 1
        if trkA["idx_ptr"] < trkA["times"].size:
            trkA["nxt_t"] = trkA["times"][trkA["idx_ptr"]] if trkA["idx_ptr"] + 1 >= trkA["times"].size else trkA["times"][trkA["idx_ptr"] + 1]
            cur_idx = np.where(seq_t == trkA["cur_t"])[0]
            if cur_idx.size > 0: trkA["cur_loc"] = seq_s[cur_idx[0]]
        else:
            trkA["nxt_t"] = np.inf

    if t_cur > trkB["nxt_t"]:
        trkB["cur_t"] = trkB["nxt_t"]; trkB["idx_ptr"] += 1
        if trkB["idx_ptr"] < trkB["times"].size:
            trkB["nxt_t"] = trkB["times"][trkB["idx_ptr"]] if trkB["idx_ptr"] + 1 >= trkB["times"].size else trkB["times"][trkB["idx_ptr"] + 1]
            cur_idx = np.where(seq_t == trkB["cur_t"])[0]
            if cur_idx.size > 0: trkB["cur_loc"] = seq_s[cur_idx[0]]
        else:
            trkB["nxt_t"] = np.inf

    if i >= trkA["start_i"] and trkA["cur_loc"] is not None:
        ax[0].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], color=type1_color, s=28, edgecolor='white', linewidths=0.4, zorder=3)
        ax[1].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], color=type1_color, s=28, edgecolor='white', linewidths=0.4, zorder=3)
    if i >= trkB["start_i"] and trkB["cur_loc"] is not None:
        ax[0].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], color=type2_color, s=28, edgecolor='white', linewidths=0.4, zorder=3)
        ax[1].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], color=type2_color, s=28, edgecolor='white', linewidths=0.4, zorder=3)

    # cosmetics
    for a in ax:
        a.tick_params(labelsize=9)
        a.grid(False)
        a.set_xlabel("");
        a.set_ylabel("")
    cbar0 = fig.colorbar(p0, ax=ax[0], fraction=0.046, pad=0.02)
    cbar1 = fig.colorbar(p1, ax=ax[1], fraction=0.046, pad=0.02)
    cbar0.set_label(f"{labelA} λ", rotation=90)
    cbar1.set_label(f"{labelB} λ", rotation=90)
    fig.suptitle(f"{date_label} — {labelA}  |  {labelB}", fontsize=18)

    savename = f'{i:04d}'
    plt.savefig(f'{savepath}/{year_list[args.sample_index]}_{savename}.jpg',
                bbox_inches='tight')
    plt.close(fig)

print("new block finished")


#  Temporal plots (windowed)
# user window in HOURS since month start
T0, T1 = 0.0, time_domain  #50.0  # time_domain  # 100.0 #

# def moving_average_1d(x, win=3):
#     if win <= 1:
#         return x
#     k = int(win)
#     k = max(k, 1)
#     kernel = np.ones(k) / k
#     # 'same' keeps length; pad at the ends to avoid shrinkage
#     pad = k // 2
#     xpad = np.pad(x, (pad, k - 1 - pad), mode='edge')
#     return np.convolve(xpad, kernel, mode='valid')
#
# def robust_cap(x, q=99.5):
#     vmax = np.nanpercentile(x, q)
#     return np.clip(x, 0.0, vmax), float(vmax)


def plot_temporal(types_to_draw, title_tag, filename_tag):
    tp_win, mask, unit_area, seq_t_valid, types_valid, obs_counts = \
        temporal_window_prep(df, ns, ns2, seq_t, point_types, args.sample_index, T0, T1)

    #colors = ['#6AAED6', '#F28E2B']

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
    plt.ylim(-0.1, 10)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"{year_cur}_{filename_tag}_{int(T0)}-{int(T1)}h.png"), dpi=300)
    plt.close()

#--- new plotting function (same signature/behavior) ---
# def plot_temporal(types_to_draw, title_tag, filename_tag):
#     # window prep
#     tp_win, mask, unit_area, seq_t_valid, types_valid, obs_counts = \
#         temporal_window_prep(df, ns, ns2, seq_t, point_types, args.sample_index, T0, T1)
#
#     # vis-only knobs
#     SMOOTH_WIN = 3     # 1 = no smoothing; try 3 or 5 for mild smoothing
#     CAP_Q      = 99.5  # clip plotted curve at this upper percentile
#     JITTER     = 0.0   # e.g., 0.02 hours if you want tiny horizontal jitter on dots
#
#     # colors (cycle if you pass >2 types)
#     color_cycle = ['#6AAED6', '#F28E2B', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7']
#
#     plt.figure(figsize=(15, 6))
#
#     for j, k in enumerate(types_to_draw):
#         # space-summed λ_k(t)
#         Ik_full    = df[f"intensity{k}"].to_numpy().reshape(-1, ns2)
#         temporal_k = (Ik_full.sum(axis=1) * unit_area)[mask]
#
#         # optional smoothing + robust cap (plotting only)
#         if SMOOTH_WIN > 1:
#             temporal_k = moving_average_1d(temporal_k, win=SMOOTH_WIN)
#         temporal_k, _ = robust_cap(temporal_k, q=CAP_Q)
#
#         # events in window for this type
#         t_k  = seq_t_valid[types_valid == k]
#         t_kw = t_k[(t_k >= T0) & (t_k <= T1)]
#         if JITTER > 0 and t_kw.size:
#             rng  = np.random.default_rng(123)
#             t_kw = t_kw + rng.uniform(-JITTER, JITTER, size=t_kw.size)
#
#         label = TYPE_LABELS[k] if k < len(TYPE_LABELS) else f"type {k}"
#         if np.sum(types_valid == k) == 0:
#             print(f"[warn] No observed events for {label} in {year_cur}")
#
#         c = color_cycle[j % len(color_cycle)]
#         plt.scatter(t_kw, np.zeros_like(t_kw), s=10, color=c)
#         plt.plot(tp_win, temporal_k, color=c,
#                  label=f"{label} λ(t) (n={int(t_kw.size)}, y={year_cur})")
#
#     plt.xlim(T0, T1)
#     plt.legend(loc="upper right")
#     plt.tight_layout()
#     plt.savefig(os.path.join(savepath, f"{year_cur}_{filename_tag}_{int(T0)}-{int(T1)}h.png"), dpi=300)
#     plt.close()


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
