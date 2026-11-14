
# Resolve project imports and legacy paths before loading dependencies.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import os, argparse, pickle, json,  torch, matplotlib.pyplot as plt
import pandas as pd, geopandas as gpd
import shapely.geometry as sg
from shapely.geometry import Point
import datetime as dt
from matplotlib.collections import LineCollection
import matplotlib.transforms as mtransforms
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
import numpy as np

from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader


# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml')
parser.add_argument('--experiment_id', default='STNHP_eval')
parser.add_argument('--ckpt_dir', default='./checkpoints/246830_22981040100352_250927-112407/models/saved_model')
parser.add_argument('--data_path', default='./data/mstnhp_pakistan_4/train.pkl')  # used by the loader config
parser.add_argument('--polygon', default='./data/gtd_pakistan/pakistan_polygon_norm.pkl')
parser.add_argument('--sample_index', type=int, default=6)

parser.add_argument('--save_root', default='../Meeting/2026/260223')
parser.add_argument('--dataname', default='mstnhp_pakistan')


parser.add_argument('--y_transform', default='log', choices=['linear', 'sqrt', 'log', 'asinh'],
                    help='Y transform for temporal curves.')
parser.add_argument('--per_type_norm', default='max', choices=['none', 'q95', 'max', 'mean'],
                    help='Normalize each type before transforming to compare shapes.')


parser.add_argument('--ylim', default=None,
                    help="Y-axis limits for temporal plot as 'ymin,ymax' (e.g., 0,1.2).")
parser.add_argument('--yquantile', type=float, default=0.995,
                    help="If --ylim not given, cap y-axis upper limit at this quantile of all shown curves.")
parser.add_argument('--yscale', default='log', choices=['linear', 'log', 'symlog'],
                    help="Axis scale.")
parser.add_argument('--event_row', type=float, default=0.03,
                    help="Relative vertical offset (0..1) from ymin for event dots.")


# font sizes
parser.add_argument('--label_size',      type=int, default=18)
parser.add_argument('--tick_size_major', type=int, default=15)
parser.add_argument('--tick_size_minor', type=int, default=15)
parser.add_argument('--legend_size',     type=int, default=15)

# event rug appearance (axes-fraction)
parser.add_argument('--event_base',   type=float, default=0.025, help='Baseline (0..1) of event rugs.')
parser.add_argument('--event_height', type=float, default=0.018, help='Height (0..1) of event rugs.')
parser.add_argument('--event_lw',     type=float, default=0.08,  help='Line width of event rugs.')


parser.add_argument('--spatial_types', default='0,3',
                    help='Two types to show on spatial maps (names or indices, comma separated).')
parser.add_argument('--temporal_types', default='0,1,2,3,',
                    help='Two-or-more types to show on temporal plots (names or indices, comma separated).')

parser.add_argument('--frame_factor', type=float, default=1.0)

args = parser.parse_args()

# Labels
TYPE_LABELS = ["TTP", "BRA", "BLA", "BLF"]


# Model / data setup
dataname = args.dataname
year_list = list(range(2008, 2020))
year_cur = year_list[args.sample_index]
day0 = dt.date(year_cur, 1, 1)  # 1-Jan

config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)

# checkpoint
n_comp = model_runner.model_wrapper.model.n_comps
n_epochs = 4000
n_times = 50
model_runner.model_wrapper.model.load_state_dict(torch.load(args.ckpt_dir, map_location='cpu'), strict=False)
model_runner.model_wrapper.model.eval()

# data
data_config = model_runner.runner_config.data_config
model_backend = model_runner.runner_config.base_config.backend
split = 'train'
data_dir = data_config.get_data_dir(split)
print(f"data_dir: {data_dir}")

data_loader = STPPDataLoader(data_config, model_backend)
data = data_loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)
tokenizer = STEventTokenizer(data_config.data_specs)
batch_size = 10
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size=batch_size, shuffle=False, space=True)

batch = next(iter(train_loader)).values()
time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch


# Sampling on a grid

ns = 40
ns2 = ns ** 2
num_sample_t = 1000
time_domain = 366     # days
model_runner.model_wrapper.model.spatial_npoints = ns

st_sample_list = model_runner.model_wrapper.model.make_multiple_equal_dtimespace_samples(
    time_seqs, [0, time_domain], space_seqs, num_sample_t=num_sample_t
)
st_sample_location = model_runner.model_wrapper.model.make_equal_dtimespace_samples(
    time_seqs, [0, time_domain], torch.zeros_like(space_seqs), num_sample_t=num_sample_t
)

input_ = time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, None, None, None
hiddens_ti, decays, output_states = model_runner.model_wrapper.model.forward(input_)

data_idx = args.sample_index
interval_t_samples = [st_sample_list[data_idx][i][:, 0, 0] - time_seqs[data_idx, i] for i in range(len(st_sample_list[data_idx]))]
interval_s_samples = [torch.norm(st_sample_list[data_idx][i][:, :, 1:], p=2, dim=-1) for i in range(len(st_sample_list[data_idx]))]

# spike guard
min_thres = 0.01
interval_t_samples_copy = interval_t_samples.copy()
for i, interval_t_sample in enumerate(interval_t_samples_copy):
    if not interval_t_sample.any():
        continue
    if interval_t_sample[0] <= min_thres:
        interval_t_sample[:ns ** 2] += min_thres
        interval_t_samples[i] = interval_t_sample
        print(f"Min threshold is added to {i} th interval_t_sample.")

# evaluate λ at sample points
state_t_samples = [
    model_runner.model_wrapper.model.compute_states_at_sample_points(
        decays[data_idx:(data_idx + 1), i:(i + 1)],
        output_states[data_idx:(data_idx + 1), i:(i + 1)],
        interval_t_sample, interval_s_sample
    )
    for i, (interval_t_sample, interval_s_sample) in enumerate(zip(interval_t_samples, interval_s_samples))
]

lambda_t_samples = [model_runner.model_wrapper.model.layer_intensity(s) for s in state_t_samples]
lambda_t_samples = [lt.detach().cpu().numpy()[0, 0] for lt in lambda_t_samples]
lambda_t_samples = np.concatenate(lambda_t_samples, axis=0)

sample_intensity = lambda_t_samples.copy()                       # shape: (frames*ns2, K)
st_sample = torch.cat(st_sample_location[data_idx], dim=0).cpu().numpy()

# fitted vs actual counts
fitted_points = sample_intensity.sum(axis=0) * time_domain / num_sample_t * (2 / ns) ** 2
expected_points = time_seqs[data_idx].argmax().item() + 1
actual_points = np.unique(type_seqs[data_idx, :expected_points].numpy(), return_counts=True)[1]
#print(f"Fitted_points:{fitted_points} \nActual_points:{actual_points}")


# Polygon (mask/area)
with open(args.polygon, 'rb') as f:
    pakistan_polygon = pickle.load(f)
pakistan_gdf = gpd.GeoSeries([pakistan_polygon])

if isinstance(pakistan_polygon, sg.Polygon):
    area = pakistan_polygon.area
else:
    area = sum(p.area for p in pakistan_polygon.geoms)
print('polygon bounds:', pakistan_polygon.bounds)


# Build per-sample DataFrame (generic K)

K = sample_intensity.shape[1]
TYPE_LABELS = TYPE_LABELS[:K]
df = pd.DataFrame({"t": st_sample[:, 0], "x": st_sample[:, 1], "y": st_sample[:, 2]})
df = pd.concat([df, pd.DataFrame(sample_intensity, columns=[f"intensity{k}" for k in range(K)])], axis=1)
df_np = df.to_numpy()
max_frame = len(df_np) // ns2
print("Grid:", ns, "×", ns, "| frames:", max_frame, "| K =", K)


# sequences (events)
seq_t = time_seqs[data_idx].numpy()
seq_s = space_seqs[data_idx].numpy()
point_types = type_seqs[data_idx].numpy()


# Utilities: type resolution

def _norm(s): return str(s).strip().lower()


def resolve_types(spec: str, labels: list, K: int, need_n: int = None):
    """
    spec = "0,2" or "TTP,BLA"
    returns list of distinct type indices, preserving order in spec.
    If need_n is not None, assert that exactly need_n were provided.
    """
    label2idx = {_norm(lbl): i for i, lbl in enumerate(labels)}
    out = []
    for token in [t for t in spec.split(',') if t.strip() != ""]:
        if token.strip().isdigit():
            idx = int(token.strip())
        else:
            key = _norm(token)
            if key not in label2idx:
                raise ValueError(f"Unknown type token '{token}'. Valid: {labels} or indices 0..{K-1}")
            idx = label2idx[key]
        if not (0 <= idx < K):
            raise ValueError(f"type index {idx} out of range [0,{K-1}]")
        if idx not in out:
            out.append(idx)
    if need_n is not None and len(out) != need_n:
        raise ValueError(f"Need exactly {need_n} types; got {out}")
    return out


spatial_types = resolve_types(args.spatial_types,  TYPE_LABELS, K, need_n=2)   # exactly two
temporal_types = resolve_types(args.temporal_types, TYPE_LABELS, K, need_n=None)  # two-or-more

# fixed marker per type
TYPE_MARKERS = {
    "TTP": "o",   # circle
    "BRA": "X",   # X
    "BLA": "D",   # diamond
    "BLF": "^",   # triangle
}

# high-contrast colors that stay visible on viridis/cividis/greys + black halo
# TYPE_COLORS = {
#     "TTP": "#ff7f00",  # orange
#     "BRA": "#e41a1c",  # red
#     "BLA": "#00c5ff",  # cyan
#     "BLF": "#c51bff",  # magenta
# }

TYPE_COLORS = {
    "TTP": "#ff7f00",  # orange
    "BRA": "#e41a1c",  # red
    "BLA": "#ffd92f",  # yellow/gold
    "BLF": "#c51bff",  # magenta
}

# def type_style(type_idx: int, size: float = 105.0) -> dict:
#     """Return a good scatter style for a given type index."""
#     lab = TYPE_LABELS[type_idx]
#     return dict(
#         marker=TYPE_MARKERS.get(lab, "o"),
#         s=size,
#         facecolor=TYPE_COLORS.get(lab, "black"),
#         edgecolor="white",
#         linewidth=1.4,
#         zorder=10,
#         # black halo makes markers readable on both dark and bright backgrounds
#         path_effects=[pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()],
#     )


def type_style(type_idx: int, size: float = 120.0) -> dict:
    lab = TYPE_LABELS[type_idx]
    return dict(
        marker=TYPE_MARKERS.get(lab, "o"),
        s=size,
        c=TYPE_COLORS.get(lab, "black"),
        edgecolors="none",   # <-- no border
        linewidths=0.0,      # <-- no border
        alpha=1.0,
        zorder=10,
    )


def nice_ticks(vmin: float, vmax: float, n_ticks: int = 6):
    # guards to avoid identical ticks
    vmin = float(vmin); vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax <= vmin:
        vmax = vmin * 1.000001 + 1e-12
    return np.linspace(vmin, vmax, int(n_ticks))




# Spatial map
# Spatial map
type1, type2 = spatial_types

# styles
styleA = type_style(type1, size=120.0)
styleB = type_style(type2, size=120.0)

print(f"Min/Max {TYPE_LABELS[type1]}:", df[f'intensity{type1}'].min(), df[f'intensity{type1}'].max())
print(f"Min/Max {TYPE_LABELS[type2]}:", df[f'intensity{type2}'].min(), df[f'intensity{type2}'].max())

savepath = f'{args.save_root}/{dataname}/{split}/{n_comp}_{n_epochs}'
if n_times != 50:
    savepath = f'{savepath}_{n_times}'
os.makedirs(savepath, exist_ok=True)


# trackers
def init_tracker(k):
    t_k = seq_t[point_types == k]
    s_k = seq_s[point_types == k]
    if t_k.size > 0:
        cur_t = t_k[0]
        nxt_t = t_k[1] if t_k.size > 1 else np.inf
        cur_loc = s_k[0]
        # first frame index where grid time >= cur_t
        start_i = int(np.searchsorted(df_np[:, 0][::ns2], cur_t, side='left'))
    else:
        cur_t = nxt_t = np.inf
        cur_loc = None
        start_i = max_frame + 1
    return {"times": t_k, "cur_t": cur_t, "nxt_t": nxt_t, "cur_loc": cur_loc, "start_i": start_i, "idx_ptr": 0}


trkA = init_tracker(type1)
trkB = init_tracker(type2)

colA = 3 + type1  # t,x,y are first 3 columns
colB = 3 + type2


N = df_np.shape[0]
pA_lo = pA_hi = pB_lo = pB_hi = None  # placeholders (unused)

# render subset of frames if requested
n_frames_to_draw = int(max_frame * max(0.0, min(1.0, args.frame_factor)))

print("Begin saving frames")

import matplotlib as mpl
from matplotlib import colors

# colors choice
# CMAP_NAME = "viridis"          # options: "cividis", "magma", "YlOrRd", "Greys", "viridis", "Blues"
USE_LOG = False              # True -> log color scale
# GAMMA = 0.7                # <1 boosts low values; >1 compresses lows
#
# cmap = mpl.colormaps[CMAP_NAME].copy()
# cmap.set_bad(color="white", alpha=0.0)


# RColorBrewer::brewer.pal(9, "Blues")
BREWER_BLUES_9 = [
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
    "#4292c6", "#2171b5", "#08519c", "#08306b"
]
cmap = ListedColormap(BREWER_BLUES_9, name="BrewerBlues")
cmap.set_bad(color="white", alpha=0.0)   # outside polygon = blank


for i in range(n_frames_to_draw):
    start, stop = i * ns2, (i + 1) * ns2
    if stop > N:
        break

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.8, wspace=0.18)

    x_block = df_np[start:stop, 1].reshape(ns, ns)
    y_block = df_np[start:stop, 2].reshape(ns, ns)

    inside_mask = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(df_np[start:stop, 1], df_np[start:stop, 2])],
        crs="EPSG:4326"
    ).within(pakistan_gdf.iloc[0]).values.reshape(ns, ns)

    frameA = df_np[start:stop, colA].reshape(ns, ns)
    frameB = df_np[start:stop, colB].reshape(ns, ns)

    mA = np.ma.masked_where(~inside_mask, frameA)
    mB = np.ma.masked_where(~inside_mask, frameB)

    # per-frame limits
    vminA, vmaxA = mA.min(), mA.max()
    vminB, vmaxB = mB.min(), mB.max()

    if USE_LOG:
        normA = colors.LogNorm(vmin=max(vminA, 1e-6), vmax=vmaxA)
        normB = colors.LogNorm(vmin=max(vminB, 1e-6), vmax=vmaxB)
    else:
        normA = colors.Normalize(vmin=vminA, vmax=vmaxA)
        normB = colors.Normalize(vmin=vminB, vmax=vmaxB)
        # normA = colors.PowerNorm(gamma=GAMMA, vmin=vminA, vmax=vmaxA)  # perceptual boost
        # normB = colors.PowerNorm(gamma=GAMMA, vmin=vminB, vmax=vmaxB)

    p0 = ax[0].pcolormesh(x_block, y_block, mA, shading='auto', cmap=cmap, norm=normA)
    p1 = ax[1].pcolormesh(x_block, y_block, mB, shading='auto', cmap=cmap, norm=normB)

    pakistan_gdf.boundary.plot(ax=ax[0], linewidth=1)
    pakistan_gdf.boundary.plot(ax=ax[1], linewidth=1)

    # current model time in days → calendar date
    t_cur = float(df_np[start, 0])
    cal_date = day0 + dt.timedelta(days=int(t_cur))
    date_label = cal_date.strftime("%#d %b")  # Windows; use "%-d %b" on mac/Linux

    # update trackers and draw markers (unchanged logic)
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
        ax[0].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], **styleA)
        ax[1].scatter(trkA["cur_loc"][0], trkA["cur_loc"][1], **styleA)

    if i >= trkB["start_i"] and trkB["cur_loc"] is not None:
        ax[0].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], **styleB)
        ax[1].scatter(trkB["cur_loc"][0], trkB["cur_loc"][1], **styleB)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.tick_params(axis='both', which='both', length=0)  # no tick marks
        # remove the box spines
        for s in a.spines.values():
            s.set_visible(False)

    n_ticks = 5
    ticksA = nice_ticks(vminA, vmaxA, n_ticks=n_ticks)
    ticksB = nice_ticks(vminB, vmaxB, n_ticks=n_ticks)

    cbar0 = fig.colorbar(p0, ax=ax[0], ticks=ticksA, fraction=0.046, pad=0.02, format="%.3f")
    cbar0.ax.tick_params(labelsize=18)

    cbar1 = fig.colorbar(p1, ax=ax[1], ticks=ticksB, fraction=0.046, pad=0.02, format="%.3f")
    cbar1.ax.tick_params(labelsize=18)

    fig.canvas.draw()  # ensure positions are up-to-date
    # union bbox of the two axes
    bbox0 = ax[0].get_position()
    bbox1 = ax[1].get_position()
    x_center = 0.5 * (min(bbox0.x0, bbox1.x0) + max(bbox0.x1, bbox1.x1))

    #fig.suptitle(date_label, fontsize=20, x=x_center, y=0.94)

    # fig.suptitle(date_label, fontsize=20, x=0.6, y=0.94)
    # ax[0].set_title(TYPE_LABELS[type1], fontsize=16, pad=6)
    # ax[1].set_title(TYPE_LABELS[type2], fontsize=16, pad=6)

    savename = f'{i:04d}'
    plt.savefig(f'{savepath}/{year_list[data_idx]}_{savename}.jpg', bbox_inches='tight')
    plt.close()

print("All frames saved")


def add_event_ticks_axesfrac(ax, times, color, base=0.07, height=0.04,
                             lw=1.2, alpha=0.95, zorder=3, capstyle='butt'):
    """
    Short vertical 'rug' lines whose height is specified in axes fraction.
    x is in data coords (time), y is in axes fraction (0..1).
    """
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


def _transformer(name: str, eps: float = 0.0, asinh_c: float = 0.25):
    # returns (func, label_suffix)
    if name == 'linear':
        return (lambda y: y, r'$\lambda(t)$')
    if name == 'sqrt':
        return (np.sqrt, r'$\sqrt{\lambda(t)}$')
    if name == 'log':
        return (lambda y: np.log(y + eps), r'$\log(\lambda(t))$')  # +\varepsilon
    if name == 'asinh':
        return (lambda y: np.arcsinh(y / asinh_c), r'$\mathrm{asinh}(\lambda/c)$')
    raise ValueError(name)


def _per_type_scale(x: np.ndarray, mode: str):
    # robust per-series scale for "shape view"
    if mode == 'none':
        return 1.0
    if mode == 'max':
        s = np.nanmax(x)
        return 1.0 if s <= 0 or not np.isfinite(s) else s
    if mode == 'mean':
        s = np.nanmean(x)
        return 1.0 if s <= 0 or not np.isfinite(s) else s
    if mode == 'q95':
        s = np.nanquantile(x, 0.95)
        return 1.0 if s <= 0 or not np.isfinite(s) else s
    raise ValueError(mode)


def plot_temporal(types_to_draw, title_tag, filename_tag):
    # time grid and unit area
    time_points = df["t"].to_numpy().reshape(-1, ns2)[:, 0]
    unit_area = (2.0 / ns) ** 2

    fig, ax = plt.subplots(figsize=(12, 5))
    import matplotlib.ticker as mticker

    # build series first so we can choose good y-lims
    series, events, labels = [], [], []
    for k in types_to_draw:
        Ik = df[f"intensity{k}"].to_numpy().reshape(-1, ns2)
        temporal_k = Ik.sum(axis=1) * unit_area
        series.append(temporal_k)
        events.append(seq_t[point_types == k])
        labels.append(TYPE_LABELS[k] if k < len(TYPE_LABELS) else f"type {k}")

    #  axis scale & limits (unchanged)
    ax.set_yscale(args.yscale, base=10)  # 'linear', 'log', or 'symlog'

    if args.ylim:
        ymin, ymax = map(float, args.ylim.split(','))
    else:
        all_vals = np.concatenate(series)
        ymin = 0.0 if args.yscale != 'log' else max(1e-6, float(np.nanmin(all_vals[all_vals > 0])))
        ymax = float(np.quantile(all_vals, args.yquantile))
        if not np.isfinite(ymax) or ymax <= ymin:
            ymax = float(np.nanmax(all_vals))
    if args.yscale == 'log':
        ymin = max(ymin, 1e-6)
        ymax = max(ymax, ymin * 1.001)

    # keep your hard cap (adjust as you wish)
    ax.set_ylim(ymin, (10**0.6) * 1.2)

    # ----- base-10 log ticks with numbers on BOTH major & minor -----
    if args.yscale == 'log':
        ax.set_yscale('log', base=10)
        # loc_maj = mticker.LogLocator(base=10, numticks=12)
        # ax.yaxis.set_major_locator(loc_maj)
        # loc_min = mticker.LogLocator(base=10, subs=np.arange(2, 8), numticks=12)
        # ax.yaxis.set_minor_locator(loc_min)
        # fmt_plain = mticker.FuncFormatter(lambda y, pos: f"{y:.3g}")
        # ax.yaxis.set_major_formatter(fmt_plain)
        # ax.yaxis.set_minor_formatter(fmt_plain)

        ticks = np.logspace(np.log10(ymin), np.log10((10**0.6) * 1.2), num=6)
        ax.set_yticks(ticks)
        ax.yaxis.set_minor_locator(mticker.NullLocator())  # hide minor tick clutter
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())  # show plain numbers

    #  plot curves + event rugs -----
    for (lab, yvals, t_k) in zip(labels, series, events):
        [line] = ax.plot(time_points, yvals, label=f"{lab}", linewidth=1.5)
        if t_k.size:
            # short vertical lines instead of scatter dots
            add_event_ticks_axesfrac(
                ax, t_k, color=line.get_color(),
                base=getattr(args, 'event_base', 0.025),
                height=getattr(args, 'event_height', 0.018),
                lw=getattr(args, 'event_lw', 0.8),
                alpha=0.95
            )

    ax.set_xlabel("Day of year", fontsize=getattr(args, 'label_size', 18))
    ax.set_ylabel("λ(t)",        fontsize=getattr(args, 'label_size', 18))

    # tick sizes (major + minor)
    ax.tick_params(axis='both', which='major', labelsize=getattr(args, 'tick_size_major', 15), length=6)
    #ax.tick_params(axis='both', which='minor', labelsize=getattr(args, 'tick_size_minor', 15),  length=3)

    ax.legend(loc="upper right", fontsize=getattr(args, 'legend_size', 15), frameon=False)
    #fig.tight_layout()
    fig.subplots_adjust(left=0.12, bottom=0.20, right=0.98, top=0.95)

    out = os.path.join(savepath, f"{year_list[args.sample_index]}_{filename_tag}.pdf")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved temporal → {out}")


#draw temporal for your chosen list (2+)
# if len(temporal_types) >= 2:
#     plot_temporal(temporal_types, "Selected", "temporal_selected")
# else:
#     print("[warn] temporal_types needs at least 2 types to be meaningful.")
