
# Resolve project imports and legacy paths before loading dependencies.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import os, sys, argparse, pickle, time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as dt

from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

from stpp_simulator.mstppg import BivariateHawkesLam
from stpp_simulator.stppg import SeparableExponentialKernel

data = "mstnhp_pakistan" #"mstnhp_pakistan"

#  CLI flags
parser = argparse.ArgumentParser()
parser.add_argument("--config_dir",    default="configs/experiment_config_gtd_pakistan.yaml")
parser.add_argument("--experiment_id", default="STNHP_eval")
parser.add_argument("--polygon",       default="./data/gtd_pakistan/pakistan_polygon_norm.pkl")
parser.add_argument("--sample_index",  type=int, default=6, help="index within first batch to visualise")
parser.add_argument("--ns",            type=int, default=40,  help="spatial lattice size (ns × ns)")
parser.add_argument("--num_t",         type=int, default=1000, help="#time samples")
parser.add_argument("--time_domain",   type=float, default=365)
parser.add_argument("--out_root",      default=f"../Meeting/MLE/2025/250909/{data}")
args = parser.parse_args()


year_list = list(range(2008, 2020))
year_cur = year_list[args.sample_index]          # e.g. 2014
day0 = dt.date(year_cur, 1, 1)      # 1‑Jan of that year


#  Hawkes parameters (MLE estimate)
#Nigeria params: updated ( sample_index= 5)
# mu = [0.719, 0.065]
# A = [1.208, -0.011, 0.535, 0.891]
# beta_t = [0.189, 0.982, 0.598, 0.463]
# beta_s = [4.551, 2.231, 0.028, 3.24]


# Nigeria params (2014 only)
# mu = [1.503, 1.039]
# A = [0.541, 0.421, -0.01, 0.657]
# beta_t = [0.079, 0.126, 0.692, 0.51]
# beta_s = [4.358, 1.857, 0.08, 3.039]

#Pakistan Params ( sample_index= 6)
# mu = [1.244, 1.032]
# A = [1.793, -0.015, 0.185, 1.203]
# beta_t = [0.137, 0.121, 1.800, 0.265]
# beta_s = [8.768, 1.751, 0.030, 9.004]

#Pakistan Params ( sample_index= 6) (where dx=dy=0.2, dt=0.1) loglik_value: 1.231199
mu = [0.03, 0.055]
A = [1.378, 1.419, -0.009, 0.764]
beta_t = [0.074, 2.392, 0.01, 0.07]
beta_s = [17.22, 23.241, 3.435, 11.838]

C = [round(a*bt*bs, 4) for a, bt, bs in zip(A, beta_t, beta_s)]
#C = A

kernels = [SeparableExponentialKernel(C=c, beta_t=bt, beta_s=bs)
           for c, bt, bs in zip(C, beta_t, beta_s)]

hawkes_intensity = BivariateHawkesLam(mu, kernels)


#  Evaluate λ grid fast enough for visualisation
def hawkes_lambda_grid(lam_obj,
                       seq_t, seq_s, seq_type,
                       t_grid, xy_grid):
    """
    Vectorised λ(t,x,y) on the Cartesian product of
        • t_grid  shape (N,)
        • xy_grid shape (M,2)

    Returns
    -------
    lam : ndarray, shape (N*M, 2)
          Flattened in row‑major order: first vary x‑y, then time.
    """
    N, M = len(t_grid), len(xy_grid)
    K = 2                                  # two target types
    lam = np.zeros((N, M, K), dtype=float)   # we’ll flatten at the end


    # Pre‑split history by source type to minimise Python overhead

    idx_type0 = np.where(seq_type == 0)[0]
    idx_type1 = np.where(seq_type == 1)[0]

    # organise kernels for easy lookup: kernels[k][l] = g_{k,l}
    kernels = lam_obj.kernel                   # already [[00,01],[10,11]]


    # Loop over history events, vectorise over lattice

    for l, idxs in enumerate([idx_type0, idx_type1]):       # source type l
        if idxs.size == 0:
            continue

        his_t = seq_t[idxs]                  # (H_l,)
        his_s = seq_s[idxs]                  # (H_l, 2)

        # distance from each lattice space‑point to each event
        #   ds  shape (M, H_l)
        dx = xy_grid[:, 0, None] - his_s[:, 0]   # broadcast
        dy = xy_grid[:, 1, None] - his_s[:, 1]
        ds = np.sqrt(dx*dx + dy*dy)

        for k in (0,1):                       # target type k
            ker = kernels[k][l]
            # temporal kernel factor for every t and every event
            #   dt_exp shape (N, H_l)
            dt = t_grid[:, None] - his_t[None, :]
            dt_exp = np.where(dt > 0, ker.C * np.exp(-ker.beta_t * dt), 0.0)

            # spatial kernel factor   shape (M, H_l)
            ds_exp = np.exp(-ker.beta_s * ds)

            # outer product over history events → accumulate into λ_k
            # Each event contributes dt_exp (N,H) * ds_expᵀ (H,M)
            # einsum broadcasts without allocating a full (N,M,H) tensor

            lam[:, :, k] += np.einsum('nh,hm->nm', dt_exp, ds_exp.T)

    # add the baselines μ_k
    lam[:, :, 0] += lam_obj.mu[0]
    lam[:, :, 1] += lam_obj.mu[1]

    return lam.reshape(N*M, K)


# Load polygon for masking
with open(args.polygon, "rb") as f:
    poly = pickle.load(f)
poly_gdf = gpd.GeoSeries([poly])


# Load one batch from EasyTPP train split

cfg = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
runner = Runner.build_from_config(cfg)
data_cfg = runner.runner_config.data_config
backend = runner.runner_config.base_config.backend

train_pkl = data_cfg.get_data_dir("train")
loader = STPPDataLoader(data_cfg, backend)
dataset = STPPDataset(loader.build_input_from_pkl(train_pkl, "train"))
tok = STEventTokenizer(data_cfg.data_specs)
dloader = get_data_loader(dataset, backend, tok,
                             batch_size=10, shuffle=False, space=True)

batch = next(iter(dloader)).values()
time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, *_ = batch

i = args.sample_index
seq_t = time_seqs[i].cpu().numpy()
seq_s = space_seqs[i].cpu().numpy()
seq_type = type_seqs[i].cpu().numpy()


#  Space‑time lattice
ns = args.ns
time_domain = args.time_domain
num_sample_t = args.num_t

t_grid = np.linspace(0, time_domain, num_sample_t, dtype=float)
x = y = np.linspace(-1, 1, ns, dtype=float)
xx, yy = np.meshgrid(x, y, indexing="ij")
xy_grid = np.column_stack([xx.ravel(), yy.ravel()])       # (ns², 2)


from shapely.geometry import Point
# if GeoPandas already imported, fine
mask_flat = gpd.GeoSeries([Point(x,y) for x,y in xy_grid]).within(poly).values
xy_grid_in  = xy_grid[mask_flat]                 # keep only inside points
M_in = len(xy_grid_in)                    # #grid cells inside polygon
unit_area = poly.area / M_in                   # replaces the square formula



print("Evaluating λ on a grid of {}×{}×{} …".format(num_sample_t, ns, ns))
tic = time.time()
lam_grid = hawkes_lambda_grid(hawkes_intensity,
                              seq_t, seq_s, seq_type,
                              t_grid, xy_grid_in)
print("done in {:.1f}s".format(time.time() - tic))


# Visualization

#  Build DataFrame
sample_intensity = lam_grid

st_sample = np.column_stack([
        np.repeat(t_grid, M_in),              # t
        np.tile(xy_grid_in[:,0], len(t_grid)),  # x
        np.tile(xy_grid_in[:,1], len(t_grid))   # y
])


# temporal intensity plotting
df_temp = pd.DataFrame({"t": st_sample[:, 0],
                   "x": st_sample[:, 1],
                   "y": st_sample[:, 2],
                   "intensity0": sample_intensity[:, 0],
                   "intensity1": sample_intensity[:, 1]})




# Output folders
frame_dir = os.path.join(args.out_root, "frames")
os.makedirs(frame_dir, exist_ok=True)
plot_dir = args.out_root
os.makedirs(plot_dir, exist_ok=True)


# Area
#ns2 = ns*ns
#unit_area = 2.0**2 / ns2
ns2 = M_in

# pre‑compute polygon mask on spatial lattice
pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy_grid], crs="EPSG:4326")
mask = pts.within(poly_gdf.iloc[0]).values.reshape(ns, ns)


# Build per‑lattice DataFrame (df, df_np)
# st_sample = np.column_stack([
#         np.repeat(t_grid, ns2),          # t  (num_t*ns²,)
#         np.tile(xy_grid[:, 0], len(t_grid)),  # x
#         np.tile(xy_grid[:, 1], len(t_grid)),  # y
# ])

st_sample = np.column_stack([
        np.repeat(t_grid, M_in),              # t
        np.tile(xy_grid_in[:,0], len(t_grid)),# x
        np.tile(xy_grid_in[:,1], len(t_grid)) # y
])


# sample_intensity comes directly from lam_grid
sample_intensity = lam_grid                           # (num_t*ns², 2)

# DataFrame with columns: t, x, y, intensity0, intensity1
df = pd.DataFrame({'t': st_sample[:, 0],
                   'x': st_sample[:, 1],
                   'y': st_sample[:, 2]})
intensity_df = pd.DataFrame(
                   sample_intensity,
                   columns=[f'intensity{i}'
                            for i in range(sample_intensity.shape[1])])
df = pd.concat([df, intensity_df], axis=1)
df_np = df.to_numpy()                                  # NumPy view
max_frame = len(df_np) // ns2


# Per‑sequence event data (seq_t, seq_s, point_types, …)
data_idx = args.sample_index
seq_t = time_seqs[data_idx].cpu().numpy()
seq_s = space_seqs[data_idx].cpu().numpy()
point_types = type_seqs[data_idx].cpu().numpy()

current_time = [seq_t[point_types == 0][0],
                seq_t[point_types == 1][0]]
next_time = [seq_t[point_types == 0][1]
                if (point_types == 0).sum() > 1 else args.time_domain+1,
                seq_t[point_types == 1][1]
                if (point_types == 1).sum() > 1 else args.time_domain+1]
current_loc = [seq_s[point_types == 0][0],
                seq_s[point_types == 1][0]]

start_is = [
    int(np.where(df_np[:,0] >= current_time[0])[0][0] / ns2),
    int(np.where(df_np[:,0] >= current_time[1])[0][0] / ns2),
]


# Spatial lattice (xx, yy)

x = y = np.linspace(-1, 1, ns, dtype=float)
xx, yy = np.meshgrid(x, y, indexing='ij')      # shapes (ns,ns)

# GeoSeries for the country outline
poly_gdf = gpd.GeoSeries([poly])

# mask: Boolean array (ns²,) showing which lattice points lie inside polygon
xy_points = gpd.GeoDataFrame(
               geometry=[Point(x, y) for x, y in xy_grid], crs="EPSG:4326")
mask_flat = xy_points.within(poly_gdf.iloc[0]).values          # (ns²,)
mask = mask_flat.reshape(ns, ns)                          # (ns,ns)


# Spatial heat maps (one PNG per frame)
type1_color = "cyan"     # observed events of type 0
type2_color = "magenta"  # observed events of type 1

# Polygon mask on the spatial lattice
pts = gpd.GeoDataFrame(
                   geometry=[Point(x, y) for x, y in xy_grid], crs="EPSG:4326")
mask_flat = pts.within(poly_gdf.iloc[0]).values          # (ns²,)
pakistan_mask = mask_flat.reshape(ns, ns)                    # (ns,ns)


frame_range = range(0, 2*max_frame//5)

print("Writing spatial frames …")
for i in frame_range:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Intensities for this frame
    # frame0 = df_np[i*ns2:(i+1)*ns2, -2].reshape(ns, ns)
    # frame1 = df_np[i*ns2:(i+1)*ns2, -1].reshape(ns, ns)

    # full ns×ns array initialised with NaN
    frame0 = np.full((ns, ns), np.nan)
    frame1 = np.full((ns, ns), np.nan)

    # fill inside‑polygon cells; pakistan_mask is Boolean (ns,ns)
    frame0[pakistan_mask] = df_np[i * M_in:(i + 1) * M_in, -2]
    frame1[pakistan_mask] = df_np[i * M_in:(i + 1) * M_in, -1]

    masked0 = np.ma.masked_where(~pakistan_mask, frame0)
    masked1 = np.ma.masked_where(~pakistan_mask, frame1)

    # Per‑frame dynamic colour limits
    vmin0, vmax0 = masked0.min(), masked0.max()
    vmin1, vmax1 = masked1.min(), masked1.max()

    # Draw heat‑maps
    p0 = ax[0].pcolormesh(xx, yy, masked0,
                          shading="auto", cmap="coolwarm",
                          vmin=vmin0, vmax=vmax0)
    p1 = ax[1].pcolormesh(xx, yy, masked1,
                          shading="auto", cmap="coolwarm",
                          vmin=vmin1, vmax=vmax1)

    # Country outline
    poly_gdf.boundary.plot(ax=ax[0], linewidth=1)
    poly_gdf.boundary.plot(ax=ax[1], linewidth=1)

    # Scatter current events
    if df_np[i*ns2, 0] > next_time[0]:
        cur_t = next_time[0]
        current_time[0] = cur_t
        cur_idx = np.where(seq_t == cur_t)[0][0]
        future_0 = seq_t[(point_types == 0) & (seq_t > cur_t)]
        next_time[0] = future_0[0] if future_0.size else args.time_domain + 1
        current_loc[0] = seq_s[cur_idx]

    if df_np[i*ns2, 0] > next_time[1]:
        cur_t = next_time[1]
        current_time[1] = cur_t
        cur_idx = np.where(seq_t == cur_t)[0][0]
        future_1 = seq_t[(point_types == 1) & (seq_t > cur_t)]
        next_time[1] = future_1[0] if future_1.size else args.time_domain + 1
        current_loc[1] = seq_s[cur_idx]

    if i >= start_is[0]:
        ax[0].scatter(*current_loc[0], color=type1_color, s=20)
        ax[1].scatter(*current_loc[0], color=type1_color, s=20)
    if i >= start_is[1]:
        ax[0].scatter(*current_loc[1], color=type2_color, s=20)
        ax[1].scatter(*current_loc[1], color=type2_color, s=20)

    # Separate colour‑bars
    plt.colorbar(p0, ax=ax[0], location="right", format="%.2f")
    plt.colorbar(p1, ax=ax[1], location="right", format="%.2f")

    #  current model time in days (float)
    t_cur = df_np[i * ns2, 0]

    #  convert to calendar date (‘int’ keeps Feb=28 assumption)
    cal_date = day0 + dt.timedelta(days=int(t_cur))

    #  formatted label, e.g.  2 May  or  02 Jan
    #date_label = cal_date.strftime("%-d %b")  # Linux / macOS
    date_label = cal_date.strftime("%#d %b")  # Windows

    plt.suptitle(f"{date_label}", fontsize=12)

    #plt.suptitle(f"time: {df_np[i*ns2, 0]:.2f}", fontsize=12)

    # Consistent filename ordering (frame_0000.png …)
    fig.savefig(os.path.join(frame_dir, f"frame_{i:04d}.png"),
                bbox_inches="tight")
    plt.close(fig)

print("Frames saved to", frame_dir)


#temporal intensity
time_points = df_temp.t.values.reshape(-1, ns2)[:, 0]
intensity_0 = df_temp.intensity0.values.reshape(-1, ns2).sum(1) * unit_area
intensity_1 = df_temp.intensity1.values.reshape(-1, ns2).sum(1) * unit_area

plt.figure(figsize=(15, 6))
plt.scatter(seq_t[seq_type == 0], np.zeros_like(seq_t[seq_type == 0]),
            s=10)
plt.scatter(seq_t[seq_type == 1], np.zeros_like(seq_t[seq_type == 1]),
            s=10, color="orange")
plt.plot(time_points, intensity_0,  label="Taliban-TTP")  # Taliban-TTP BH
plt.plot(time_points, intensity_1, color="orange",  label="Baloch")  # Baloch FE
plt.legend()
#plt.ylim(None, 3.0)
plt.savefig(os.path.join(plot_dir, "mle_temporal_intensity.jpg"),
            bbox_inches="tight")
plt.show()

print("Temporal plot saved to", plot_dir)



# ================================================================
# TEMPORALLY‑AGGREGATED SPATIAL PROFILES  (MLE Hawkes)
# ================================================================
# unit_time = t_grid[1] - t_grid[0]
#
# print("Building spatial 1‑D profiles…")
#
# # 1. cell geometry (inside‑polygon grid only)
# cell_A = poly.area / M_in            # each inside cell’s physical area
#
# # 2. integrals over time  ->  expected *events* per cell
# events_cell0 = (sample_intensity[:,0]          # λ₀(t,x,y) flattened
#                 .reshape(-1, M_in)             # (num_t, M_in)
#                 .sum(axis=0) * unit_time * cell_A)     # (M_in,)
# events_cell1 = (sample_intensity[:,1]
#                 .reshape(-1, M_in)
#                 .sum(axis=0) * unit_time * cell_A)
#
# # 3. DataFrame with cell coordinates and expected counts
# space_df = pd.DataFrame({
#     "x" : st_sample[:M_in, 1],   # same x,y repeat every time‑slice
#     "y" : st_sample[:M_in, 2],
#     "ev0": events_cell0,
#     "ev1": events_cell1
# })
#
# # ------------------------------------------------
# # 3a. East–West profile (collapse y)
# # ------------------------------------------------
# x_vals = np.sort(space_df["x"].unique())
# lambda0_x = space_df.groupby("x")["ev0"].sum().reindex(x_vals).values
# lambda1_x = space_df.groupby("x")["ev1"].sum().reindex(x_vals).values
#
# # plt.figure(figsize=(15, 6))
# # plt.plot(x_vals, lambda0_x, label="Type 0 (TTP/BH)")
# # plt.plot(x_vals, lambda1_x, label="Type 1 (Baloch/FE)", color="orange")
# # plt.xlabel("x (normalised longitude)")
# # plt.ylabel("expected events in 366 days\nper unit‑x strip")
# # plt.title("Temporally aggregated spatial profile (east–west)")
# # plt.legend(); plt.tight_layout()
# # plt.savefig(os.path.join(plot_dir, "mle_spatial_intensity_x.jpg"),
# #             bbox_inches="tight")
# # plt.show()
# # plt.close()
#
# # ------------------------------------------------
# # 3b. North–South profile (collapse x)
# # ------------------------------------------------
# y_vals = np.sort(space_df["y"].unique())
# lambda0_y = space_df.groupby("y")["ev0"].sum().reindex(y_vals).values
# lambda1_y = space_df.groupby("y")["ev1"].sum().reindex(y_vals).values
#
# # plt.figure(figsize=(7,4))
# # plt.plot(y_vals, lambda0_y, label="Type 0 (TTP/BH)")
# # plt.plot(y_vals, lambda1_y, label="Type 1 (Baloch/FE)", color="orange")
# # plt.xlabel("y (normalised latitude)")
# # plt.ylabel("expected events in 366 days\nper unit‑y strip")
# # plt.title("Temporally aggregated spatial profile (north–south)")
# # plt.legend(); plt.tight_layout()
# # plt.savefig(os.path.join(plot_dir, "mle_spatial_intensity_y.jpg"),
# #             bbox_inches="tight")
# # plt.show()
# # plt.close()
#
# print("Spatial profiles saved to", plot_dir)
#
#
#
#
#
# # diagnostics
# unit_time = t_grid[1] - t_grid[0]          # Δt  (days)
#
# exp_counts_mle = sample_intensity.sum(axis=0) * unit_area * unit_time
# obs_counts = np.bincount(point_types, minlength=2)       # [type‑0, type‑1]
#
#
# inside_idx = mask_flat          # Boolean of shape (ns²,)
# unit_area_poly = poly.area / inside_idx.sum()
#
# expected = lam_grid.sum(axis=0) * unit_area_poly * unit_time
# observed = np.bincount(point_types, minlength=2)
# print(f"Expected: {expected}  \nObserved: {observed}")
#
#
#
#
#
# # ================================================================
# #  ADD DEGREE TICK‑LABELS TO SPATIAL PROFILES (EW & NS)
# # ================================================================
#
# # ── 0.  ORIGINAL geographic bounds (replace with your values) ─────
#
# LON_MIN, LON_MAX = 60,  78                   # °E
# LAT_MIN, LAT_MAX = 23,  38                    # °N
#
#
# # ── 1. Vectorised transforms  [-1,1] ↔ degrees  ---------------------
# def xnorm_to_lon(xn):                      # xn can be scalar *or* array
#     return LON_MIN + 0.5 * (xn + 1.0) * (LON_MAX - LON_MIN)
#
# def lon_to_xnorm(lon):
#     return 2 * (lon - LON_MIN) / (LON_MAX - LON_MIN) - 1
#
# def ynorm_to_lat(yn):
#     return LAT_MIN + 0.5 * (yn + 1.0) * (LAT_MAX - LAT_MIN)
#
# def lat_to_ynorm(lat):
#     return 2 * (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1
#
#
#
# # ------------- east–west profile -----------------------------------
# # fig, ax = plt.subplots(figsize=(15, 6))
# # ax.plot(x_vals, lambda0_x, label="Type 0")
# # ax.plot(x_vals, lambda1_x, label="Type 1", color="orange")
# # ax.set_xlabel("x (normalised)")
# # ax.set_ylabel("expected events / 366 days / x‑strip")
# # ax.set_title("Temporally aggregated spatial profile  (east–west)")
# # ax.legend()
# #
# # # bottom ticks labelled in degrees
# # tick_idx   = np.linspace(0, len(x_vals)-1, 6, dtype=int)
# # x_tick_pos = x_vals[tick_idx]
# # ax.set_xticks(x_tick_pos)
# # ax.set_xticklabels([f"{lon:.1f}" for lon in xnorm_to_lon(x_tick_pos)])
# #
# # # secondary longitude axis
# # secax = ax.secondary_xaxis("top",
# #             functions=(xnorm_to_lon, lon_to_xnorm))
# # secax.set_xlabel("longitude (°E)")
# #
# # fig.tight_layout()
# # fig.savefig(os.path.join(plot_dir,
# #             f"mle_spatial_profile_EW_deg.png"),
# #             bbox_inches="tight")
# # plt.show(); plt.close()
# #
# #
# # # ------------- north–south profile ---------------------------------
# # fig, ax = plt.subplots(figsize=(15, 6))
# # ax.plot(y_vals, lambda0_y, label="Type 0")
# # ax.plot(y_vals, lambda1_y, label="Type 1", color="orange")
# # ax.set_xlabel("y (normalised)")
# # ax.set_ylabel("expected events / 366 days / y‑strip")
# # ax.set_title("Temporally aggregated spatial profile  (north–south)")
# # ax.legend()
# #
# # tick_idx   = np.linspace(0, len(y_vals)-1, 6, dtype=int)
# # y_tick_pos = y_vals[tick_idx]
# # ax.set_xticks(y_tick_pos)
# # ax.set_xticklabels([f"{lat:.1f}" for lat in ynorm_to_lat(y_tick_pos)])
# #
# # secax = ax.secondary_xaxis("top",
# #             functions=(ynorm_to_lat, lat_to_ynorm))
# # secax.set_xlabel("latitude (°N)")
# #
# # fig.tight_layout()
# # fig.savefig(os.path.join(plot_dir,
# #             f"mle_spatial_profile_NS_deg.png"),
# #             bbox_inches="tight")
# # plt.show(); plt.close()
# #
# # print("Spatial profiles with degree labels saved to", plot_dir)
# #
# #
# # # integral vs expected counts
# # total_x0 = lambda0_x.sum()
# # total_x1 = lambda1_x.sum()
# # total_y0 = lambda0_y.sum()
# #
# # print(f"x‑profile totals  : {total_x0:.1f}  {total_x1:.1f}")
# # print(f"y‑profile total 0 : {total_y0:.1f}")
# # print(f"Expected counts   : {expected}")
# # print(f"Observed counts   : {observed}")
