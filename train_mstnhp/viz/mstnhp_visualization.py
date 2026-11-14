
# Resolve project imports and legacy paths before loading dependencies.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import os, argparse, pickle, numpy as np, torch, matplotlib.pyplot as plt

import shapely.geometry as sg

from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader
import datetime as dt


parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml',
                    help='Dir of configuration yaml to train and evaluate the model. ')

parser.add_argument('--experiment_id', type=str, required=False, default='STNHP_eval',
                    help='Experiment id in the config file.')

parser.add_argument('--ckpt_dir', type=str, required=False, default='./checkpoints/Nigeria_checkers/79448_22566429611008_250524-064729/models/saved_model',
                        help='Path to model')

parser.add_argument('--data_path', type=str, required=False, default='./data/mstnhp_nigeria/train.pkl',
                        help='Path to dataset file')

parser.add_argument('--polygon',      required=False,    default='./data/gtd_pakistan/nigeria_polygon_norm.pkl',
                       help='pickled Shapely Polygon')

parser.add_argument('--sample_index', type=int, required=False, default=5, help='Index of the sample to evaluate')


parser.add_argument('--save_root',    default='../Meeting/2025/250731',
                    help='root where dataname/… folder tree is created')

parser.add_argument('--dataname',     default='mstnhp_nigeria')
args = parser.parse_args()



# with open('./data/gtd_pakistan/pakistan_polygon.pkl', 'rb') as f:
#     polygon = pickle.load(f)

## HJ: Define model and data configurations
# dataname = 'bistpp_separable_zhang'
dataname = 'mstnhp_pakistan'
year_list = list(range(2008, 2020))
year_cur = year_list[args.sample_index]          # e.g. 2014
day0 = dt.date(year_cur, 1, 1)      # 1‑Jan of that year


#config_suffix = dataname.split('_')[0]
#model_name = 'STNHP'
#experiment_name = model_name + '_train'

config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
## HJ: Create a model runner from the configuration and train the model
model_runner = Runner.build_from_config(config)

## HJ: Load model from the checkpoint
n_comp = model_runner.model_wrapper.model.n_comps
n_epochs = 2000
n_times = 50
#model_name = f'{model_name}_{n_comp}_{n_times}_{n_epochs}'

model_runner.model_wrapper.model.load_state_dict(torch.load(args.ckpt_dir, map_location='cpu'),
                                           strict=False)


# model_runner.model_wrapper.model.load_state_dict(
#     torch.load('checkpoints/' + dataname + '/' + model_name, map_location='cpu'))

model_runner.model_wrapper.model.eval()

## HJ: get data and model configurations
data_config = model_runner.runner_config.data_config
model_backend = model_runner.runner_config.base_config.backend

## HJ: Load train dataset
split = 'train'
data_dir = data_config.get_data_dir(split)

print(f"data_dir: {data_dir}")

data_source_type = data_dir.split('.')[-1]
data_loader = STPPDataLoader(data_config, model_backend)  # data_loader = TPPDataLoader(data_config, model_backend)
data = data_loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)  # dataset = TPPDataset(data)
sample_size = len(dataset)
tokenizer = STEventTokenizer(data_config.data_specs)  # tokenizer = EventTokenizer(data_config.data_specs)
batch_size = 10  # 13

## HJ: Load train dataloader
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size=batch_size, shuffle=False, space=True)

## HJ: get batch data
batch = next(iter(train_loader))
batch = batch.values()
time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch

## HJ: evaluate conditional intensity at equally-spaced space-time grid points
ns = 40
num_sample_t = 1000
time_domain = 366
model_runner.model_wrapper.model.spatial_npoints = ns
st_sample_list = model_runner.model_wrapper.model.make_multiple_equal_dtimespace_samples(time_seqs, [0, time_domain],
                                                                                         space_seqs,
                                                                                         num_sample_t=num_sample_t)
st_sample_location = model_runner.model_wrapper.model.make_equal_dtimespace_samples(time_seqs, [0, time_domain],
                                                                                    torch.zeros_like(space_seqs),
                                                                                    num_sample_t=num_sample_t)



# st_sample_location[0][28][6400:8000,0] = 26.60
input_ = time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, None, None, None
hiddens_ti, decays, output_states = model_runner.model_wrapper.model.forward(input_)

## HJ: Select data index for evaluation
data_idx = args.sample_index
# for data_idx in range(batch_size):
# year = year_list[data_idx]
interval_t_samples = [st_sample_list[data_idx][i][:, 0, 0] - time_seqs[data_idx, i] for i in
                      range(len(st_sample_list[data_idx]))]
interval_s_samples = [torch.norm(st_sample_list[data_idx][i][:, :, 1:], p=2, dim=-1) for i in
                      range(len(st_sample_list[data_idx]))]

## HJ: Add minimum threshold to interval_t_samples to avoid peaks.
min_thres = .01
interval_t_samples_copy = interval_t_samples.copy()
for i, interval_t_sample in enumerate(interval_t_samples_copy):
    if not interval_t_sample.any():
        continue
    first_t_sample = interval_t_sample[0]
    if first_t_sample <= min_thres:
        interval_t_sample[:ns ** 2] += min_thres
        interval_t_samples[i] = interval_t_sample
        print(f"Min threshold is added to {i} th interval_t_sample.")

## HJ: evaluate conditional intensity at interval_t_samples and interval_s_samples
state_t_samples = [
    model_runner.model_wrapper.model.compute_states_at_sample_points(decays[data_idx:(data_idx + 1), i:(i + 1)],
                                                                     output_states[data_idx:(data_idx + 1), i:(i + 1)],
                                                                     interval_t_sample, interval_s_sample)
    for i, (interval_t_sample, interval_s_sample) in enumerate(zip(interval_t_samples, interval_s_samples))]

## HJ: get intensity samples
lambda_t_samples = [model_runner.model_wrapper.model.layer_intensity(state_t_sample) for state_t_sample in
                    state_t_samples]
lambda_t_samples = [lambda_t_sample.detach().cpu().numpy()[0, 0] for lambda_t_sample in lambda_t_samples]
lambda_t_samples = np.concatenate(lambda_t_samples, axis=0)
lambda_t_samples.shape

sample_intensity = lambda_t_samples.copy()
st_sample = torch.cat(st_sample_location[data_idx], dim=0).cpu().numpy()

## HJ: Compare the number of fitted points and the number of actual points
fitted_points = sample_intensity.sum(axis=0) * time_domain / num_sample_t * (2 / ns) ** 2
expected_points = time_seqs[data_idx].argmax().item() + 1
actual_points = np.unique(type_seqs[data_idx, :expected_points].numpy(), return_counts=True)[1]
print(f"Fitted_points:{fitted_points} \nActual_points:{actual_points}")

## HJ: Get quantiles of sample intensity to determine ranges for intensity plots.
np.quantile(sample_intensity, [.001, 0.01, 0.9, 0.99, 0.999], axis=0)
np.min(sample_intensity, axis=0), np.max(sample_intensity, axis=0)

## HJ: Load polygon
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

with open(args.polygon, 'rb') as f:
    pakistan_polygon = pickle.load(f)
pakistan_gdf = gpd.GeoSeries([pakistan_polygon])

# ------ 1.  true spatial area -----------------
if isinstance(pakistan_polygon, sg.Polygon):
    area = pakistan_polygon.area
else:                               # MultiPolygon
    area = sum(p.area for p in pakistan_polygon.geoms)
print('polygon bounds:', pakistan_polygon.bounds)

## Visualization
# cmap = plt.get_cmap('coolwarm')
cmap = plt.get_cmap('viridis')
ns = model_runner.model_wrapper.model.spatial_npoints
ns2 = ns ** 2
# ns2 = model_runner.model_wrapper.model.inside_polygon_index.shape[0]
df = pd.DataFrame({'t': st_sample[:, 0], 'x': st_sample[:, 1], 'y': st_sample[:, 2]})
intensity_df = pd.DataFrame(sample_intensity, columns=[f'intensity{i}' for i in range(sample_intensity.shape[1])])
df = pd.concat([df, intensity_df], axis=1)
df_np = df.to_numpy()
savepath = os.path.join(args.save_root, dataname, split, f'{n_comp}_{n_epochs}')
if n_times != 50:
    savepath = f'{savepath}_{n_times}'
os.makedirs(savepath, exist_ok=True)
max_frame = len(df_np) // ns2

seq_t = time_seqs[data_idx].numpy()
seq_s = space_seqs[data_idx].numpy()
point_types = type_seqs[data_idx].numpy()

current_time = [seq_t[point_types == 0][0], seq_t[point_types == 1][0]]
next_time = [seq_t[point_types == 0][1], seq_t[point_types == 1][1]]
current_loc = [seq_s[point_types == 0][0], seq_s[point_types == 1][0]]
start_is = [int(np.where(df_np[:, 0] >= current_time[0])[0][0] / ns2),
            int(np.where(df_np[:, 0] >= current_time[1])[0][0] / ns2)]

print("Min/Max Type0:", df['intensity0'].min(), df['intensity0'].max())
print("Min/Max Type1:", df['intensity1'].min(), df['intensity1'].max())


type1_color = 'cyan'  #'black'
type2_color = 'magenta' #'yellow'

for i in range(0, 2 * max_frame // 5):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))  # (10,4)
    xy_points = [Point(x, y) for x, y in zip(df_np[i * ns2:(i + 1) * ns2, 1], df_np[i * ns2:(i + 1) * ns2, 2])]
    gdf_points = gpd.GeoDataFrame(geometry=xy_points, crs="EPSG:4326")
    gdf_within = gdf_points.within(pakistan_gdf.iloc[0])

    # before:
    # pakistan_mask = gdf_within.values.reshape(df_np[i * ns2:(i + 1) * ns2, 1].shape)

    # instead do:
    flat_mask = gdf_within.values  # shape (ns2,)
    pakistan_mask = flat_mask.reshape(ns, ns)  # now shape (ns, ns)

    #pakistan_mask = gdf_within.values.reshape(df_np[i * ns2:(i + 1) * ns2, 1].shape)
    #masked_intensity0 = np.ma.masked_where(~pakistan_mask, df_np[i * ns2:(i + 1) * ns2, -2])
    #masked_intensity1 = np.ma.masked_where(~pakistan_mask, df_np[i * ns2:(i + 1) * ns2, -1])

    frame0 = df_np[i * ns2:(i + 1) * ns2, -2].reshape(ns, ns)
    frame1 = df_np[i * ns2:(i + 1) * ns2, -1].reshape(ns, ns)

    # mask out-of-polygon points
    masked0 = np.ma.masked_where(~pakistan_mask, frame0)
    masked1 = np.ma.masked_where(~pakistan_mask, frame1)

    # compute per-frame vmin/vmax
    vmin0, vmax0 = masked0.min(), masked0.max()
    vmin1, vmax1 = masked1.min(), masked1.max()

    # plot with dynamic vmin/vmax
    p0 = ax[0].pcolormesh(
        df_np[i * ns2:(i + 1) * ns2, 1].reshape(ns, ns),
        df_np[i * ns2:(i + 1) * ns2, 2].reshape(ns, ns),
        masked0,
        shading='auto', cmap='coolwarm',
        vmin=vmin0, vmax=vmax0
    )
    p1 = ax[1].pcolormesh(
        df_np[i * ns2:(i + 1) * ns2, 1].reshape(ns, ns),
        df_np[i * ns2:(i + 1) * ns2, 2].reshape(ns, ns),
        masked1,
        shading='auto', cmap='coolwarm',
        vmin=vmin1, vmax=vmax1
    )


    # p0 = ax[0].pcolormesh(df_np[i * ns2:(i + 1) * ns2, 1].reshape(ns, -1).round(4),
    #                       df_np[i * ns2:(i + 1) * ns2, 2].reshape(ns, -1).round(4), masked_intensity0.reshape(ns, -1),
    #                       shading='auto', cmap=plt.cm.coolwarm, vmin=0.028, vmax=0.4)
    # p1 = ax[1].pcolormesh(df_np[i * ns2:(i + 1) * ns2, 1].reshape(ns, -1).round(4),
    #                       df_np[i * ns2:(i + 1) * ns2, 2].reshape(ns, -1).round(4), masked_intensity1.reshape(ns, -1),
    #                       shading='auto', cmap=plt.cm.coolwarm, vmin=0.017, vmax=0.12)
    #


    pakistan_gdf.boundary.plot(ax=ax[0], linewidth=1)
    pakistan_gdf.boundary.plot(ax=ax[1], linewidth=1)

    # p0=ax[0].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -2].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.05, vmax = 0.6)
    # p1=ax[1].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -1].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.05, vmax = 0.6)

    ## HJ: Update current time & loc and next time & loc for plotting actual events on intensity map
    if df_np[i * ns2, 0] > next_time[0]:
        temp_time = next_time[0]
        current_time[0] = temp_time
        current_idx = np.where(seq_t == current_time[0])[0][0]
        next_time[0] = seq_t[point_types == 0][np.where(seq_t[point_types == 0] > temp_time)[0][0]]
        current_loc[0] = seq_s[current_idx]
    if df_np[i * ns2, 0] > next_time[1]:
        temp_time = next_time[1]
        current_time[1] = temp_time
        current_idx = np.where(seq_t == current_time[1])[0][0]
        next_time[1] = seq_t[point_types == 1][np.where(seq_t[point_types == 1] > temp_time)[0][0]]
        current_loc[1] = seq_s[current_idx]

    ## HJ: Plot actual events on intensity map
    if i >= start_is[0]:
        ax[0].scatter(current_loc[0][0], current_loc[0][1], color=type1_color, s=20)
        ax[1].scatter(current_loc[0][0], current_loc[0][1], color=type1_color, s=20)
    if i >= start_is[1]:
        ax[0].scatter(current_loc[1][0], current_loc[1][1], color=type2_color, s=20)
        ax[1].scatter(current_loc[1][0], current_loc[1][1], color=type2_color, s=20)

    ## two separate colorbars for ax[0] and ax[1]
    plt.colorbar(p0, ax=ax[0], location='right', format='%.2f')
    plt.colorbar(p1, ax=ax[1], location='right', format='%.2f')
    # plt.colorbar(p1, ax = [ax[0], ax[1]], location='right', format='%.2f')

    #  current model time in days (float)
    t_cur = df_np[i * ns2, 0]

    #  convert to calendar date (‘int’ keeps Feb=28 assumption)
    cal_date = day0 + dt.timedelta(days=int(t_cur))

    #  formatted label, e.g.  2 May  or  02 Jan
    # date_label = cal_date.strftime("%-d %b")  # Linux / macOS
    date_label = cal_date.strftime("%#d %b")  # Windows

    plt.suptitle(f"{date_label}", fontsize=12)

    #plt.suptitle(f'time: {df_np[i * ns2, 0]:.2f}', fontsize=12)

    ## HJ: Rename plots for proper temporal order.
    if i < 10:
        savename = f'000{i}'
    elif i < 100:
        savename = f'00{i}'
    elif i < 1000:
        savename = f'0{i}'
    else:
        savename = f'{i}'
    # plt.savefig(f'{savepath}/{data_idx}_{savename}.jpg', bbox_inches='tight')
    plt.savefig(f'{savepath}/{year_list[data_idx]}_{savename}.jpg', bbox_inches='tight')
    plt.close()


#HJ: Temporal conditional intensity
#Temporal component comparison
time_points = df.t.to_numpy().reshape(-1, ns2)
intensity_0 = df.iloc[:, 3].to_numpy().reshape(-1, ns2)
intensity_1 = df.iloc[:, 4].to_numpy().reshape(-1, ns2)

unit_area = 2 ** 2 / (ns2)

temporal_intensity_0 = intensity_0.sum(axis=1) * unit_area
temporal_intensity_1 = intensity_1.sum(axis=1) * unit_area

time_df = pd.DataFrame(
    {'t': time_points[:, 0], 'intensity_0': temporal_intensity_0, 'intensity_1': temporal_intensity_1})
time_df.describe()

min_value = .5
plt.figure(figsize=(15, 6))
plt.scatter(seq_t[point_types == 0], np.zeros_like(seq_t[point_types == 0]), s=10)
plt.scatter(seq_t[point_types == 1], np.zeros_like(seq_t[point_types == 1]), color="orange",   s=10)
plt.plot(time_df.t, time_df.intensity_0, label='BH')  # Taliban-TTP BH)
plt.plot(time_df.t, time_df.intensity_1, color="orange",  label='FE')  # Baloch FE) color="orange",
#plt.ylim(None, 3.0)

plt.legend()
plt.savefig(f'{savepath}/../{year_list[data_idx]}_time_intensity_{n_times}_{n_epochs}.jpg', bbox_inches='tight')
# plt.savefig(f'{savepath}/../../{data_idx}_{n_comp}_{n_epochs}_time_intensity.jpg', bbox_inches='tight')
plt.show()
plt.close()




# spatial intensity: temporally aggregated

# ---------------------------------------------------------------
# TEMPORALLY AGGREGATED SPATIAL PROFILE  (correct area handling)
# ---------------------------------------------------------------

# 1. unique longitudes and cell geometry
ns2     = ns ** 2
x_vals  = np.unique(st_sample[:ns2, 1])
x_vals.sort()
dx      = x_vals[1] - x_vals[0]          # width of a grid cell   (2/ns)
dy      = dx                             # because grid is square
cell_A  = dx * dy                        # = (2/ns)^2

# 2. integrate λ over time  ->  λ_T(x,y)  (still per unit‑area)
unit_time      = time_points[1] - time_points[0]
lambda_T_cell0 = intensity_0.sum(axis=0) * unit_time     # (ns²,)
lambda_T_cell1 = intensity_1.sum(axis=0) * unit_time

# 3. convert to expected *counts* per cell by multiplying area
events_cell0 = lambda_T_cell0 * cell_A
events_cell1 = lambda_T_cell1 * cell_A

# 4. build DataFrame and sum over y to obtain λ*(x)
space_df = pd.DataFrame({
    "x" : st_sample[:ns2, 1],
    "events0" : events_cell0,
    "events1" : events_cell1
})

lambda0_x = space_df.groupby("x")["events0"].sum().reindex(x_vals).values
lambda1_x = space_df.groupby("x")["events1"].sum().reindex(x_vals).values

# 5. plot
# plt.figure(figsize=(15, 6))
# plt.plot(x_vals, lambda0_x, label="Type 0 (TTP/BH)")
# plt.plot(x_vals, lambda1_x, label="Type 1 (Baloch/FE)", color="orange")
# plt.xlabel("x (normalised longitude)")
# plt.ylabel("expected events in 366 days per unit‑x strip")
# plt.title("Temporally aggregated spatial intensity profile")
# plt.legend(); plt.tight_layout()
# plt.savefig(f'{savepath}/../{year_list[data_idx]}_spatial_intensity_x{n_times}_{n_epochs}.jpg', bbox_inches='tight')
#
# plt.show()
# plt.close()


###################################################
# group by y
##################################################


# --------------------------------------------------------------------
# NORTH–SOUTH PROFILE  ( λ*(y) )
# --------------------------------------------------------------------

# 1. unique latitudes and cell geometry
y_vals  = np.unique(st_sample[:ns2, 2])   # y‑coordinates of grid lines
y_vals.sort()
dy      = y_vals[1] - y_vals[0]           # height of one grid cell (2/ns)
dx      = dy                              # square cells: dx == dy
cell_A  = dx * dy

# 2. integrate λ over time  (events / unit‑area)  ->  multiply by cell area
unit_time     = time_points[1] - time_points[0]
events_cell0  = intensity_0.sum(axis=0) * unit_time * cell_A
events_cell1  = intensity_1.sum(axis=0) * unit_time * cell_A

# 3. build DataFrame and sum over x to obtain λ*(y)
space_df_y = pd.DataFrame({
    "x" : st_sample[:ns2, 1],
    "y" : st_sample[:ns2, 2],
    "events0" : events_cell0,
    "events1" : events_cell1
})

lambda0_y = space_df_y.groupby("y")["events0"].sum().reindex(y_vals).values
lambda1_y = space_df_y.groupby("y")["events1"].sum().reindex(y_vals).values

# 4. plot
# plt.figure(figsize=(15, 6))
# plt.plot(y_vals, lambda0_y, label="Type 0 (TTP/BH)")
# plt.plot(y_vals, lambda1_y, label="Type 1 (Baloch/FE)", color="orange")
# plt.xlabel("y (normalised latitude)")
# plt.ylabel("expected events in 366 days per unit‑y strip")
# plt.title("Temporally aggregated spatial intensity profile (north–south)")
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'{savepath}/../{year_list[data_idx]}_spatial_intensity_y{n_times}_{n_epochs}.jpg', bbox_inches='tight')
#
# plt.show()
# plt.close()




# ==================================================================
#  DEGREE LABELS FOR NHP SPATIAL PROFILES (east–west & north–south)
# ==================================================================

# ── 0.  Geographic limits used in the original normalisation ──────

LON_MIN, LON_MAX = 60,  78                   # °E
LAT_MIN, LAT_MAX = 23,  38                    # °N


# ── 1.  Vectorised forward / inverse transforms ───────────────────
def xnorm_to_lon(xn):                     # xn can be scalar or ndarray
    return LON_MIN + 0.5 * (xn + 1.0) * (LON_MAX - LON_MIN)

def lon_to_xnorm(lon):
    return 2 * (lon - LON_MIN) / (LON_MAX - LON_MIN) - 1

def ynorm_to_lat(yn):
    return LAT_MIN + 0.5 * (yn + 1.0) * (LAT_MAX - LAT_MIN)

def lat_to_ynorm(lat):
    return 2 * (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 1


# ------------------------------------------------------------------
# 2.  EAST–WEST profile with longitude ticks
# ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.plot(x_vals, lambda0_x, label="Type 0 (TTP/BH)")
# ax.plot(x_vals, lambda1_x, label="Type 1 (Baloch/FE)", color="orange")
# ax.set_xlabel("x (normalised)")
# ax.set_ylabel("expected events / 366 days / x‑strip")
# ax.set_title("Temporally aggregated spatial profile  (east–west)")
# ax.legend()
#
# tick_idx   = np.linspace(0, len(x_vals)-1, 6, dtype=int)
# x_tick_pos = x_vals[tick_idx]
# ax.set_xticks(x_tick_pos)
# ax.set_xticklabels([f"{lon:.1f}" for lon in xnorm_to_lon(x_tick_pos)])
#
# secax = ax.secondary_xaxis("top",
#             functions=(xnorm_to_lon, lon_to_xnorm))
# secax.set_xlabel("longitude (°E)")
#
# fig.tight_layout()
# fig.savefig(os.path.join(savepath,
#             f"nhp_spatial_profile_EW_deg_{year_list[data_idx]}.png"),
#             bbox_inches="tight")
# plt.show()
# plt.close()
#
#
# # ------------------------------------------------------------------
# # 3.  NORTH–SOUTH profile with latitude ticks
# # ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.plot(y_vals, lambda0_y, label="Type 0 (TTP/BH)")
# ax.plot(y_vals, lambda1_y, label="Type 1 (Baloch/FE)", color="orange")
# ax.set_xlabel("y (normalised)")
# ax.set_ylabel("expected events / 366 days / y‑strip")
# ax.set_title("Temporally aggregated spatial profile  (north–south)")
# ax.legend()
#
# tick_idx   = np.linspace(0, len(y_vals)-1, 6, dtype=int)
# y_tick_pos = y_vals[tick_idx]
# ax.set_xticks(y_tick_pos)
# ax.set_xticklabels([f"{lat:.1f}" for lat in ynorm_to_lat(y_tick_pos)])
#
# secax = ax.secondary_xaxis("top",
#             functions=(ynorm_to_lat, lat_to_ynorm))
# secax.set_xlabel("latitude (°N)")
#
# fig.tight_layout()
# fig.savefig(os.path.join(savepath,
#             f"nhp_spatial_profile_NS_deg_{year_list[data_idx]}.png"),
#             bbox_inches="tight")
# plt.show()
# plt.close()
#
# print("NHP spatial profiles with degree labels saved to", savepath)
#

