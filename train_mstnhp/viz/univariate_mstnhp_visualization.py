import os, argparse, pickle, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
import shapely.geometry as sg
from matplotlib.collections import PolyCollection
from shapely.vectorized import contains as shp_contains
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml',
                    help='Dir of configuration yaml to train and evaluate the model. ')

parser.add_argument('--experiment_id', type=str, required=False, default='STNHP_eval',
                    help='Experiment id in the config file.')

parser.add_argument('--ckpt_dir', type=str, required=False, default='./checkpoints/46651_22553673290752_250517-081316/models/saved_model',
                        help='Path to model')

parser.add_argument('--data_path', type=str, required=False, default='./data/mstnhp_bra/train.pkl',
                        help='Path to dataset file')

parser.add_argument('--polygon',      required=False,    default='./data/gtd_pakistan/pakistan_polygon_norm.pkl',
                       help='pickled Shapely Polygon')

parser.add_argument('--sample_index', type=int, required=False, default=1, help='Index of the sample to evaluate')


parser.add_argument('--save_root',    default='../Meeting/2025/250517',
                    help='root where dataname/… folder tree is created')

parser.add_argument('--dataname',     default='mstnhp_bra')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Build model runner & load checkpoint
# -----------------------------------------------------------------------------
year_list = list(range(2013, 2020))
config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)
model_runner.model_wrapper.model.load_state_dict(
    torch.load(args.ckpt_dir, map_location='cpu'), strict=False)
model_runner.model_wrapper.model.eval()

# -----------------------------------------------------------------------------
# Dataset and dataloader
# -----------------------------------------------------------------------------
data_config   = model_runner.runner_config.data_config
model_backend = model_runner.runner_config.base_config.backend

split       = 'train'
data_dir    = data_config.get_data_dir(split)
data_loader = STPPDataLoader(data_config, model_backend)
data        = data_loader.build_input_from_pkl(data_dir, split)
dataset     = STPPDataset(data)
tokenizer   = STEventTokenizer(data_config.data_specs)

batch_size  = 7
train_loader = get_data_loader(dataset, model_backend, tokenizer,
                               batch_size=batch_size, shuffle=False, space=True)

batch = next(iter(train_loader)).values()
time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, \
    batch_non_pad_mask, attention_mask, type_mask = batch

# -----------------------------------------------------------------------------
# Space–time grid samples & hidden state propagation
# -----------------------------------------------------------------------------
ns = 40
num_sample_t = 1000
time_domain = 365
n_epochs = 1000
model_runner.model_wrapper.model.spatial_npoints = ns

st_sample_list = model_runner.model_wrapper.model.make_multiple_equal_dtimespace_samples(
    time_seqs, [0, time_domain], space_seqs, num_sample_t=num_sample_t)
st_sample_loc  = model_runner.model_wrapper.model.make_equal_dtimespace_samples(
    time_seqs, [0, time_domain], torch.zeros_like(space_seqs), num_sample_t=num_sample_t)

input_     = (time_seqs, time_delta_seqs, space_seqs, space_delta_seqs,
              type_seqs, None, None, None)
hiddens_ti, decays, output_states = model_runner.model_wrapper.model.forward(input_)

# -----------------------------------------------------------------------------
# Choose one sample
# -----------------------------------------------------------------------------
data_idx = args.sample_index
interval_t_samples = [st_sample_list[data_idx][i][:, 0, 0] - time_seqs[data_idx, i]
                      for i in range(len(st_sample_list[data_idx]))]
interval_s_samples = [torch.norm(st_sample_list[data_idx][i][:, :, 1:], p=2, dim=-1)
                      for i in range(len(st_sample_list[data_idx]))]

# ---- add numerical safety offset
min_thres = .01
for i, ts in enumerate(interval_t_samples):
    if ts.any() and ts[0] <= min_thres:
        ts[:ns ** 2] += min_thres
        interval_t_samples[i] = ts

# -----------------------------------------------------------------------------
# Hidden‑state → intensity on grid
# -----------------------------------------------------------------------------
state_t_samples = [model_runner.model_wrapper.model.compute_states_at_sample_points(
        decays[data_idx:data_idx+1, i:i+1],
        output_states[data_idx:data_idx+1, i:i+1],
        dt, ds)
    for i, (dt, ds) in enumerate(zip(interval_t_samples, interval_s_samples))]

lambda_t_samples = [model_runner.model_wrapper.model.layer_intensity(s).detach().cpu().numpy()[0, 0]
                    for s in state_t_samples]
lambda_t_samples = np.concatenate(lambda_t_samples, axis=0)           # (T×ns², 1)
sample_intensity = lambda_t_samples.copy()
st_sample        = torch.cat(st_sample_loc[data_idx], dim=0).cpu().numpy()

# -----------------------------------------------------------------------------
# Fitted vs actual counts (univariate ⇢ scalar)
# -----------------------------------------------------------------------------
fitted_points  = sample_intensity.sum() * time_domain / num_sample_t * (2 / ns) ** 2
expected_n     = time_seqs[data_idx].argmax().item() + 1
print(f'λ‑integral ≈ {fitted_points:.1f}  •  realised events = {expected_n}')

# -----------------------------------------------------------------------------
# Quantiles – for plotting ranges
# -----------------------------------------------------------------------------
print('Intensity quantiles:',
      np.quantile(sample_intensity, [.001, .01, .9, .99, .999]).round(4))

# -----------------------------------------------------------------------------
# Load polygon  './data/gtd_pakistan/pakistan_polygon.pkl'
# -----------------------------------------------------------------------------
with open(args.polygon, 'rb') as f:
    pakistan_polygon = pickle.load(f)
pakistan_gdf = gpd.GeoSeries([pakistan_polygon])





# -----------------------------------------------------------------------------
# Prepare dataframe (only one intensity column)
# -----------------------------------------------------------------------------
ns2 = ns ** 2
df  = pd.DataFrame({'t': st_sample[:, 0],
                    'x': st_sample[:, 1],
                    'y': st_sample[:, 2],
                    'intensity': sample_intensity[:, 0]})

savepath = Path(args.save_root, args.dataname, split,
                f'{model_runner.model_wrapper.model.n_comps}_'
                f'{n_epochs}')
savepath.mkdir(parents=True, exist_ok=True)

max_frame   = len(df) // ns2
seq_t       = time_seqs[data_idx].numpy()
seq_s       = space_seqs[data_idx].numpy()
point_types = type_seqs[data_idx].numpy()      #  all zeros now
type_color  = 'cyan'

# -----------------------------------------------------------------------------
# Per‑frame maps --------------------------------------------------------------
# -----------------------------------------------------------------------------
for i in range(0, 2 * max_frame // 5):
    fig, ax = plt.subplots(figsize=(6, 4))
    xy = [Point(x, y) for x, y in
          zip(df.x.iloc[i*ns2:(i+1)*ns2], df.y.iloc[i*ns2:(i+1)*ns2])]
    mask = gpd.GeoDataFrame(geometry=xy, crs="EPSG:4326")\
               .within(pakistan_gdf.iloc[0]).values.reshape(ns, ns)

    frame   = df.intensity.iloc[i*ns2:(i+1)*ns2].to_numpy().reshape(ns, ns)
    masked  = np.ma.masked_where(~mask, frame)
    vmin, vmax = masked.min(), masked.max()

    pcm = ax.pcolormesh(df.x.iloc[i*ns2:(i+1)*ns2].to_numpy().reshape(ns, ns),
                        df.y.iloc[i*ns2:(i+1)*ns2].to_numpy().reshape(ns, ns),
                        masked, shading='auto', cmap='coolwarm',
                        vmin=vmin, vmax=vmax)
    pakistan_gdf.boundary.plot(ax=ax, linewidth=1)

    # Actual event overlay: first upcoming event
    if np.any(seq_t <= df.t.iloc[i*ns2]):
        idx      = np.where(seq_t <= df.t.iloc[i*ns2])[0][-1]
        ax.scatter(seq_s[idx, 0], seq_s[idx, 1], c=type_color, s=20)

    plt.colorbar(pcm, ax=ax, location='right', format='%.2f')
    plt.title(f'time={df.t.iloc[i*ns2]:.2f}')
    fname = f'{year_list[data_idx]}_{i:04d}.jpg'
    plt.savefig(savepath / fname, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Temporal intensity trace ----------------------------------------------------
# -----------------------------------------------------------------------------
unit_area = 4 / ns2              # (2×2 square because coords are normalised to ±1)
temporal_intensity = df.intensity.values.reshape(-1, ns2).sum(axis=1) * unit_area
time_points        = df.t.values.reshape(-1, ns2)[:, 0]

plt.figure(figsize=(14, 4))
plt.scatter(seq_t, np.zeros_like(seq_t), s=10, label='Events')
plt.plot(time_points, temporal_intensity, label='λ(t)', lw=1.2)
plt.legend()
plt.xlabel('t')
plt.ylabel('Conditional intensity (integrated over space)')
plt.savefig(savepath.parent / f'{year_list[data_idx]}_λt.jpg',
            bbox_inches='tight')
plt.show()

