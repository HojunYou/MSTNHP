import argparse
import numpy as np
import torch
import pickle, shapely.geometry as sg
from shapely.geometry import Point, Polygon, MultiPolygon
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml',
                    help='Dir of configuration yaml to train and evaluate the model. ')

parser.add_argument('--experiment_id', type=str, required=False, default='STNHP_eval',
                    help='Experiment id in the config file.')

parser.add_argument('--ckpt_dir', type=str, required=False, default='./checkpoints/246153_22781026365888_250615-180958/models/saved_model',
                        help='Path to model')

parser.add_argument('--polygon',      required=False,    default='./data/gtd_pakistan/pakistan_polygon_norm.pkl',
                       help='pickled Shapely Polygon')


parser.add_argument('--device',      default='cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
## HJ: Create a model runner from the configuration and train the model
model_runner = Runner.build_from_config(config)

## HJ: Load model from the checkpoint
n_comp = model_runner.model_wrapper.model.n_comps



model_runner.model_wrapper.model.load_state_dict(torch.load(args.ckpt_dir, map_location=args.device),
                                           strict=False)

model = model_runner.model_wrapper.model.to(args.device)
model.eval()

## HJ: get data and model configurations
data_config = model_runner.runner_config.data_config
model_backend = model_runner.runner_config.base_config.backend

## HJ: Load train dataset
split = 'test'
data_dir = data_config.get_data_dir(split)
data_source_type = data_dir.split('.')[-1]
data_loader = STPPDataLoader(data_config, model_backend)  # data_loader = TPPDataLoader(data_config, model_backend)
data = data_loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)  # dataset = TPPDataset(data)
sample_size = len(dataset)
tokenizer = STEventTokenizer(data_config.data_specs)  # tokenizer = EventTokenizer(data_config.data_specs)
batch_size = 1  # 13

## HJ: Load train dataloader
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size=batch_size, shuffle=False, space=True)

with open(args.polygon, 'rb') as f:
    polygon = pickle.load(f)

#   true spatial area
if isinstance(polygon, sg.Polygon):
    area = polygon.area
else:                               # MultiPolygon
    area = sum(p.area for p in polygon.geoms)


def sample_points_in_polygon(polygon, n, minx, miny, maxx, maxy):
    sampled = np.zeros((0, 2), dtype=float)
    while sampled.shape[0] < n:
        M = max((n - sampled.shape[0]) * 2, 50)
        xs = np.random.uniform(minx, maxx, M)
        ys = np.random.uniform(miny, maxy, M)
        pts = np.stack([xs, ys], axis=1)
        mask = np.array([polygon.contains(Point(x, y)) for x, y in pts])
        valid = pts[mask]
        if valid.shape[0] > 0:
            need = n - sampled.shape[0]
            sampled = np.vstack([sampled, valid[:need]])
    return sampled


def generate_homogeneous_points(time_delta, M_upper, polygon, seed=0):
    np.random.seed(seed)
    B, L = time_delta.shape
    minx, miny, maxx, maxy = polygon.bounds
    poly_area = polygon.area if isinstance(polygon, (Polygon, MultiPolygon)) else sum(p.area for p in polygon.geoms)

    lam = M_upper * (poly_area * time_delta)
    N = np.random.poisson(lam)

    homo_t = [[None] * L for _ in range(B)]
    homo_xy = [[None] * L for _ in range(B)]
    for b in range(B):
        for j in range(L):
            n_ij = int(N[b, j])
            if n_ij == 0:
                homo_t[b][j] = np.zeros(0, dtype=float)
                homo_xy[b][j] = np.zeros((0, 2), dtype=float)
                continue
            t_vals = np.random.uniform(0.0, time_delta[b, j], size=n_ij)
            xy = sample_points_in_polygon(polygon, n_ij, minx, miny, maxx, maxy)
            order = np.argsort(t_vals)
            homo_t[b][j] = t_vals[order]
            homo_xy[b][j] = xy[order]
    return homo_t, homo_xy


def get_upperbound_stnhp(model, batch):
    time_seqs, time_deltas, space_seqs, space_deltas, type_seqs, _, _, _ = batch
    dtimespace = model.make_multiple_dtimespace_loss_samples(time_deltas[:, 1:], space_seqs[:, :-1])
    dtime = dtimespace[..., 0, 0]
    dspace = torch.norm(dtimespace[..., :, :, 1:], dim=-1, p=2)
    intensities = model.compute_intensities_at_sample_spacetimes(
        time_seqs, time_deltas, space_seqs, space_deltas, type_seqs,
        dtime, dspace, compute_last_step_only=False
    )
    M_upper = intensities.sum(dim=-1).max(dim=-1)[0] * model.event_sampler.over_sample_rate


    return M_upper.detach().cpu().numpy()


def get_space_components(space_seq, homo_xy):
    B, L, _ = space_seq.shape
    space_samples = [
        [hxy[:, None, :].repeat(1, 3, 1) for hxy in hxys]
        for hxys in homo_xy
    ]
    for b in range(B):
        for j in range(L):
            if j == 0:
                space_samples[b][j][:, :2, :] = 0
                space_samples[b][j][:, 2, :] -= space_seq[b, j, :]
            elif j == 1:
                space_samples[b][j][:, 0, :] = 0
                space_samples[b][j][:, 1:, :] -= space_seq[b, j-1:j+1]
            else:
                space_samples[b][j] -= space_seq[b, j-2:j+1]
    return [ [ss.norm(p=2, dim=-1) for ss in row] for row in space_samples ]


def thinning_predict(model, time_seqs, time_deltas, space_seqs, space_deltas,
                     type_seqs_np, homo_t, homo_xy, λ_samples, M_upper):
    B, Lm1, M_max, T = λ_samples.shape
    lam_sum = λ_samples.sum(-1)
    U = np.random.rand(B, Lm1, M_max)
    keep = lam_sum > (U * M_upper[..., None])
    first_m = np.full((B, Lm1), M_max, dtype=int)
    for b,j,m in np.argwhere(keep):
        if m < first_m[b, j]: first_m[b, j] = m

    pred_t = np.zeros((B, Lm1))
    pred_xy = np.zeros((B, Lm1, 2))
    pred_type = np.zeros((B, Lm1), dtype=int)
    for b in range(B):
        for j in range(Lm1):
            m = first_m[b, j]
            if m < M_max:
                pred_t[b, j] = float(homo_t[b][j][m])
                pred_xy[b, j] = homo_xy[b][j][m]
                pred_type[b, j] = int(λ_samples[b, j, m].argmax())

    ends = np.argmax(type_seqs_np[:, 1:] == 2, axis=1)
    ends[ends == 0] = Lm1

    fb = [(b, j) for b in range(B) for j in range(Lm1)
          if first_m[b, j] == M_max and j < ends[b]]


    # for any (b,j) that never got a sample accepted, just fall back to Δt and type=0

    for b, j in fb:
        pred_t[b, j] = float(time_deltas[b, j + 1].item())
        pred_xy[b, j] = 0.0
        pred_type[b, j] = 0  # fall back to type 0



    Bfull, Lfull = type_seqs_np.shape
    pred_time_seqs = np.zeros((Bfull, Lfull))
    pred_space_seqs = np.zeros((Bfull, Lfull, 2))
    pred_type_seqs = np.zeros((Bfull, Lfull), dtype=int)
    pred_time_seqs[:, :Lm1] = pred_t
    pred_space_seqs[:, :Lm1] = pred_xy
    pred_type_seqs[:, :Lm1] = pred_type
    for b,j in np.argwhere(type_seqs_np[:, 1:] == 2):
        pred_time_seqs[b, j] = 2.0
        pred_space_seqs[b, j] = 0.0
        pred_type_seqs[b, j] = 2
    total_len = B * Lm1 - np.sum(type_seqs_np[:, 1:] == 2)
    return pred_time_seqs, pred_space_seqs, pred_type_seqs, total_len



def prediction_stnhp(model, batch, polygon, seed=0):
    time_seqs, time_deltas, space_seqs, space_deltas, type_seqs, *_ = batch
    B, L = time_seqs.shape
    Lm1 = L - 1

    # 1) bump sampling resolution
    old_nt, old_ns = model.loss_integral_num_sample_per_step, model.spatial_npoints
    model.loss_integral_num_sample_per_step = 50
    model.spatial_npoints = 10

    # 2) get upper bound M_upper [B, L-1]
    M_upper = get_upperbound_stnhp(model, batch)

    # 3) generate hom. samples in numpy
    td_np = time_deltas[:, 1:].cpu().numpy()          # [B, L-1]
    homo_t_np, homo_xy_np = generate_homogeneous_points(td_np, M_upper, polygon, seed)

    # ─── convert all to torch.float32 on the right device ───
    device = model.device
    homo_t  = [
        [ torch.tensor(arr, dtype=torch.float32, device=device)
          for arr in row ]
        for row in homo_t_np
    ]
    homo_xy = [
        [ torch.tensor(arr, dtype=torch.float32, device=device)
          for arr in row ]
        for row in homo_xy_np
    ]


    # 4) build space‐offset grids (they accept torch.Tensor)
    space_samples = get_space_components(space_seqs[:, :-1], homo_xy)

    # 5) collect λ‐samples
    M_max = max(len(homo_t[b][j]) for b in range(B) for j in range(Lm1))
    λ_samples = np.zeros((B, Lm1, M_max, model.num_event_types), dtype=float)
    for b in range(B):
        for j in range(Lm1):
            n = len(homo_t[b][j])
            if n == 0:
                continue
            with torch.no_grad():
                lam = model.compute_intensities_at_sample_spacetimes(
                    time_seqs[b:b+1],
                    time_deltas[b:b+1],
                    space_seqs[b:b+1],
                    space_deltas[b:b+1],
                    type_seqs[b:b+1],
                    homo_t[b][j][None, None, :],      # already float32 on device
                    space_samples[b][j][None, None, :],# already float32 on device
                    compute_last_step_only=False
                )
            λ_samples[b, j, :n, :] = lam[0, j].cpu().numpy()

    # 6) restore original sampling settings
    model.loss_integral_num_sample_per_step = old_nt
    model.spatial_npoints = old_ns

    # 7) thinning and fallback
    pred_time_delta_seqs, pred_space_seqs, pred_type_seqs, total_len = thinning_predict(
        model,
        time_seqs, time_deltas, space_seqs, space_deltas,
        type_seqs.cpu().numpy(),
        homo_t, homo_xy,
        λ_samples, M_upper
    )

    # 8) compute errors over the first L-1 steps
    time_pred_diff = (time_deltas[:, 1:].cpu().numpy() - pred_time_delta_seqs[:, :Lm1]) ** 2
    space_pred_diff = np.sum((space_seqs[:, 1:].cpu().numpy() - pred_space_seqs[:, :Lm1]) ** 2, axis=-1)
    type_pred_diff = (type_seqs[:, 1:].cpu().numpy() != pred_type_seqs[:, :Lm1])

    return time_pred_diff, space_pred_diff, type_pred_diff, total_len



import os, logging, sys
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/prediction.log", mode="w")

    ],
)
logger = logging.getLogger(__name__)




import time
from datetime import datetime
start_time = time.time()
## start_time to hour:min:seconds
logger.info(f"Start time: {datetime.now()}")

time_errors = []
type_errors = []
space_time_errors = []
space_errors = []

n_rep = 2

for i in range(n_rep):
    total_len = 0
    time_pred_diff = 0
    total_time_pred_diff = []
    space_pred_diff = 0
    type_pred_diff = 0
    torch.manual_seed(i)
    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(args.device) for t in batch.values())
        batch_time_pred_diff, batch_space_pred_diff, batch_type_pred_diff, batch_len = prediction_stnhp(model, batch, polygon, seed=i)
        time_pred_diff+=batch_time_pred_diff.sum()
        space_pred_diff+=batch_space_pred_diff.sum()
        type_pred_diff+=batch_type_pred_diff.sum()
        total_len+=batch_len
        logger.info(f"Batch time prediction diff: {np.sqrt(batch_time_pred_diff.sum()/batch_len).item()}")
        logger.info(f"Batch space prediction diff: {np.sqrt(batch_space_pred_diff.sum()/batch_len).item()}")
        logger.info(f"Batch type prediction diff: {batch_type_pred_diff.sum()/batch_len}")
        logger.info(f"Batch {batch_idx}/{len(train_loader)} done.")
        if np.sqrt(batch_time_pred_diff.sum()/batch_len).item()>20:
            break

    end_time = time.time()
    logger.info(f"Time prediction diff: {np.sqrt(time_pred_diff/total_len).item()}")
    logger.info(f"Space prediction diff: {np.sqrt(space_pred_diff/total_len).item()}")
    logger.info(f"Type prediction diff: {type_pred_diff/total_len}")
    logger.info(f"Space-time prediction diff: {np.sqrt((time_pred_diff+space_pred_diff).sum()/total_len).item()}")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Time elapsed: {end_time-start_time}")

    type_errors.append(type_pred_diff/total_len)
    time_errors.append(np.sqrt(time_pred_diff/total_len).item())
    space_errors.append(np.sqrt(space_pred_diff/total_len).item())
    space_time_errors.append(np.sqrt((time_pred_diff+space_pred_diff).sum()/total_len).item())

type_errors = np.array(type_errors)
time_errors = np.array(time_errors)
space_time_errors = np.array(space_time_errors)
space_errors = np.array(space_errors)
logger.info(f"Distribution of type errors: {np.mean(type_errors), np.std(type_errors)}")
logger.info(f"Distribution of time errors: {np.mean(time_errors), np.std(time_errors)}")
logger.info(f"Distribution of space errors: {np.mean(space_errors), np.std(space_errors)}")
logger.info(f"Distribution of space-time errors: {np.mean(space_time_errors), np.std(space_time_errors)}")

