#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-step prediction for a bivariate spatio-temporal Hawkes process
fitted by MLE.  Uses polygon-aware Ogata thinning and records the
number of fall-backs (intervals where no proposal survives thinning).
"""

# Resolve project imports and legacy paths before loading dependencies.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()


import argparse, logging, pickle, itertools, sys, time
import numpy as np, shapely.geometry as sg
from shapely.geometry import Point
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

from stpp_simulator.mstppg import BivariateHawkesLam
from stpp_simulator.stppg  import SeparableExponentialKernel

# ───────────────────────────── CLI flags ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml')
parser.add_argument('--experiment_id', default='STNHP_eval')
parser.add_argument('--polygon',    default='./data/gtd_pakistan/pakistan_polygon_norm.pkl')
parser.add_argument('--seed',       type=int, default=0)
args = parser.parse_args()

# ─────────────────────────── Get data ─────────────────────────────────
config       = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)
data_cfg     = model_runner.runner_config.data_config
backend      = model_runner.runner_config.base_config.backend
split        = 'test'
data_dir     = data_cfg.get_data_dir(split)
loader       = STPPDataLoader(data_cfg, backend)
data         = loader.build_input_from_pkl(data_dir, split)
dataset      = STPPDataset(data)
tokenizer    = STEventTokenizer(data_cfg.data_specs)
data_loader  = get_data_loader(dataset, backend, tokenizer,
                               batch_size=1, shuffle=False, space=True)

# ─────────────────────── polygon helpers ────────────────────────────
with open(args.polygon, 'rb') as f:
    polygon = pickle.load(f)
poly_area  = polygon.area
minx, miny, maxx, maxy = polygon.bounds

def sample_inside_polygon(n, rng):
    """Uniform points inside the polygon by rejection sampling."""
    points = np.empty((0, 2), float)
    while points.shape[0] < n:
        batch_size = max(2 * (n - points.shape[0]), 100)
        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)
        candidates = np.stack((xs, ys), axis=1)
        mask = [polygon.contains(Point(x, y)) for x, y in candidates]
        points = np.vstack((points, candidates[mask]))
    return points[:n]

#  Hawkes intensity (MLE parameters)
mu      = [1.244, 1.032]
A       = [1.793, -0.015, 0.185, 1.203]
beta_t  = [0.137, 0.121, 1.800, 0.265]
beta_s  = [8.768, 1.751, 0.030, 9.004]
C       = [round(a * bt * bs, 4) for a, bt, bs in zip(A, beta_t, beta_s)]
kernels = [SeparableExponentialKernel(C=c, beta_t=bt, beta_s=bs)
           for c, bt, bs in zip(C, beta_t, beta_s)]
hawkes_intensity = BivariateHawkesLam(mu, kernels)

# ─────────────────── upper-bound on intensity per interval ───────────────────
def compute_M_upper(time_seq, dt_seq, space_seq, type_seq,
                    n_time_probe=10, n_space_probe=10):
    batch_size, seq_len = time_seq.shape
    time_ratio  = np.linspace(0.0, 1.0, n_time_probe)[None, None, :]
    time_probes = dt_seq[:, 1:, None] * time_ratio    # [B, L-1, nT]
    space_grid  = np.linspace(-1.0, 1.0, n_space_probe,
                              endpoint=False) + 1.0/n_space_probe

    M_upper = np.zeros((batch_size, seq_len - 1))
    for b in range(batch_size):
        for j in range(seq_len - 1):
            absolute_t = time_seq[b, j] + time_probes[b, j]
            grid = np.array(list(itertools.product(absolute_t, space_grid, space_grid)))
            taus = grid[:, 0]
            poss = grid[:, 1:]
            past_t   = time_seq[b, :j+1]
            past_pos = space_seq[b, :j+1]
            past_typ = type_seq[b, :j+1]

            vals = []
            for τ, x in zip(taus, poss):
                vals.append(
                    sum(hawkes_intensity.value(τ, past_t, x, past_pos, k, past_typ)
                        for k in (0,1))
                )
            M_upper[b, j] = 10.0 * max(vals)
    return M_upper

# ─────────────────── homogeneous Poisson proposals ───────────────────
def draw_homogeneous_proposals(dt_seq, M_up, rng):
    """
    Return per-interval lists of proposal times and positions,
    using the *raw* Poisson parameter Λ_raw = M_up * area * Δt.
    """
    Lambda_raw = M_up * poly_area * dt_seq       # [B, L-1]
    counts     = rng.poisson(Lambda_raw)         # [B, L-1]

    batch_times, batch_pos = [], []
    for dt_row, cnt_row in zip(dt_seq, counts):
        row_times, row_pos = [], []
        for Δt, cnt in zip(dt_row, cnt_row):
            times = rng.uniform(0.0, Δt, cnt)
            pos   = sample_inside_polygon(cnt, rng)
            idx   = np.argsort(times)
            row_times.append(times[idx])
            row_pos  .append(pos  [idx])
        batch_times.append(row_times)
        batch_pos  .append(row_pos)
    return batch_times, batch_pos

# ─────────────────── one-step prediction core ───────────────────
def predict_one_batch(t_seq, dt_seq, x_seq, type_seq, seed):
    rng = np.random.default_rng(seed)
    batch_size, seq_len = t_seq.shape

    M_up = compute_M_upper(t_seq, dt_seq, x_seq, type_seq)
    prop_times, prop_pos = draw_homogeneous_proposals(dt_seq[:,1:], M_up, rng)

    pred_dt   = np.zeros((batch_size, seq_len-1))
    pred_pos  = np.zeros((batch_size, seq_len-1, 2))
    pred_type = np.zeros((batch_size, seq_len-1), int)
    fb_count  = 0

    for b in range(batch_size):
        for j in range(seq_len-1):
            times = prop_times[b][j]
            poss  = prop_pos[b][j]
            accepted = -1

            if times.size:
                lambs = np.array([
                    sum(hawkes_intensity.value(t_seq[b,j]+τ,
                                               t_seq[b,:j+1],
                                               x,
                                               x_seq[b,:j+1],
                                               k, type_seq[b,:j+1])
                        for k in (0,1))
                    for τ, x in zip(times, poss)
                ])
                keep = lambs > rng.random(lambs.size)*M_up[b,j]
                if keep.any():
                    accepted = np.where(keep)[0][0]

            if accepted >= 0:
                τ, x = times[accepted], poss[accepted]
                pred_dt[b,j]   = τ
                pred_pos[b,j]  = x
                comp = [hawkes_intensity.value(t_seq[b,j]+τ,
                                               t_seq[b,:j+1],
                                               x,
                                               x_seq[b,:j+1],
                                               k, type_seq[b,:j+1])
                        for k in (0,1)]
                pred_type[b,j] = int(np.argmax(comp))
            else:
                fb_count += 1
                pred_dt[b,j]   = dt_seq[b,j+1]
                pred_pos[b,j]  = 0.0
                comp = [hawkes_intensity.value(t_seq[b,j+1],
                                               t_seq[b,:j+1],
                                               np.zeros(2),
                                               x_seq[b,:j+1],
                                               k, type_seq[b,:j+1])
                        for k in (0,1)]
                pred_type[b,j] = int(np.argmax(comp))

    return pred_dt, pred_pos, pred_type, fb_count

# ───────────────────────── evaluation loop ─────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("MLE-eval")

n_rep = 20
errors_t, errors_s, errors_y, fb_rates = [], [], [], []
start = time.time()

for rep in range(n_rep):
    sum_t = sum_s = sum_y = total = fb = 0
    for idx, batch in enumerate(data_loader):
        t, dt, x, dx, y = [t.cpu().numpy() for t in list(batch.values())[:5]]
        pred_dt, pred_x, pred_k, fbc = predict_one_batch(t, dt, x, y, seed=rep*1000+idx)

        mask = y[:,1:]!=2
        batch_t_diff = ((dt[:,1:]-pred_dt)**2)[mask]
        batch_s_diff = (((x[:,1:]-pred_x)**2).sum(-1))[mask]
        batch_y_diff = (y[:,1:]!=pred_k)[mask]

        # accumulate
        sum_t += batch_t_diff.sum()
        sum_s += batch_s_diff.sum()
        sum_y += batch_y_diff.sum()
        total += mask.sum()
        fb    += fbc

        # log all four metrics per batch
        logger.info((
            "[run %02d] batch %03d  "
            "RMSE_time %.3f  RMSE_space %.3f  "
            "TypeErr %.3f  FB_rate %.3f"
        ) % (
            rep, idx,
            np.sqrt(batch_t_diff.mean()),
            np.sqrt(batch_s_diff.mean()),
            batch_y_diff.mean(),
            fbc / mask.sum()
        ))

    # per-run summary
    errors_t .append(np.sqrt(sum_t/total))
    errors_s .append(np.sqrt(sum_s/total))
    errors_y .append(sum_y/total)
    fb_rates.append(fb/total)

logger.info("=== FINAL ===")
logger.info("RMSE_time   : %.3f ± %.3f", np.mean(errors_t), np.std(errors_t))
logger.info("RMSE_space  : %.3f ± %.3f", np.mean(errors_s), np.std(errors_s))
logger.info("TypeErr     : %.3f ± %.3f", np.mean(errors_y), np.std(errors_y))
logger.info("Fallback    : %.3f ± %.3f", np.mean(fb_rates), np.std(fb_rates))
logger.info("Elapsed     : %.1f s", time.time()-start)
