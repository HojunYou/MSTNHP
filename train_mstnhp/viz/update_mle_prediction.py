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


#  imports
import argparse, logging, pickle, itertools, sys
import numpy as np
from shapely.geometry import Point
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

from stpp_simulator.mstppg import BivariateHawkesLam
from stpp_simulator.stppg import SeparableExponentialKernel

#  CLI flags
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml')
parser.add_argument('--experiment_id', default='STNHP_eval')
parser.add_argument('--polygon',default='./data/gtd_pakistan/nigeria_polygon_norm.pkl')
parser.add_argument('--target_fallback', type=float, default=0.05,help='desired expected fall-back probability per interval')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()



# Get data
config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)


data_cfg = model_runner.runner_config.data_config
backend = model_runner.runner_config.base_config.backend
split = 'test'
data_dir = data_cfg.get_data_dir(split)
loader = STPPDataLoader(data_cfg, backend)
data = loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)
tokenizer = STEventTokenizer(data_cfg.data_specs)
data_loader = get_data_loader(dataset, backend, tokenizer,
                               batch_size=1, shuffle=False, space=True)


#  polygon
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


#  Hawkes intensity (MLE parameters pakistan)
# mu = [1.244, 1.032]
# A = [1.793, -0.015, 0.185, 1.203]
# beta_t = [0.137, 0.121, 1.800, 0.265]
# beta_s = [8.768, 1.751, 0.030, 9.004]


#  Hawkes intensity (MLE parameters Nigeria(10 years))
# mu = [1.503, 1.03]
# A = [0.541, 0.421, -0.010, 0.657]
# beta_t = [0.079, 0.126, 0.692, 0.510]
# beta_s = [4.358, 1.857, 0.080, 3.039]


#  Hawkes intensity (MLE parameters Nigeria(2014))
mu = [1.503, 1.03]
A = [0.541, 0.421, -0.010, 0.657]
beta_t = [0.079, 0.126, 0.692, 0.510]
beta_s = [4.358, 1.857, 0.080, 3.039]
C = [round(a * bt * bs, 4) for a, bt, bs in zip(A, beta_t, beta_s)]

kernels = [SeparableExponentialKernel(C=c, beta_t=bt, beta_s=bs)
           for c, bt, bs in zip(C, beta_t, beta_s)]
hawkes_intensity = BivariateHawkesLam(mu, kernels)      # λ_k(t, x | history)


#  upper-bound on intensity per interval
def compute_M_upper(time_seq, dt_seq, space_seq, type_seq,
                    n_time_probe=10, n_space_probe=10):
    """
    Very cheap upper bound: evaluate the summed intensity on a small
    spatio-temporal grid and multiply by 10.
    """
    batch_size, seq_len = time_seq.shape
    time_ratio = np.linspace(0.0, 1.0, n_time_probe)[None, None, :]
    time_probes = dt_seq[:, 1:, None] * time_ratio              # [B, L-1, nT]
    space_grid = np.linspace(-1.0, 1.0, n_space_probe,
                             endpoint=False) + 1.0 / n_space_probe

    M_upper = np.zeros((batch_size, seq_len - 1))
    for b in range(batch_size):
        for j in range(seq_len - 1):
            absolute_t = time_seq[b, j] + time_probes[b, j]      # shape (nT,)
            grid = np.array(list(itertools.product(
                absolute_t, space_grid, space_grid)))
            tau  = grid[:, 0]
            pos  = grid[:, 1:]

            past_t   = time_seq[b, :j + 1]
            past_pos = space_seq[b, :j + 1]
            past_typ = type_seq[b, :j + 1]

            lambda_values = []
            for t_val, x_val in zip(tau, pos):
                total = sum(
                    hawkes_intensity.value(t_val, past_t,
                                           x_val, past_pos,
                                           k,       past_typ)
                    for k in (0, 1))
                lambda_values.append(total)
            M_upper[b, j] = 2.0 * max(lambda_values)           # oversample×10
    return M_upper                                             # shape (B, L-1)


#  homogeneous Poisson proposals
def draw_homogeneous_proposals(dt_seq, M_up, rng, target_fb):
    """Return per-interval lists of proposal times and positions."""
    POSSOIN_OVERSAMPLE = 5
    Lambda_raw = M_up * (poly_area * dt_seq)     # naive mean count
    Lambda_min = -np.log(target_fb)            # so P(fallback) ≤ target
    Lambda = np.maximum(Lambda_raw, Lambda_min)

    counts = rng.poisson(Lambda)               # [B, L-1]
    batch_times, batch_pos = [], []

    for dt_row, count_row in zip(dt_seq, counts):
        row_times, row_pos = [], []
        for dt_val, cnt in zip(dt_row, count_row):
            times = rng.uniform(0.0, dt_val, cnt)
            pos = sample_inside_polygon(cnt, rng)
            idx = np.argsort(times)
            row_times.append(times[idx])
            row_pos  .append(pos[idx])
        batch_times.append(row_times)
        batch_pos  .append(row_pos)
    return batch_times, batch_pos              # list[B][L-1][*]


#  one-step prediction core
def predict_one_batch(t_seq, dt_seq, x_seq, type_seq,
                      target_fb, seed):
    """
    Returns predictions + number of fallbacks for one batch.
    Input arrays are numpy, shape (B, L[, 2]).
    """
    rng = np.random.default_rng(seed)
    batch_size, seq_len = t_seq.shape
    M_up = compute_M_upper(t_seq, dt_seq, x_seq, type_seq)       # (B, L-1)

    prop_times, prop_pos = draw_homogeneous_proposals(
        dt_seq[:, 1:], M_up, rng, target_fb)

    pred_dt   = np.zeros((batch_size, seq_len - 1))
    pred_pos  = np.zeros((batch_size, seq_len - 1, 2))
    pred_type = np.zeros((batch_size, seq_len - 1), int)
    fallback_count = 0

    for b in range(batch_size):
        for j in range(seq_len - 1):
            proposals_t = prop_times[b][j]
            proposals_x = prop_pos  [b][j]

            accepted_index = -1
            if proposals_t.size > 0:
                intensities = []
                for tau, xpos in zip(proposals_t, proposals_x):
                    total = sum(
                        hawkes_intensity.value(t_seq[b, j] + tau,
                                               t_seq[b, :j + 1],
                                               xpos,
                                               x_seq[b, :j + 1],
                                               k,
                                               type_seq[b, :j + 1])
                        for k in (0, 1))
                    intensities.append(total)
                intensities = np.array(intensities)
                keep_mask = intensities > rng.random(intensities.size) * M_up[b, j]
                if keep_mask.any():
                    accepted_index = np.where(keep_mask)[0][0]


            # if accepted_index >= 0:            # accepted proposal
            #     tau  = proposals_t[accepted_index]
            #     xpos = proposals_x[accepted_index]
            #     pred_dt  [b, j] = tau
            #     pred_pos [b, j] = xpos
            #     # choose type with largest component intensity
            #     comp_int = [hawkes_intensity.value(t_seq[b, j] + tau,
            #                                        t_seq[b, :j + 1],
            #                                        xpos,
            #                                        x_seq[b, :j + 1],
            #                                        k,
            #                                        type_seq[b, :j + 1])
            #                 for k in (0, 1)]
            #     pred_type[b, j] = int(np.argmax(comp_int))
            # else:                              # fall-back
            #     fallback_count += 1
            #     pred_dt [b, j] = dt_seq[b, j + 1]
            #     pred_pos[b, j] = 0.0
            #     comp_int = [hawkes_intensity.value(t_seq[b, j + 1],
            #                                        t_seq[b, :j + 1],
            #                                        np.zeros(2),
            #                                        x_seq[b, :j + 1],
            #                                        k,
            #                                        type_seq[b, :j + 1])
            #                 for k in (0, 1)]
            #     pred_type[b, j] = int(np.argmax(comp_int))
            # ---------------------------------------------------------------
            #  inside the j–loop, keep everything else unchanged
            # ---------------------------------------------------------------
            if accepted_index >= 0:  # proposal survived
                tau = proposals_t[accepted_index]
                xpos = proposals_x[accepted_index]
                pred_dt[b, j] = tau
                pred_pos[b, j] = xpos
                # component type = arg-max λ_k
                comp = [hawkes_intensity.value(t_seq[b, j] + tau,
                                               t_seq[b, :j + 1],
                                               xpos,
                                               x_seq[b, :j + 1],
                                               k,
                                               type_seq[b, :j + 1])
                        for k in (0, 1)]
                pred_type[b, j] = int(np.argmax(comp))

            else:  # fallback (no survivor)
                fallback_count += 1
                if proposals_t.size > 0:
                    # Horizon = last proposal time + 1 s
                    tau_fallback = proposals_t[-1] + 1.0
                else:
                    # No proposals at all: use exponential(Λ_min) draw
                    Lambda_min = -np.log(target_fb)
                    tau_fallback = rng.exponential(scale=1.0 / Lambda_min)

                pred_dt[b, j] = tau_fallback
                pred_pos[b, j] = 0.0  # centre of polygon
                comp = [hawkes_intensity.value(t_seq[b, j] + tau_fallback,
                                               t_seq[b, :j + 1],
                                               np.zeros(2),
                                               x_seq[b, :j + 1],
                                               k,
                                               type_seq[b, :j + 1])
                        for k in (0, 1)]
                pred_type[b, j] = int(np.argmax(comp))

    return pred_dt, pred_pos, pred_type, fallback_count


# runner
# logging set‑up
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("mle_nigeria_2014_prediction_7_31")

# evaluation loop
import time
from datetime import datetime

n_rep = 10                    # independent repeats
target_fallback_probability = args.target_fallback
start_wall = datetime.now()
tic = time.time()
log.info("Evaluation started at %s", start_wall.strftime("%Y‑%m‑%d %H:%M:%S"))

rmse_t_runs = []
rmse_s_runs = []
rmse_st_runs = []
type_err_runs = []
fb_runs = []

for rep in range(n_rep):
    Σ_t = Σ_s = Σ_y = 0.0          # numerators
    Σ_N = Σ_fb = 0                 # denominators

    for batch_idx, batch_dict in enumerate(data_loader):
        time_seqs, dt_seqs, space_seqs, dspace_seqs, type_seqs = \
            [t.cpu().numpy() for t in list(batch_dict.values())[:5]]

        pred_dt, pred_pos, pred_type, fb_cnt = predict_one_batch(
            time_seqs, dt_seqs, space_seqs, type_seqs, target_fallback_probability,
            seed=rep * 1000 + batch_idx)

        mask = type_seqs[:, 1:] != 2                # ignore padding
        N = mask.sum()                           # observed intervals

        err_t2 = ((dt_seqs[:, 1:] - pred_dt)  ** 2)[mask]
        err_s2 = (((space_seqs[:, 1:] - pred_pos) ** 2).sum(-1))[mask]
        err_ty = (type_seqs[:, 1:] != pred_type)[mask]

        Σ_t += err_t2.sum()
        Σ_s += err_s2.sum()
        Σ_y += err_ty.sum()
        Σ_N += N
        Σ_fb += fb_cnt


    # per‑run statistics
    rmse_t = np.sqrt(Σ_t / Σ_N)
    rmse_s = np.sqrt(Σ_s / Σ_N)
    rmse_st = np.sqrt((Σ_t + Σ_s) / Σ_N)   # √(time²+space²)
    err_ty = Σ_y  / Σ_N
    fb_rate = Σ_fb / Σ_N

    rmse_t_runs.append(rmse_t)
    rmse_s_runs.append(rmse_s)
    rmse_st_runs.append(rmse_st)
    type_err_runs.append(err_ty)
    fb_runs.append(fb_rate)

    log.info(">>> run %02d summary:  RMSE_t %.3f  RMSE_s %.3f  "
             "RMSE_st %.3f  TypeErr %.3f  FB %.3f",
             rep, rmse_t, rmse_s, rmse_st, err_ty, fb_rate)

#  overall summary
rmse_t_runs = np.asarray(rmse_t_runs)
rmse_s_runs = np.asarray(rmse_s_runs)
rmse_st_runs = np.asarray(rmse_st_runs)
type_err_runs = np.asarray(type_err_runs)
fb_runs = np.asarray(fb_runs)

log.info("=====  OVERALL (%d runs)  =====", n_rep)
log.info("RMSE_time       : %.3f ± %.3f", rmse_t_runs.mean(),  rmse_t_runs.std())
log.info("RMSE_space      : %.3f ± %.3f", rmse_s_runs.mean(),  rmse_s_runs.std())
log.info("RMSE_space‑time : %.3f ± %.3f", rmse_st_runs.mean(), rmse_st_runs.std())
log.info("Type‑error rate : %.3f ± %.3f", type_err_runs.mean(),type_err_runs.std())
log.info("Fallback rate   : %.3f ± %.3f", fb_runs.mean(),      fb_runs.std())
log.info("Total wall‑clock elapsed: %.1f s", time.time() - tic)

