#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One‑step prediction for a bivariate spatio‑temporal Hawkes process fitted
by MLE.  Sampling is uniform in the square [-1,1]^2 and Ogata thinning
tracks the number of fall‑backs (intervals where no proposal survives).
"""

# Use the shared training directory for imports and relative paths.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()


#  imports

import argparse, logging, itertools, sys, time
import numpy as np
from shapely.geometry import Point
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess.dataset import get_data_loader
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from stpp_simulator.mstppg import BivariateHawkesLam
from stpp_simulator.stppg import SeparableExponentialKernel

#  constants
SQUARE_MIN, SQUARE_MAX = -1.0, 1.0
SQUARE_AREA = 4.0                     # (2×2)

TARGET_FB = None      # adaptive Poisson floor; set to None for “raw” Poisson

#  CLI
p = argparse.ArgumentParser()
p.add_argument('--config_dir',   default='configs/experiment_config_stnhp.yaml')
p.add_argument('--experiment_id',default='STNHP_eval')
p.add_argument('--seed',         type=int, default=0)
args = p.parse_args()

#  data loader
cfg          = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
runner       = Runner.build_from_config(cfg)
data_cfg     = runner.runner_config.data_config
backend      = runner.runner_config.base_config.backend
split        = 'test'
data_dir     = data_cfg.get_data_dir(split)

loader   = STPPDataLoader(data_cfg, backend)
raw_data = loader.build_input_from_pkl(data_dir, split)
dataset  = STPPDataset(raw_data)
tokenizer= STEventTokenizer(data_cfg.data_specs)
data_loader = get_data_loader(dataset, backend, tokenizer,
                              batch_size=1, shuffle=False, space=True)

#  square‑sampling
def sample_inside_square(n, rng):
    """Uniform points in the square [-1,1]^2."""
    xs = rng.uniform(SQUARE_MIN, SQUARE_MAX, n)
    ys = rng.uniform(SQUARE_MIN, SQUARE_MAX, n)
    return np.stack((xs, ys), axis=1)


#  Hawkes intensity (MLE parameters)
mu = [0.11035684, 0.09951600]
A = [0.248, 0.017, 0.199, 0.091]
beta_t = [0.853, 0.111, 0.494, 0.356]
beta_s = [2.005, 1.754, 1.975, 1.824]
C = [round(a*bt*bs, 4) for a,bt,bs in zip(A, beta_t, beta_s)]
kernels = [SeparableExponentialKernel(C=c, beta_t=bt, beta_s=bs)
           for c,bt,bs in zip(C, beta_t, beta_s)]
hawkes_intensity = BivariateHawkesLam(mu, kernels)


#   upper bound M_j
def compute_M_upper(time_seq, dt_seq, space_seq, type_seq,
                    n_time_probe=20, n_space_probe=20):
    B, L = time_seq.shape
    t_ratio  = np.linspace(0.0, 1.0, n_time_probe)[None, None, :]
    t_probe  = dt_seq[:, 1:, None] * t_ratio                 # B×(L-1)×nT
    x_grid   = np.linspace(-1.0, 1.0, n_space_probe, endpoint=False) + 1/n_space_probe

    M_up = np.zeros((B, L-1))
    for b in range(B):
        for j in range(L-1):
            grid = np.array(list(itertools.product(
                       time_seq[b, j] + t_probe[b, j], x_grid, x_grid)))
            τs, xs = grid[:,0], grid[:,1:]
            past_t, past_x, past_y = time_seq[b,:j+1], space_seq[b,:j+1], type_seq[b,:j+1]
            vals = [
                sum(hawkes_intensity.value(τ, past_t, x, past_x, k, past_y)
                    for k in (0,1))
                for τ,x in zip(τs, xs)
            ]
            M_up[b, j] = 2.0 * max(vals)        # oversample rate
    return M_up


#  homogeneous proposals
def draw_homogeneous_proposals(dt_seq, M_up, rng):
    Poisson_oversample = 5
    Λ_raw = M_up * SQUARE_AREA * dt_seq * Poisson_oversample        # B×(L-1)

    if TARGET_FB is not None:
        Λ_min = -np.log(TARGET_FB)
        Λ     = np.maximum(Λ_raw, Λ_min)
    else:
        Λ = Λ_raw

    counts = rng.poisson(Λ)
    B, Lm1 = counts.shape
    batch_times, batch_pos = [], []
    for b in range(B):
        times_row, pos_row = [], []
        for j in range(Lm1):
            cnt  = int(counts[b, j])
            t    = rng.uniform(0.0, dt_seq[b, j], cnt)
            xy   = sample_inside_square(cnt, rng)
            idx  = np.argsort(t)
            times_row.append(t[idx])
            pos_row .append(xy[idx])
        batch_times.append(times_row)
        batch_pos  .append(pos_row)
    return batch_times, batch_pos


#  one‑step prediction
def predict_one_batch(t_seq, dt_seq, x_seq, type_seq, seed):
    rng = np.random.default_rng(seed)
    B, L = t_seq.shape
    M_up = compute_M_upper(t_seq, dt_seq, x_seq, type_seq)
    prop_t, prop_x = draw_homogeneous_proposals(dt_seq[:,1:], M_up, rng)

    pred_dt   = np.zeros((B, L-1))
    pred_pos  = np.zeros((B, L-1, 2))
    pred_type = np.zeros((B, L-1), int)
    fallback  = 0

    Λ_min = (-np.log(TARGET_FB)) if TARGET_FB is not None else 1.0

    for b in range(B):
        for j in range(L-1):
            times = prop_t[b][j]
            poss  = prop_x[b][j]
            accept = -1
            if times.size:
                λ = np.array([
                    sum(hawkes_intensity.value(t_seq[b,j]+τ,
                                               t_seq[b,:j+1],
                                               x,
                                               x_seq[b,:j+1],
                                               k, type_seq[b,:j+1])
                        for k in (0,1))
                    for τ,x in zip(times, poss)])
                keep = λ > rng.random(λ.size)*M_up[b,j]
                if keep.any():
                    accept = np.where(keep)[0][0]

            if accept >= 0:                     # accepted proposal
                τ, x = times[accept], poss[accept]
                pred_dt[b,j]  = τ
                pred_pos[b,j] = x
                comp = [hawkes_intensity.value(t_seq[b,j]+τ,
                                               t_seq[b,:j+1],
                                               x,
                                               x_seq[b,:j+1],
                                               k, type_seq[b,:j+1])
                        for k in (0,1)]
                pred_type[b,j] = int(np.argmax(comp))
            else:                               # fallback – no GT leakage
                fallback += 1
                τ_fb = rng.exponential(scale=1.0/Λ_min) \
                       if times.size == 0 else times[-1] + rng.exponential(scale=1.0/Λ_min)
                pred_dt[b,j]  = τ_fb
                pred_pos[b,j] = 0.0
                comp = [hawkes_intensity.value(t_seq[b,j]+τ_fb,
                                               t_seq[b,:j+1],
                                               np.zeros(2),
                                               x_seq[b,:j+1],
                                               k, type_seq[b,:j+1])
                        for k in (0,1)]
                pred_type[b,j] = int(np.argmax(comp))

    return pred_dt, pred_pos, pred_type, fallback

# evaluation loop
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("MLE-eval")

n_rep   = 10
errs_t, errs_s, errs_y, fb_rates = [], [], [], []
tic = time.time()

for rep in range(n_rep):
    Σt=Σs=Σy=tot=fb = 0
    for idx, batch in enumerate(data_loader):
        t,dt,x,dx,y = [v.cpu().numpy() for v in list(batch.values())[:5]]
        dt_hat, pos_hat, type_hat, fb_cnt = predict_one_batch(
            t, dt, x, y, seed = rep*1000+idx)

        mask = y[:,1:] != 2
        diff_t = ((dt[:,1:]-dt_hat)**2)[mask]
        diff_s = (((x[:,1:]-pos_hat)**2).sum(-1))[mask]
        diff_y = (y[:,1:] != type_hat)[mask]

        Σt += diff_t.sum(); Σs += diff_s.sum(); Σy += diff_y.sum()
        tot+= mask.sum();    fb += fb_cnt

        log.info("[run %02d] batch %03d  RMSE_t %.3f  FB %.3f",
                 rep, idx, np.sqrt(diff_t.mean()), fb_cnt/mask.sum())

    errs_t.append(np.sqrt(Σt/tot))
    errs_s.append(np.sqrt(Σs/tot))
    errs_y.append(Σy/tot)
    fb_rates.append(fb/tot)

log.info("=== FINAL ===")
log.info("RMSE_time   : %.3f ± %.3f", np.mean(errs_t), np.std(errs_t))
log.info("RMSE_space  : %.3f ± %.3f", np.mean(errs_s), np.std(errs_s))
log.info("TypeErr     : %.3f ± %.3f", np.mean(errs_y), np.std(errs_y))
log.info("Fallback    : %.3f ± %.3f", np.mean(fb_rates), np.std(fb_rates))
log.info("Elapsed %.1f s", time.time()-tic)
