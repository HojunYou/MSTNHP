#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Average negative log-likelihood for a bivariate spatio-temporal Hawkes
process fitted by MLE (polygon domain, EasyTPP splits).
--------------------------------------------------------------------
* Events term:  exact per-event λ evaluation.
* Compensator : Monte-Carlo over time × spatial polygon.
--------------------------------------------------------------------
Author: <you>
"""

import argparse, pickle, random, time, math, sys, pathlib
import numpy as np
import shapely.geometry as sg
from shapely.geometry import Point, Polygon
from tqdm.auto import tqdm

from easy_tpp.config_factory import Config
from easy_tpp.runner          import STPPRunner as Runner
from easy_tpp.preprocess      import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

from stpp_simulator.mstppg import BivariateHawkesLam
from stpp_simulator.stppg  import SeparableExponentialKernel


# CLI

p = argparse.ArgumentParser()
p.add_argument("--config_dir",  default="configs/experiment_config_gtd_pakistan.yaml")
p.add_argument("--experiment_id", default="STNHP_eval")
p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
p.add_argument("--polygon",  default="./data/gtd_pakistan/pakistan_polygon_norm.pkl")
p.add_argument("--seed",     type=int, default=0)
p.add_argument("--nt",       type=int, default=1500,
               help="#time samples for Monte-Carlo compensator")
p.add_argument("--ns",       type=int, default=8000,
               help="#spatial samples for Monte-Carlo compensator")
args = p.parse_args()


# Hawkes parameters (pakistan)

mu = [1.244, 1.032]
A = [1.793, -0.015, 0.185, 1.203]
beta_t = [0.137, 0.121, 1.800, 0.265]
beta_s = [8.768, 1.751, 0.030, 9.004]
C = [round(a*bt*bs, 4) for a, bt, bs in zip(A, beta_t, beta_s)]

kernels = [SeparableExponentialKernel(C=c, beta_t=bt, beta_s=bs)
           for c, bt, bs in zip(C, beta_t, beta_s)]
hawkes = BivariateHawkesLam(mu, kernels)          # λ_k(t,x|H)


# sample points
def sample_points_in_polygon(poly: Polygon, n, rng):
    """Uniform random (x,y) inside arbitrary polygon."""
    minx,miny,maxx,maxy = poly.bounds
    pts = []
    while len(pts) < n:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x,y)):
            pts.append((x,y))
    return np.asarray(pts, float)                # (n,2)


def loglik_one_sequence(seq_t, seq_s, seq_c, T_end,
                        poly: Polygon, lam: BivariateHawkesLam,
                        n_time, n_space, rng):
    """Return logL (events term, compensator) for one seq."""
    n_events = len(seq_t)

    # events loglike
    log_events = 0.0
    for i in range(n_events):
        lam_val = lam.value(seq_t[i],
                            seq_t[:i], seq_s[i], seq_s[:i],
                            int(seq_c[i]), seq_c[:i])
        if lam_val <= 0:
            raise ValueError("non-positive λ encountered")
        log_events += math.log(lam_val)

    #  non-event loglike
    t_samp  = rng.random(n_time) * T_end                   # (0,T]
    s_samp  = sample_points_in_polygon(poly, n_space, rng) # (n_space,2)
    area    = poly.area
    comp = np.zeros(2)

    for t in t_samp:
        hist_mask = seq_t < t
        his_t, his_s, his_c = seq_t[hist_mask], seq_s[hist_mask], seq_c[hist_mask]
        for k in (0,1):
            # vectorised over all spatial samples
            lam_vals = np.array([lam.value(t, his_t, s, his_s, k, his_c)
                                 for s in s_samp], dtype=float)
            comp[k] += lam_vals.mean()

    comp *= area * (T_end / n_time)     # ∑_k ∫∫ λ_k

    return log_events - comp.sum(), log_events, comp


# load dataset
cfg   = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
runner= Runner.build_from_config(cfg)
backend = runner.runner_config.base_config.backend
data_dir= runner.runner_config.data_config.get_data_dir(args.split)

loader   = STPPDataLoader(runner.runner_config.data_config, backend)
dataset  = STPPDataset(loader.build_input_from_pkl(data_dir, args.split))
tokenizer= STEventTokenizer(runner.runner_config.data_config.data_specs)
dl       = get_data_loader(dataset, backend, tokenizer,
                           batch_size=1, shuffle=False, space=True)

# polygon
with open(args.polygon,'rb') as f:
    poly = pickle.load(f)
if isinstance(poly, sg.MultiPolygon):
    poly = max(poly.geoms, key=lambda p:p.area)   # take largest piece

# calculate loglike
rng = random.Random(args.seed)
np_rng = np.random.default_rng(args.seed)

tot_events = 0
sum_logL   = 0.0

print(f"Evaluating log-likelihood on {args.split} split …")
for batch in tqdm(dl, ncols=80):
    t_seq   = batch['time_seqs'][0].cpu().numpy()
    s_seq   = batch['space_seqs'][0].cpu().numpy()
    c_seq   = batch['type_seqs'][0].cpu().numpy().astype(int)

    # remove zero-padded tail
    valid   = (t_seq > 0)
    t_seq   = t_seq[valid];   s_seq = s_seq[valid];   c_seq = c_seq[valid]
    T_end   = t_seq.max()

    ll, lv, comp = loglik_one_sequence(
        t_seq, s_seq, c_seq, T_end,
        poly, hawkes,
        n_time=args.nt, n_space=args.ns,
        rng=np_rng
    )

    sum_logL   += ll
    tot_events += len(t_seq)

avg_neg_logL = -sum_logL / tot_events
print(f"\nAvg -log-likelihood per event  :  {avg_neg_logL:.5f}")
print(f"(total logL = {sum_logL:.2f} over {tot_events} events)")
