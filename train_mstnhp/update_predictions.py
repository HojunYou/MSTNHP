#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatio-temporal prediction with polygon sampling + fallback tracker
"""

import argparse, os, sys, time, logging, pickle, numpy as np, shapely.geometry as sg
from shapely.geometry import Point, Polygon, MultiPolygon
import torch

#  CLI
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml')
parser.add_argument('--experiment_id', default='STNHP_eval')
parser.add_argument('--ckpt_dir',
                    default='./checkpoints/147251_22734722094080_250721-220752/models/saved_model')
parser.add_argument('--polygon',  default='./data/gtd_pakistan/pakistan_polygon_norm.pkl')
parser.add_argument('--device',   default='cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

#  easy_tpp initialisation
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader

config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
model_runner = Runner.build_from_config(config)
model_runner.model_wrapper.model.load_state_dict(
    torch.load(args.ckpt_dir, map_location=args.device), strict=False)
model = model_runner.model_wrapper.model.to(args.device).eval()

data_cfg = model_runner.runner_config.data_config
backend = model_runner.runner_config.base_config.backend
split = 'test'
data_dir = data_cfg.get_data_dir(split)
loader = STPPDataLoader(data_cfg, backend)
data = loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)
tokenizer = STEventTokenizer(data_cfg.data_specs)
train_loader = get_data_loader(dataset, backend, tokenizer,
                               batch_size=1, shuffle=False, space=True)

with open(args.polygon, 'rb') as f:
    polygon = pickle.load(f)

area = polygon.area if isinstance(polygon, sg.Polygon)           \
       else sum(p.area for p in polygon.geoms)


#polygon sampling
def sample_points_in_polygon(poly, n, minx, miny, maxx, maxy):
    pts = np.zeros((0, 2), dtype=float)
    while pts.shape[0] < n:
        M = max(2 * (n - pts.shape[0]), 200)
        xs = np.random.uniform(minx, maxx, M)
        ys = np.random.uniform(miny, maxy, M)
        test = np.stack((xs, ys), axis=1)
        mask = np.array([poly.contains(Point(x, y)) for x, y in test])
        valid = test[mask]
        need = n - pts.shape[0]
        pts = np.vstack((pts, valid[:need]))
    return pts


def generate_homogeneous_points(dt_seq, m_upper, poly, seed_i=0):
    """dt_seq shape [B,L]; m_upper shape [B,L]"""
    np.random.seed(seed_i)
    B, L = dt_seq.shape
    minx, miny, maxx, maxy = poly.bounds
    poly_area = poly.area if isinstance(poly, (Polygon, MultiPolygon)) \
                    else sum(p.area for p in poly.geoms)


    # TARGET_FB = 0.02  # want at least <5% fallbacks
    # min_Lambda = -np.log(TARGET_FB)  # ≈3.00
    #
    # EPS = 1e-12  # tiny floor to avoid divide-by-zero
    # Lambda = np.maximum(EPS, m_upper * poly_area * dt_seq)
    # k_factor = np.maximum(1.0, min_Lambda / Lambda)
    # lam_poiss = k_factor * Lambda
    #
    # #print(f"k_factor: {k_factor}, length: {len(k_factor[0])}")
    # LAM_MAX = 1e6  # >100× larger than we’ll ever need
    # lam_poiss = np.clip(lam_poiss, 0.0, LAM_MAX)
    #
    # print("lam_poiss stats:", lam_poiss.min(), lam_poiss.mean(), lam_poiss.max())
    #
    # N = np.random.poisson(lam_poiss)

    POISSON_OVERSAMPLE = 5
    lam = m_upper * (poly_area * dt_seq) * POISSON_OVERSAMPLE
    N = np.random.poisson(lam)                        # [B,L]

    homo_t, homo_xy = [[None]*L for _ in range(B)], [[None]*L for _ in range(B)]
    for b in range(B):
        for j in range(L):
            n_ij = int(N[b, j])
            if n_ij == 0:
                homo_t[b][j] = np.zeros(0, dtype=float)
                homo_xy[b][j] = np.zeros((0,2), dtype=float)
                continue
            t = np.random.uniform(0.0, dt_seq[b, j], n_ij)
            xy = sample_points_in_polygon(poly, n_ij, minx, miny, maxx, maxy)
            idx = np.argsort(t)
            homo_t[b][j] = t[idx]
            homo_xy[b][j] = xy[idx]
    return homo_t, homo_xy


#  intensity upper-bound
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
    return [[ss.norm(p=2, dim=-1) for ss in row] for row in space_samples]


# thinning + fallback
def thinning_predict(model,
                     time_seqs, time_dts,          # [B, L]
                     space_seqs, space_dts,        # [B, L, 2]
                     type_np,                      # [B, L]  (numpy)
                     homo_t, homo_xy,              # python lists
                     lam_samples,                  # [B, L-1, M, K]
                     m_upper):                     # [B, L-1] (numpy)
    """
    Ogata-thinning + polygon fallback with logging.
    Returns:
        p_t, p_xy, p_ty : [B,L], [B,L,2], [B,L]  (numpy)
        total_len       : int
        fb_cnt          : int   (# fallback intervals)
    """
    device = next(model.parameters()).device
    n_comps = model.n_comps
    B, Lm1, M_max, _ = lam_samples.shape
    import math, random
    FB_ALPHA = 0.02
    seed = 100
    LAMBDA_MIN = -math.log(FB_ALPHA)  # ⩾ 3.0
    rng = random.Random(seed)

    # first accepted index
    lam_sum = lam_samples.sum(-1)
    keep = lam_sum > np.random.rand(*lam_sum.shape) * m_upper[..., None]
    first_m = np.full((B, Lm1), M_max, dtype=int)
    for b, j, m in np.argwhere(keep):
        if m < first_m[b, j]:
            first_m[b, j] = m

    # accepted predictions
    pred_t = np.zeros((B, Lm1))
    pred_xy = np.zeros((B, Lm1, 2))
    pred_type = np.zeros((B, Lm1), dtype=int)
    for b in range(B):
        for j in range(Lm1):
            m = first_m[b, j]
            if m < M_max:
                pred_t[b, j] = homo_t[b][j][m]
                pred_xy[b, j] = homo_xy[b][j][m]
                pred_type[b, j] = int(lam_samples[b, j, m].argmax())

    # fallback set
    eos = np.argmax(type_np[:, 1:] == 2, axis=1);  eos[eos == 0] = Lm1
    fb = [(b, j) for b in range(B) for j in range(Lm1)
            if first_m[b, j] == M_max and j < eos[b]]
    fb_cnt = len(fb)

    #  vectorised fallback pass
    if fb_cnt:
        rows_b, rows = [], []
        dt_in, sp_in = [], []
        tau_list = []

        for b, j in fb:
            rows.append((b, j))
            rows_b.append(b)

            last_prop = homo_t[b][j][-1] if len(homo_t[b][j]) else 0.0
            tau_fb = last_prop + rng.expovariate(LAMBDA_MIN)
            tau_list.append(tau_fb)

            dt_in.append(torch.tensor([[[tau_fb]]], device=device))
            sp_in.append(torch.zeros((1, 1, 1, n_comps), device=device))

        rows_b = torch.tensor(rows_b, dtype=torch.long, device=device)
        dt_in = torch.cat(dt_in, dim=0)
        sp_in = torch.cat(sp_in, dim=0)

        # gather only the sequences that actually fallback
        t_seq_fb = time_seqs[rows_b]
        dt_seq_fb = time_dts[rows_b]
        s_seq_fb = space_seqs[rows_b]
        ds_seq_fb = space_dts[rows_b]
        y_seq_fb = torch.from_numpy(type_np[rows_b.cpu()]).to(device)

        with torch.no_grad():
            lam_fb = model.compute_intensities_at_sample_spacetimes(
                t_seq_fb, dt_seq_fb, s_seq_fb, ds_seq_fb, y_seq_fb,
                dt_in, sp_in, compute_last_step_only=True
            ).squeeze(1)  # [N, K]

        for idx, (b, j) in enumerate(rows):
            pred_t[b, j] = tau_list[idx]
            pred_xy[b, j] = 0.0
            pred_type[b, j] = int(lam_fb[idx].argmax().cpu())

    # pad back to full length

    Bfull, Lfull = type_np.shape
    p_t = np.zeros((Bfull, Lfull));          p_t [:, :Lm1] = pred_t
    p_xy = np.zeros((Bfull, Lfull, 2));       p_xy[:, :Lm1] = pred_xy
    p_ty = np.zeros((Bfull, Lfull), int);     p_ty[:, :Lm1] = pred_type
    for b, j in np.argwhere(type_np[:, 1:] == 2):
        p_t[b, j] = 2.0;  p_xy[b, j] = 0.0;  p_ty[b, j] = 2

    total_len = B * Lm1 - (type_np[:, 1:] == 2).sum()
    return p_t, p_xy, p_ty, total_len, fb_cnt


#  prediction wrapper
def prediction_stnhp(model, batch, poly, seed=0):
    t, dt, s, ds, y, *_ = batch
    B, L = t.shape
    Lm1 = L - 1

    old_nt, old_ns = model.loss_integral_num_sample_per_step, model.spatial_npoints
    model.loss_integral_num_sample_per_step = 60
    model.spatial_npoints = 20

    m_up = get_upperbound_stnhp(model, batch)            # [B,L-1]
    homo_t_np, homo_xy_np = generate_homogeneous_points(
        dt[:, 1:].cpu().numpy(), m_up, poly, seed)

    device = next(model.parameters()).device
    homo_t = [[torch.tensor(a, dtype=torch.float32, device=device) for a in row] for row in homo_t_np]
    homo_xy = [[torch.tensor(a, dtype=torch.float32, device=device) for a in row] for row in homo_xy_np]

    space_samp = get_space_components(s[:, :-1], homo_xy)
    M_max = max(len(a) for row in homo_t for a in row)

    lam_samp = np.zeros((B, Lm1, M_max, model.num_event_types), dtype=float)

    for b in range(B):
        for j in range(Lm1):
            n = len(homo_t[b][j])
            if n == 0: continue
            lam = model.compute_intensities_at_sample_spacetimes(
                t[b:b+1], dt[b:b+1], s[b:b+1], ds[b:b+1], y[b:b+1],
                homo_t[b][j][None,None,:], space_samp[b][j][None,None,:],
                compute_last_step_only=False)
            lam_samp[b, j, :n] = lam[0, j].cpu().detach().numpy()

    model.loss_integral_num_sample_per_step, model.spatial_npoints = old_nt, old_ns

    p_t, p_xy, p_ty, total_len, fb_cnt = thinning_predict(
        model, t, dt, s, ds, y.cpu().numpy(),
        homo_t, homo_xy, lam_samp, m_up)

    err_t = (dt[:, 1:].cpu().numpy() - p_t[:, :Lm1])**2
    err_s = ((s[:, 1:].cpu().numpy() - p_xy[:, :Lm1])**2).sum(-1)
    err_y = (y[:, 1:].cpu().numpy() != p_ty[:, :Lm1])
    return err_t, err_s, err_y, total_len, fb_cnt

#  logging set‑up
import os, logging, sys, time, numpy as np, torch
from datetime import datetime
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/nhp_pakistan_prediction_7_30.log", mode="w")
    ],
)
log = logging.getLogger(__name__)

#  evaluation
log.info("data dir  : %s", data_dir)
log.info("start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

n_rep = 10                                  # ← number of independent repeats
time_rmse_runs = []
space_rmse_runs = []
stype_rmse_runs = []                       # √(time²+space²)
type_err_runs = []
fb_rate_runs = []

t_wall_start = time.time()

for rep in range(n_rep):
    torch.manual_seed(rep)

    Σ_t = Σ_s = Σ_type = 0.0               # numerators
    Σ_int = Σ_fb = 0                       # denominators

    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(args.device) for t in batch.values())

        # prediction_stnhp returns squared errors already
        err_t2, err_s2, err_type, n_int, n_fb = prediction_stnhp(
            model, batch, polygon, seed=rep)

        Σ_t += err_t2.sum()
        Σ_s += err_s2.sum()
        Σ_type += err_type.sum()
        Σ_int += n_int
        Σ_fb += n_fb


    #  per–run metrics
    rmse_t_run = np.sqrt(Σ_t / Σ_int)
    rmse_s_run = np.sqrt(Σ_s / Σ_int)
    rmse_st_run = np.sqrt((Σ_t + Σ_s) / Σ_int)   # space–time RMSE
    type_err_run = Σ_type / Σ_int
    fb_rate_run = Σ_fb / Σ_int

    time_rmse_runs .append(rmse_t_run)
    space_rmse_runs.append(rmse_s_run)
    stype_rmse_runs.append(rmse_st_run)
    type_err_runs  .append(type_err_run)
    fb_rate_runs   .append(fb_rate_run)

    log.info(">>> run %02d summary:  RMSE_t %.3f  RMSE_s %.3f  "
             "RMSE_st %.3f  TypeErr %.3f  FB %.3f",
             rep, rmse_t_run, rmse_s_run, rmse_st_run,
             type_err_run, fb_rate_run)

#  aggregated over n_rep
log.info("=====  OVERALL (%d runs)  =====", n_rep)
log.info("RMSE_time        : %.3f ± %.3f",
         np.mean(time_rmse_runs),  np.std(time_rmse_runs))
log.info("RMSE_space       : %.3f ± %.3f",
         np.mean(space_rmse_runs), np.std(space_rmse_runs))
log.info("RMSE_space‑time  : %.3f ± %.3f",
         np.mean(stype_rmse_runs), np.std(stype_rmse_runs))
log.info("Type‑error rate  : %.3f ± %.3f",
         np.mean(type_err_runs),   np.std(type_err_runs))
log.info("Fallback rate    : %.3f ± %.3f",
         np.mean(fb_rate_runs),    np.std(fb_rate_runs))
log.info("total wall‑clock : %.1f s", time.time() - t_wall_start)
