#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatio-temporal prediction with uniform square sampling + fallback tracker
"""

# Use the shared training directory for imports and relative paths.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()


import argparse, os, sys, time, logging
import numpy as np
import torch

# ─────────────────────────────── CLI ────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir',  default='configs/experiment_config_stnhp.yaml')
parser.add_argument('--experiment_id',default='STNHP_eval')
parser.add_argument('--ckpt_dir',
    default='./checkpoints/175452_22836215469056_250501-090246/models/saved_model')
parser.add_argument('--device',
    default='cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# ───────────────── easy_tpp initialisation ───────────────────────────
from easy_tpp.config_factory        import Config
from easy_tpp.runner                import STPPRunner as Runner
from easy_tpp.preprocess            import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset    import get_data_loader

config = Config.build_from_yaml_file(args.config_dir,
                                     experiment_id=args.experiment_id)
runner = Runner.build_from_config(config)
runner.model_wrapper.model.load_state_dict(
    torch.load(args.ckpt_dir, map_location=args.device), strict=False)
model = runner.model_wrapper.model.to(args.device).eval()

data_cfg     = runner.runner_config.data_config
backend      = runner.runner_config.base_config.backend
split        = 'test'
data_dir     = data_cfg.get_data_dir(split)
loader       = STPPDataLoader(data_cfg, backend)
data         = loader.build_input_from_pkl(data_dir, split)
dataset      = STPPDataset(data)
tokenizer    = STEventTokenizer(data_cfg.data_specs)
train_loader = get_data_loader(dataset, backend, tokenizer,
                               batch_size=1, shuffle=False, space=True)

# ─────────────────── helper: square sampling ─────────────────────────
def generate_homogeneous_points(dt_seq, m_upper, seed=0):
    """
    dt_seq: [B,L]  array of Δt’s
    m_upper: [B,L]  intensity upper bound from get_upperbound_stnhp
    We assume unit square [-1,1]^2 (area=4).
    """
    np.random.seed(seed)
    B, L = dt_seq.shape
    area = 4.0
    # raw Poisson means
    lam_raw = 5 * m_upper * (area * dt_seq)    # shape [B,L]
    # draw counts
    N = np.random.poisson(lam_raw)         # [B,L]

    homo_t  = [[None]*L for _ in range(B)]
    homo_xy = [[None]*L for _ in range(B)]
    for b in range(B):
        for j in range(L):
            n = int(N[b,j])
            if n == 0:
                homo_t[b][j]  = np.zeros(0,  dtype=float)
                homo_xy[b][j] = np.zeros((0,2),  dtype=float)
            else:
                # times uniform in [0,Δt]
                t_samples  = np.random.uniform(0.0, dt_seq[b,j], n)
                # xy uniform in [-1,1]^2
                xy_samples = np.random.uniform(-1.0, 1.0, (n,2))
                idx = np.argsort(t_samples)
                homo_t[b][j]  = t_samples[idx]
                homo_xy[b][j] = xy_samples[idx]
    return homo_t, homo_xy

# ─────────────────── intensity upper‐bound ───────────────────────────
def get_upperbound_stnhp(model, batch):
    time_seqs, time_deltas, space_seqs, space_deltas, type_seqs, *_ = batch
    dtimespace = model.make_multiple_dtimespace_loss_samples(
        time_deltas[:,1:], space_seqs[:,:-1]
    )
    dtime  = dtimespace[...,0,0]
    dspace = torch.norm(dtimespace[..., :, :, 1:], dim=-1, p=2)
    ints   = model.compute_intensities_at_sample_spacetimes(
        time_seqs, time_deltas, space_seqs, space_deltas, type_seqs,
        dtime, dspace, compute_last_step_only=False
    )
    M_up = ints.sum(dim=-1).max(dim=-1)[0] * model.event_sampler.over_sample_rate
    return M_up.detach().cpu().numpy()

# ─────────────────── space offsets ───────────────────────────
def get_space_components(space_seq, homo_xy):
    B, L, _ = space_seq.shape
    space_samples = [
        [hxy[:,None,:].repeat(1,3,1) for hxy in row]
        for row in homo_xy
    ]
    for b in range(B):
        for j in range(L):
            if j==0:
                space_samples[b][j][:,:2,:] = 0
                space_samples[b][j][:,2,:]  -= space_seq[b,j]
            elif j==1:
                space_samples[b][j][:,0,:]  = 0
                space_samples[b][j][:,1:,:] -= space_seq[b,j-1:j+1]
            else:
                space_samples[b][j] -= space_seq[b,j-2:j+1]
    return [[ss.norm(p=2,dim=-1) for ss in row] for row in space_samples]

# ─────────────────── thinning + fallback ──────────────────────────
def thinning_predict(model,
    time_seqs, time_dts, space_seqs, space_dts,
    type_np, homo_t, homo_xy, lam_samples, m_upper
):
    device = next(model.parameters()).device
    n_comps = model.n_comps
    B, Lm1, M_max, _ = lam_samples.shape

    # 1) find first surviving proposal
    lam_sum = lam_samples.sum(-1)
    keep    = lam_sum > np.random.rand(*lam_sum.shape)*m_upper[...,None]
    first_m = np.full((B,Lm1), M_max, int)
    for b,j,m in np.argwhere(keep):
        first_m[b,j] = min(first_m[b,j], m)

    # 2) record accepted
    pred_t    = np.zeros((B,Lm1))
    pred_xy   = np.zeros((B,Lm1,2))
    pred_type = np.zeros((B,Lm1),int)
    for b in range(B):
        for j in range(Lm1):
            m = first_m[b,j]
            if m<M_max:
                pred_t[b,j]    = homo_t[b][j][m]
                pred_xy[b,j]   = homo_xy[b][j][m]
                pred_type[b,j] = int(lam_samples[b,j,m].argmax())

    # 3) fallback
    eos = np.argmax(type_np[:,1:]==2,axis=1)
    eos[eos==0] = Lm1
    fb = [(b,j) for b in range(B) for j in range(Lm1)
          if first_m[b,j]==M_max and j<eos[b]]
    fb_cnt = len(fb)

    # 4) vectorized fallback
    if fb_cnt:
        rows, dt_in, sp_in = [], [], []
        for b,j in fb:
            rows.append((b,j))
            dt_in.append(time_dts[b,j+1].view(1,1,1))
            sp_in.append(torch.zeros((1,1,1,n_comps),
                                     dtype=torch.float32,
                                     device=device))
        dt_in = torch.cat(dt_in,0)
        sp_in = torch.cat(sp_in,0)
        N = fb_cnt

        with torch.no_grad():
            lam_fb = model.compute_intensities_at_sample_spacetimes(
                time_seqs.repeat(N,1),
                time_dts.repeat(N,1),
                space_seqs.repeat(N,1,1),
                space_dts.repeat(N,1,1),
                torch.from_numpy(type_np).to(device).repeat(N,1),
                dt_in, sp_in, compute_last_step_only=True
            ).squeeze(1)  # [N,K]

        for idx,(b,j) in enumerate(rows):
            pred_t[b,j]    = float(time_dts[b,j+1])
            pred_xy[b,j]   = 0.0
            pred_type[b,j] = int(lam_fb[idx].argmax().cpu())

    # 5) pad back
    Bfull,Lfull = type_np.shape
    p_t  = np.zeros((Bfull,Lfull));   p_t [:,:Lm1]=pred_t
    p_xy= np.zeros((Bfull,Lfull,2));   p_xy[:,:Lm1]=pred_xy
    p_ty= np.zeros((Bfull,Lfull),int); p_ty[:,:Lm1]=pred_type
    for b,j in np.argwhere(type_np[:,1:]==2):
        p_t[b,j]=2.0; p_xy[b,j]=0.0; p_ty[b,j]=2

    total_len = B*Lm1 - (type_np[:,1:]==2).sum()
    return p_t, p_xy, p_ty, total_len, fb_cnt

# ─────────────────── prediction wrapper ─────────────────────────────
def prediction_stnhp(model, batch, seed=0):
    t, dt, s, ds, y, *_ = batch
    # ensure float32
    t  = t .float()
    dt = dt.float()
    s  = s .float()
    ds = ds.float()
    B,L = t.shape;  Lm1=L-1

    old_nt, old_ns = model.loss_integral_num_sample_per_step, model.spatial_npoints
    model.loss_integral_num_sample_per_step = 60
    model.spatial_npoints                  = 20

    m_up = get_upperbound_stnhp(model, batch)
    homo_t, homo_xy = generate_homogeneous_points(
        dt[:,1:].cpu().numpy(), m_up, seed=seed)

    device = next(model.parameters()).device
    # explicitly cast to float32
    homo_t  = [[torch.tensor(a, dtype=torch.float32, device=device)
                 for a in row] for row in homo_t]
    homo_xy = [[torch.tensor(a, dtype=torch.float32, device=device)
                 for a in row] for row in homo_xy]

    space_samp = get_space_components(s[:,:-1], homo_xy)
    M_max      = max(len(a) for row in homo_t for a in row)

    lam_samp = np.zeros((B,Lm1,M_max,model.num_event_types), dtype=float)
    for b in range(B):
        for j in range(Lm1):
            n = len(homo_t[b][j])
            if n==0: continue
            lam = model.compute_intensities_at_sample_spacetimes(
                t[b:b+1], dt[b:b+1], s[b:b+1], ds[b:b+1], y[b:b+1],
                homo_t[b][j][None,None,:], space_samp[b][j][None,None,:],
                compute_last_step_only=False
            )
            lam_samp[b,j,:n] = lam[0,j].cpu().detach().numpy()

    model.loss_integral_num_sample_per_step, model.spatial_npoints = old_nt, old_ns

    return thinning_predict(
        model, t, dt, s, ds, y.cpu().numpy(),
        homo_t, homo_xy, lam_samp, m_up
    )

# ─────────────────── logging & loop ───────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/update_prediction.log", mode="w")])
logger = logging.getLogger(__name__)

logger.info(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
rmse_t, type_err, fb_rate = [], [], []
n_rep = 2

for rep in range(n_rep):
    Σ_t=Σ_s=Σ_y=n_int=n_fb=0
    torch.manual_seed(rep)
    for idx, batch in enumerate(train_loader):
        batch = tuple(v.to(args.device) for v in batch.values())
        e_t,e_s,e_y,n_i,n_f = prediction_stnhp(model, batch, seed=rep)
        Σ_t += e_t.sum();   Σ_s += e_s.sum();   Σ_y += e_y.sum()
        n_int += n_i;       n_fb += n_f

        logger.info(f"Batch {idx}: "
            f"RMSE_t={np.sqrt(Σ_t/n_int):.4f} "
            f"RMSE_s={np.sqrt(Σ_s/n_int):.4f} "
            f"TypeErr={Σ_y/n_int:.4f} "
            f"FallbackRate={n_f/n_i:.3f}"
        )

    rmse_t.append(np.sqrt(Σ_t/n_int))
    type_err.append(Σ_y/n_int)
    fb_rate.append(n_fb/n_int)
    logger.info(f"[Run {rep}] RMSE_t={rmse_t[-1]:.4f} "
                f"TypeErr={type_err[-1]:.4f} "
                f"FallbackRate={fb_rate[-1]:.3f}")

logger.info("--- summary ---")
logger.info(f"RMSE_t  mean={np.mean(rmse_t):.4f} std={np.std(rmse_t):.4f}")
logger.info(f"TypeErr mean={np.mean(type_err):.4f} std={np.std(type_err):.4f}")
logger.info(f"Fallback mean={np.mean(fb_rate):.3f} std={np.std(fb_rate):.3f}")
