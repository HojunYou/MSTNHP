#!/usr/bin/env python
# simulate_mstpp_separable.py  – generates a bivariate STPP dataset that **only**
# uses a separable exponential kernel with *no cross‑triggering* (k01 = k10 = 0).
# -----------------------------------------------------------------------------
# Usage:
#   python simulate_mstpp_separable.py               # default seed=0
#   python simulate_mstpp_separable.py -s 123        # custom RNG seed
# -----------------------------------------------------------------------------


# Resolve relative data/output paths under train_mstnhp.
from pathlib import Path as _Path
import os as _os
import sys as _sys
_training_dir = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_training_dir.parent))
_sys.path.insert(0, str(_training_dir))
_os.chdir(_training_dir)

import argparse, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

from final_mstpp_simulator.mstppg import BivariateHawkesLam, BivariateSpatialTemporalPointProcess
from final_mstpp_simulator.stppg import SeparableExponentialKernel

# -----------------------------------------------------------------------------#
#  Helper functions
# -----------------------------------------------------------------------------#

def build_kernel_set():
    """Return [k00, k01, k10, k11] with separable‑exponential self‑kernels and
    zero cross‑kernels (no cross‑triggering)."""

    # self‑excitation kernels (type 0→0 and type 1→1)
    k00 = SeparableExponentialKernel(C=0.15, beta_t=0.5, beta_s=2.0)
    k11 = SeparableExponentialKernel(C=0.15, beta_t=0.5, beta_s=2.0)

    # cross kernels are identically zero with C=0

    k01 = SeparableExponentialKernel(C=0.0, beta_t=1.0, beta_s=1.0)
    k10 = SeparableExponentialKernel(C=0.0, beta_t=1.0, beta_s=1.0)

    return [k00, k01, k10, k11]


def convert_sequence(df):
    """Convert to dictionary list used by EasyTPP."""
    seq = []
    for i in range(len(df)):
        d = {
            "time_since_start": float(df.time.iloc[i]),
            "location_from_origin": [float(df.x.iloc[i]), float(df.y.iloc[i])],
            "type_event": int(df.event_type.iloc[i]),
        }
        if i == 0:
            d["time_since_last_event"] = float(df.time.iloc[i])
            d["location_from_last_event"] = [float(df.x.iloc[i]), float(df.y.iloc[i])]
        else:
            d["time_since_last_event"] = float(df.time.iloc[i] - df.time.iloc[i - 1])
            d["location_from_last_event"] = [float(df.x.iloc[i] - df.x.iloc[i - 1]),
                                             float(df.y.iloc[i] - df.y.iloc[i - 1])]
        seq.append(d)
    return seq


def dump_easytpp_pickle(folder, split_name, split_list):
    path = folder / f"{split_name}.pkl"
    content = dict(train=[], dev=[], test=[], dim_process=2, args=[])
    content[split_name] = split_list
    pickle.dump(content, path.open("wb"))


def temporal_spatial_grid(seq_txy, seq_types, lam, folder, n_grid=40):
    """Plot true λ(x,y,t) for the first sequence.

    *Spatial* heat‑maps autoscale **per frame** (each λ0/λ1 panel uses its
    own min/max)."""

    x = np.linspace(-1 + 1/n_grid, 1 - 1/n_grid, n_grid)
    y = x.copy()
    xy = np.stack(np.meshgrid(x, y, indexing='xy'), -1).reshape(-1, 2)
    t = np.linspace(0, 100, 1001)[1:]  # skip t=0
    nt, ns = len(t), xy.shape[0]
    int0, int1 = np.zeros((nt, ns)), np.zeros((nt, ns))

    seq_t, seq_s = seq_txy[:, 0], seq_txy[:, 1:]
    for k, tk in enumerate(t):
        mask = seq_t < tk
        hist_t, hist_s, hist_m = seq_t[mask], seq_s[mask], seq_types[mask]
        for j, (xx, yy) in enumerate(xy):
            int0[k, j] = lam.value(tk, hist_t, np.array([xx, yy]), hist_s, 0, hist_m)
            int1[k, j] = lam.value(tk, hist_t, np.array([xx, yy]), hist_s, 1, hist_m)

    viz_dir = folder / "viz" / "frames_true"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, nt, 4):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # frame‑specific colour scaling
        ax[0].pcolor(x, y, int0[k].reshape(n_grid, n_grid), cmap="viridis",
                     vmin=0, vmax=int0[k].max())
        ax[1].pcolor(x, y, int1[k].reshape(n_grid, n_grid), cmap="viridis",
                     vmin=0, vmax=int1[k].max())
        plt.colorbar(ax[1].collections[0], ax=[ax[0], ax[1]], shrink=.8)
        ax[0].set_title(f"λ0  t={t[k]:.1f}")
        ax[1].set_title(f"λ1  t={t[k]:.1f}")
        plt.savefig(viz_dir / f"frame_{k:04d}.jpg", bbox_inches="tight")
        plt.close()

    # ––– temporal curves  –––
    dxdy = (2 / n_grid) ** 2
    lam0_time = int0.sum(1) * dxdy
    lam1_time = int1.sum(1) * dxdy
    plt.figure(figsize=(10, 4))
    plt.plot(t, lam0_time, label="λ0")
    plt.plot(t, lam1_time, label="λ1")
    plt.scatter(seq_txy[seq_types == 0, 0], np.zeros_like(seq_txy[seq_types == 0, 0]),
                s=8, c='k')
    plt.scatter(seq_txy[seq_types == 1, 0], np.zeros_like(seq_txy[seq_types == 1, 0]),
                s=8, c='r')
    plt.legend(); plt.tight_layout()
    plt.savefig(folder / "viz" / "temporal_intensity.png")
    plt.close()


# -----------------------------------------------------------------------------#
#  main routine
# -----------------------------------------------------------------------------#

def main(seed):
    np.random.seed(seed)

    # build kernel & simulator
    kernel = build_kernel_set()
    lam = BivariateHawkesLam(mu=[0.1, 0.1], kernel=kernel, maximum=5e2)
    pp = BivariateSpatialTemporalPointProcess(lam)

    # simulate 1000 sequences on [0,100] × [-1,1]^2
    T, S = [0.0, 100.0], [[-1.0, 1.0], [-1.0, 1.0]]
    points, types, sizes = pp.generate(T, S, batch_size=1000, verbose=False)

    # EasyTPP split
    seq_dicts, n_train, n_dev = [], 900, 50
    for n in range(1000):
        N = sizes[n]
        arr, mks = points[n][:N], types[n][:N]
        mask = arr[:, 0] > 0
        df = pd.DataFrame(dict(time=arr[mask, 0],
                               x=arr[mask, 1],
                               y=arr[mask, 2],
                               event_type=mks[mask]))
        seq_dicts.append(convert_sequence(df))

    folder = Path("data/bistpp_separable_exp")
    folder.mkdir(parents=True, exist_ok=True)
    dump_easytpp_pickle(folder, "train", seq_dicts[:n_train])
    dump_easytpp_pickle(folder, "dev", seq_dicts[n_train:n_train + n_dev])
    dump_easytpp_pickle(folder, "test", seq_dicts[n_train + n_dev:])

    pickle.dump([points, types], (folder / "points.pkl").open("wb"))

    # visualise first sequence
    seq0 = points[0][:sizes[0]]
    seq0_types = types[0][:sizes[0]]
    temporal_spatial_grid(seq0, seq0_types, lam, folder)

    print("✓ finished dataset 'separable_exp'  →", folder)


# -----------------------------------------------------------------------------#
#  Run code
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0, help="random seed")
    args = parser.parse_args()
    main(args.seed)
