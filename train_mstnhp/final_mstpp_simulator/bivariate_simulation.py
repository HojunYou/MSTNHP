
# Resolve relative data/output paths under train_mstnhp.
from pathlib import Path as _Path
import os as _os
import sys as _sys
_training_dir = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_training_dir.parent))
_sys.path.insert(0, str(_training_dir))
_os.chdir(_training_dir)

import os, argparse, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from final_mstpp_simulator.mstppg import BivariateHawkesLam, BivariateSpatialTemporalPointProcess
from final_mstpp_simulator.stppg import (
    SeparableExponentialKernel, SeparablePowerKernel,
    SeparableMixtureExponentialKernel, SeparableSinKernel,
    NonseparableKernel, SeparableExpTimeGaussianSpaceKernel
)


def build_kernel_set(tag):
    """Return [k00,k01,k10,k11] for the requested tag."""
    if tag == "bivariate_1":
        # α = (0.25, 0.10, 0.10, 0.25)
        # β = (0.3, 0.3, 0.3, 0.3)
        # σ^2 = (0.5, 0.5, 0.5, 0.5)
        k00 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25, beta_t=0.3, sigma2=0.5)
        k01 = SeparableExpTimeGaussianSpaceKernel(alpha=0.10, beta_t=0.3, sigma2=0.5)
        k10 = SeparableExpTimeGaussianSpaceKernel(alpha=0.10, beta_t=0.3, sigma2=0.5)
        k11 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25, beta_t=0.3, sigma2=0.5)

    elif tag == "bivariate_2":
        # α = (0.25, -0.10, -0.10, 0.25)
        # β = (0.3, 0.3, 0.3, 0.3)
        # σ^2 = (0.5, 0.5, 0.5, 0.5)
        k00 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.3, sigma2=0.5)
        k01 = SeparableExpTimeGaussianSpaceKernel(alpha=-0.10, beta_t=0.3, sigma2=0.5)
        k10 = SeparableExpTimeGaussianSpaceKernel(alpha=-0.10, beta_t=0.3, sigma2=0.5)
        k11 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.3, sigma2=0.5)

    elif tag == "bivariate_3":
        # α = (0.25, -0.10, -0.10, 0.25)
        # β = (0.1, 0.1, 0.1, 0.1)
        # σ^2 = (0.5, 0.25, 0.25, 0.5)
        k00 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.1, sigma2=0.5)
        k01 = SeparableExpTimeGaussianSpaceKernel(alpha=-0.10, beta_t=0.1, sigma2=0.25)
        k10 = SeparableExpTimeGaussianSpaceKernel(alpha=-0.10, beta_t=0.1, sigma2=0.25)
        k11 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.1, sigma2=0.5)

    elif tag == "bivariate_4":
        # g11: 0.15*(0.5+Δt)^(-1.3) * exp(-2||Δs||)
        # g12: 0.03*exp(-0.3Δt) * exp(-2||Δs||)
        # g21: (0.05*exp(-0.2Δt) + 0.16*exp(-0.8Δt)) * exp(-2||Δs||)
        # g22: max(0, sin(Δt)/8) on 0≤Δt≤4, times exp(-2||Δs||)
        k00 = SeparablePowerKernel(C=0.15, alpha_t=0.5, beta_t=1.3, beta_s=2.0)
        k01 = SeparableExponentialKernel(C=0.03, beta_t=0.3, beta_s=2.0)
        k10 = SeparableMixtureExponentialKernel(Cs=[0.05, 0.16], beta_ts=[0.2, 0.8], beta_s=2.0)
        k11 = SeparableSinKernel(C=1.0, scale_t=8.0, beta_s=2.0)
    else:
        raise ValueError(tag)
    return [k00, k01, k10, k11]


def convert_sequence(df):
    out = []
    for i in range(len(df)):
        d = dict(
            time_since_start=float(df.time.iloc[i]),
            location_from_origin=[float(df.x.iloc[i]), float(df.y.iloc[i])],
            type_event=int(df.event_type.iloc[i]),
        )
        if i == 0:
            d["time_since_last_event"] = float(df.time.iloc[i])
            d["location_from_last_event"] = [float(df.x.iloc[i]), float(df.y.iloc[i])]
        else:
            d["time_since_last_event"] = float(df.time.iloc[i] - df.time.iloc[i - 1])
            d["location_from_last_event"] = [
                float(df.x.iloc[i] - df.x.iloc[i - 1]),
                float(df.y.iloc[i] - df.y.iloc[i - 1]),
            ]
        out.append(d)
    return out


def dump_easytpp_pickle(folder, split_name, split_list):
    path = folder / f"{split_name}.pkl"
    content = dict(train=[], dev=[], test=[], dim_process=2, args=[])
    content[split_name] = split_list
    pickle.dump(content, path.open("wb"))


def temporal_spatial_grid(seq_txy, seq_types, lam, folder):
    # --- build 40×40×1000 grid
    n_grid = 40
    x = np.linspace(-1 + 1 / n_grid, 1 - 1 / n_grid, n_grid)
    y = x.copy()
    xy = np.stack(np.meshgrid(x, y, indexing='xy'), -1).reshape(-1, 2)
    t = np.linspace(0, 100, 1001)[1:]
    nt, ns = len(t), xy.shape[0]
    int0, int1 = np.zeros((nt, ns)), np.zeros((nt, ns))
    seq_t, seq_s = seq_txy[:, 0], seq_txy[:, 1:]
    for k, tk in enumerate(t):
        mask = seq_t < tk
        hist_t = seq_t[mask]
        hist_s = seq_s[mask]
        hist_m = seq_types[mask]
        for j, (xx, yy) in enumerate(xy):
            int0[k, j] = lam.value(tk, hist_t, np.array([xx, yy]), hist_s, 0, hist_m)
            int1[k, j] = lam.value(tk, hist_t, np.array([xx, yy]), hist_s, 1, hist_m)
    # ----- save frames (PER-FRAME scaling: vmin/vmax computed at each k)
    viz_dir = folder / "viz" / "frames_true"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, nt, 4):
        vmin = min(float(int0[k].min()), float(int1[k].min()))
        vmax = max(float(int0[k].max()), float(int1[k].max()))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].pcolor(x, y, int0[k].reshape(n_grid, n_grid), cmap="viridis", vmin=vmin, vmax=vmax)
        im1 = ax[1].pcolor(x, y, int1[k].reshape(n_grid, n_grid), cmap="viridis", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im1, ax=[ax[0], ax[1]], shrink=.8)
        ax[0].set_title(f"λ0  t={t[k]:.1f}"); ax[1].set_title(f"λ1  t={t[k]:.1f}")
        plt.savefig(viz_dir / f"frame_{k:04d}.jpg", bbox_inches="tight"); plt.close()
    # ----- temporal curves
    dxdy = (2 / n_grid) ** 2
    lam0 = int0.sum(1) * dxdy
    lam1 = int1.sum(1) * dxdy
    plt.figure(figsize=(10, 4))
    plt.plot(t, lam0, label="λ0"); plt.plot(t, lam1, label="λ1")
    plt.scatter(seq_txy[seq_types == 0, 0], np.zeros_like(seq_txy[seq_types == 0, 0]), s=8, c='k')
    plt.scatter(seq_txy[seq_types == 1, 0], np.zeros_like(seq_txy[seq_types == 1, 0]), s=8, c='r')
    plt.legend(); plt.tight_layout()
    plt.savefig(folder / "viz" / "temporal_intensity.png"); plt.close()




# main
def main(seed, tags):
    np.random.seed(seed)
    for tag in tags:
        kernel = build_kernel_set(tag)
        lam = BivariateHawkesLam(mu=[0.1, 0.1], kernel=kernel, maximum=5e2)
        pp = BivariateSpatialTemporalPointProcess(lam)

        T, S = [0., 100.], [[-1., 1.], [-1., 1.]]
        points, types, sizes = pp.generate(T, S, batch_size=1000, verbose=False)

        seq_dicts = []; n_train = 900; n_dev = 50
        for n in range(1000):
            N = sizes[n]
            arr = points[n][:N]; mks = types[n][:N]
            mask = arr[:, 0] > 0
            df = pd.DataFrame({
                "time": arr[mask, 0],
                "x":    arr[mask, 1],
                "y":    arr[mask, 2],
                "event_type": mks[mask]
            })
            seq_dicts.append(convert_sequence(df))

        folder = Path(f"data/bistpp_{tag}")
        folder.mkdir(parents=True, exist_ok=True)
        dump_easytpp_pickle(folder, "train", seq_dicts[:n_train])
        dump_easytpp_pickle(folder, "dev",   seq_dicts[n_train:n_train + n_dev])
        dump_easytpp_pickle(folder, "test",  seq_dicts[n_train + n_dev:])
        pickle.dump([points, types], (folder / "points.pkl").open("wb"))

        seq0 = points[0][:sizes[0]]; seq0_m = types[0][:sizes[0]]
        temporal_spatial_grid(seq0, seq0_m, lam, folder)
        print(f"✓ finished dataset '{tag}'  → {folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument(
        "--tag", "-t",
        choices=["bivariate_1","bivariate_2","bivariate_3","bivariate_4","all"],
        default="all",
        help="Which kernel preset to simulate. Use 'all' to run every preset."
    )
    args = parser.parse_args()
    tag_list = ["bivariate_1","bivariate_2","bivariate_3","bivariate_4"] if args.tag == "all" else [args.tag]
    main(args.seed, tag_list)
