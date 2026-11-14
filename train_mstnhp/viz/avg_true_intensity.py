
# Use the shared training directory for imports and relative paths.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.transforms as mtransforms
import matplotlib as mpl
from matplotlib import colors

import argparse, math, pickle
from pathlib import Path


#  Kernels
class SeparableExponentialKernel(object):
    def __init__(self, C=1., beta_t=1.0, beta_s=.1):
        self.C, self.beta_t, self.beta_s = C, beta_t, beta_s

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s; delta_t = t - his_t
        if len(his_s.shape)==2:
            return self.C*np.exp(-self.beta_t*delta_t)*np.exp(-self.beta_s*np.linalg.norm(delta_s,axis=1))
        else:
            return self.C*np.exp(-self.beta_t*delta_t)*np.exp(-self.beta_s*np.linalg.norm(delta_s))


class SeparableMixtureExponentialKernel(object):
    def __init__(self, Cs, beta_ts, beta_s):
        self.Cs, self.beta_ts, self.beta_s = Cs, beta_ts, beta_s

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s; delta_t = t - his_t
        if len(his_s.shape)==2:
            return np.sum([self.Cs[i]*np.exp(-self.beta_ts[i]*delta_t)*np.exp(-self.beta_s*np.linalg.norm(delta_s,axis=1))
                           for i in range(len(self.Cs))], axis=0)
        else:
            return np.sum([self.Cs[i]*np.exp(-self.beta_ts[i]*delta_t)*np.exp(-self.beta_s*np.linalg.norm(delta_s))
                           for i in range(len(self.Cs))])


class SeparablePowerKernel(object):
    def __init__(self, C, alpha_t, beta_t, beta_s):
        self.C, self.alpha_t, self.beta_t, self.beta_s = C, alpha_t, beta_t, beta_s

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s; delta_t = t - his_t
        if len(his_s.shape)==2:
            return self.C*(self.alpha_t+delta_t)**(-self.beta_t)*np.exp(-self.beta_s*np.linalg.norm(delta_s,axis=1))
        else:
            return self.C*(self.alpha_t+delta_t)**(-self.beta_t)*np.exp(-self.beta_s*np.linalg.norm(delta_s))


class SeparableSinKernel(object):
    # fixed vectorized implementation
    def __init__(self, C, scale_t, beta_s):
        self.C, self.scale_t, self.beta_s = C, scale_t, beta_s

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s; delta_t = t - his_t
        if len(his_s.shape)==2:
            mask = ((delta_t > 0) & (delta_t <= self.scale_t/2)).astype(float)
            time_term  = self.C * np.maximum(np.sin(delta_t)/self.scale_t, 0.0) * mask
            space_term = np.exp(-self.beta_s*np.linalg.norm(delta_s,axis=1))
            return time_term*space_term
        else:
            if not (delta_t > 0 and delta_t <= self.scale_t/2): return 0.0
            time_term  = self.C * max(np.sin(delta_t)/self.scale_t, 0.0)
            space_term = np.exp(-self.beta_s*np.linalg.norm(delta_s))
            return time_term*space_term


class SeparableExpTimeGaussianSpaceKernel(object):
    # φ(Δs,Δt)=α·[β_t e^{-β_t Δt} 1_{Δt>0}]·[(1/(2πσ^2)) e^{-||Δs||^2/(2σ^2)}]
    def __init__(self, alpha=1.0, beta_t=1.0, sigma2=1.0):
        self.alpha, self.beta_t, self.sigma2 = alpha, beta_t, sigma2

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s; delta_t = t - his_t
        if len(his_s.shape)==2:
            mask = (delta_t > 0).astype(float)
            time_term = self.beta_t*np.exp(-self.beta_t*delta_t)*mask
            ds2 = np.sum(delta_s*delta_s,axis=1)
            space_term = (1.0/(2.0*math.pi*self.sigma2))*np.exp(-ds2/(2.0*self.sigma2))
            return self.alpha * time_term * space_term
        else:
            if delta_t <= 0: return 0.0
            time_term = self.beta_t*np.exp(-self.beta_t*delta_t)
            ds2 = float(np.dot(delta_s,delta_s))
            space_term = (1.0/(2.0*math.pi*self.sigma2))*np.exp(-ds2/(2.0*self.sigma2))
            return self.alpha * time_term * space_term


#  Bivariate intensity
class BivariateHawkesLam(object):
    def __init__(self, mu, kernel, maximum=1e4):
        self.mu = mu
        self.kernel = [kernel[:2], kernel[2:]]  # [[k00,k01],[k10,k11]]
        self.maximum = maximum

    def value(self, t, his_t, s, his_s, point_type, his_type):
        if len(his_t) == 0:
            return self.mu[point_type]
        total = self.mu[point_type]
        for (htyp, ht, hs) in zip(his_type, his_t, his_s):
            total += self.kernel[point_type][htyp].nu(t, s, ht, hs)
        return total


#  Kernel presets
def build_kernel_set(tag):
    if tag == "bivariate_1":
        k00 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.3, sigma2=0.5)
        k01 = SeparableExpTimeGaussianSpaceKernel(alpha=0.10,  beta_t=0.3, sigma2=0.5)
        k10 = SeparableExpTimeGaussianSpaceKernel(alpha=0.10,  beta_t=0.3, sigma2=0.5)
        k11 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.3, sigma2=0.5)
    elif tag == "bivariate_2":
        k00 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.3, sigma2=0.5)
        k01 = SeparableExpTimeGaussianSpaceKernel(alpha=-0.10, beta_t=0.3, sigma2=0.5)
        k10 = SeparableExpTimeGaussianSpaceKernel(alpha=-0.10, beta_t=0.3, sigma2=0.5)
        k11 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.3, sigma2=0.5)
    elif tag == "bivariate_3":
        k00 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.1, sigma2=0.5)
        k01 = SeparableExpTimeGaussianSpaceKernel(alpha=0.10, beta_t=0.1, sigma2=0.25)
        k10 = SeparableExpTimeGaussianSpaceKernel(alpha=0.10, beta_t=0.1, sigma2=0.25)
        k11 = SeparableExpTimeGaussianSpaceKernel(alpha=0.25,  beta_t=0.1, sigma2=0.5)
    elif tag == "bivariate_4":
        k00 = SeparablePowerKernel(C=0.15, alpha_t=0.5, beta_t=1.3, beta_s=2.0)
        k01 = SeparableExponentialKernel(C=0.03, beta_t=0.3, beta_s=2.0)
        k10 = SeparableMixtureExponentialKernel(Cs=[0.05, 0.16], beta_ts=[0.2, 0.8], beta_s=2.0)
        k11 = SeparableSinKernel(C=1.0, scale_t=8.0, beta_s=2.0)
    else:
        raise ValueError(f"Unknown tag: {tag}")
    return [k00, k01, k10, k11]


#  I/O
def load_first_sequence(train_pkl_path: Path):
    content = pickle.load(train_pkl_path.open("rb"))
    seq = content["train"][0]  # first sequence
    t = np.array([e["time_since_start"] for e in seq], dtype=float)
    xy = np.array([e["location_from_origin"] for e in seq], dtype=float)
    m = np.array([e["type_event"] for e in seq], dtype=int)
    txy = np.concatenate([t[:,None], xy], axis=1)
    return txy, m


# def plot_temporal_average(seq_txy, seq_types, lam, out_path, n_grid=40, t_max=100.0, n_t=1000):
#     # grids
#     x = np.linspace(-1 + 1/n_grid, 1 - 1/n_grid, n_grid)
#     y = x.copy()
#     xy = np.stack(np.meshgrid(x, y, indexing='xy'), -1).reshape(-1, 2)
#     t = np.linspace(0.0, t_max, n_t+1)[1:]
#
#     nt, ns = len(t), xy.shape[0]
#     int0, int1 = np.zeros((nt, ns)), np.zeros((nt, ns))
#
#     seq_t, seq_s = seq_txy[:,0], seq_txy[:,1:]
#
#     for k, tk in enumerate(t):
#         mask   = seq_t < tk
#         hist_t = seq_t[mask]
#         hist_s = seq_s[mask]
#         hist_m = seq_types[mask]
#         for j, (xx, yy) in enumerate(xy):
#             s = np.array([xx, yy])
#             int0[k, j] = lam.value(tk, hist_t, s, hist_s, 0, hist_m)
#             int1[k, j] = lam.value(tk, hist_t, s, hist_s, 1, hist_m)
#
#     # (sum * dxdy) / area
#     dxdy = (2.0 / n_grid) ** 2      # cell area for the uniform grid over [-1,1]^2
#     area = 4.0                      # |S| = area([-1,1]^2)
#     lam0_avg = (int0.sum(axis=1) * dxdy)  # / area
#     lam1_avg = (int1.sum(axis=1) * dxdy)  # / area
#     # (equivalently: int0.mean(axis=1), int1.mean(axis=1))
#
#     fig, ax = plt.subplots(figsize=(10, 4))
#
#     # lines
#     line0, = ax.plot(t, lam0_avg, label='λ0')
#     line1, = ax.plot(t, lam1_avg, label='λ1')
#
#     # match marker colors to line colors
#     c0, c1 = line0.get_color(), line1.get_color()
#
#     # manual y-limits (adjust per dataset)
#     ymin, ymax = 0.35, 0.65
#     ax.set_ylim(ymin, ymax)
#
#     # events on SAME baseline in axes coords
#     baseline = 0.02  # 2% above axes bottom
#     x0 = seq_txy[seq_types == 0, 0]
#     x1 = seq_txy[seq_types == 1, 0]
#     ax.scatter(x0, np.full_like(x0, baseline, dtype=float), s=10, c=c0,
#                transform=ax.get_xaxis_transform(), clip_on=False)
#     ax.scatter(x1, np.full_like(x1, baseline, dtype=float), s=10, c=c1,
#                transform=ax.get_xaxis_transform(), clip_on=False)
#
#     ax.legend()
#     fig.tight_layout()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.show()
#     plt.close()
#     print(f"Saved spatial-AVERAGE temporal intensity plot → {out_path}")
#

#  Temporal average plot

def add_event_ticks_axesfrac(ax, times, color, base=0.07, height=0.04,
                             lw=1.2, alpha=0.95, zorder=3, capstyle='butt'):
    """
    Short vertical 'rug' lines with height specified in axes fraction.
    X in data coords, Y in axes fraction (independent of y-lims).
    """
    times = np.asarray(times)
    if times.size == 0:
        return
    y0, y1 = base, base + height
    segs = [((t, y0), (t, y1)) for t in times]
    lc = LineCollection(segs, colors=[color], linewidths=lw, alpha=alpha,
                        capstyle=capstyle, zorder=zorder)
    lc.set_transform(mtransforms.blended_transform_factory(ax.transData, ax.transAxes))
    lc.set_clip_on(True)
    ax.add_collection(lc)


def plot_temporal_average(seq_txy, seq_types, lam, out_path,
                          n_grid=40, t_max=100.0, n_t=1000,
                          ylim=None, yq=0.99,
                          figsize=(10, 4), dpi=300,
                          label_size=18, tick_size=15, legend_size=15,
                          line_lw=1.6,
                          rug_base=0.025, rug_height=0.018, rug_lw=0.8):
    """
    Spatially-averaged temporal intensity for a bivariate ST process (types 0 and 1)
    with vertical event 'rug' markers and font-size controls.
    """

    #  grids
    x = np.linspace(-1 + 1/n_grid, 1 - 1/n_grid, n_grid)
    y = x.copy()
    xy = np.stack(np.meshgrid(x, y, indexing='xy'), -1).reshape(-1, 2)
    t = np.linspace(0.0, t_max, n_t+1)[1:]

    nt, ns = len(t), xy.shape[0]
    int0, int1 = np.zeros((nt, ns)), np.zeros((nt, ns))

    seq_t, seq_s = seq_txy[:, 0], seq_txy[:, 1:]

    for k, tk in enumerate(t):
        mask = seq_t < tk
        hist_t = seq_t[mask]
        hist_s = seq_s[mask]
        hist_m = seq_types[mask]
        for j, (xx, yy) in enumerate(xy):
            s = np.array([xx, yy])
            int0[k, j] = lam.value(tk, hist_t, s, hist_s, 0, hist_m)
            int1[k, j] = lam.value(tk, hist_t, s, hist_s, 1, hist_m)

    # spatial int
    dxdy = (2.0 / n_grid) ** 2
    lam0_avg = int0.sum(axis=1) * dxdy
    lam1_avg = int1.sum(axis=1) * dxdy

    # plotting
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=figsize)

    # lines
    line0, = ax.plot(t, lam0_avg, color='C0', lw=line_lw, )  # label='type 1'
    line1, = ax.plot(t, lam1_avg, color='C1', lw=line_lw, )  # label='type 2'

    # y-limits
    if ylim is None:
        all_vals = np.concatenate([lam0_avg, lam1_avg])
        ymin = 0.0
        ymax = float(np.nanquantile(all_vals[~np.isnan(all_vals)], yq)) if np.isfinite(all_vals).any() else 1.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = float(np.nanmax(all_vals)) if np.isfinite(np.nanmax(all_vals)) else 1.0
    else:
        ymin, ymax = float(ylim[0]), float(ylim[1])
    ax.set_ylim(ymin, ymax)

    # event rugs at fixed baseline in axes coords
    x0 = seq_txy[seq_types == 0, 0]
    x1 = seq_txy[seq_types == 1, 0]
    add_event_ticks_axesfrac(ax, x0, color=line0.get_color(), base=rug_base, height=rug_height, lw=rug_lw)
    add_event_ticks_axesfrac(ax, x1, color=line1.get_color(), base=rug_base, height=rug_height, lw=rug_lw)

    # labels/legend/fonts
    #ax.set_xlabel("Time", fontsize=label_size)
    #ax.set_ylabel("Conditional intensity", fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    #ax.legend(loc='upper right', fontsize=legend_size, frameon=False)

    #fig.tight_layout()
    fig.subplots_adjust(left=0.12, bottom=0.20, right=0.98, top=0.95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, )
    plt.close(fig)
    print(f"Saved spatial-AVERAGE temporal intensity plot → {out_path}")


#  Spatial map style choices
def spatial_style_setup(spatial_types=(0, 1)):
    type1, type2 = spatial_types

    type1_color = 'red'
    type2_color = 'red'

    # choose shapes
    type1_marker = 'o'  # 'o' = tpp,  "X" = bra
    type2_marker = '^'  # "D" = bla,  '^' = blf

    # style bundles
    t1_style = dict(color=type1_color, marker=type1_marker, s=30,
                    edgecolors='white', linewidths=0.6, zorder=4)
    t2_style = dict(color=type2_color, marker=type2_marker, s=30,
                    edgecolors='white', linewidths=0.6, zorder=4)

    return type1, type2, t1_style, t2_style


def build_spatial_df_np(seq_txy, seq_types, lam,
                        n_grid=40, t_max=100.0, n_t=1000):
    """
    Compute intensity frames on a space-time grid and pack into df_np
    compatible with your indexing style.

    """
    ns = int(n_grid)
    x = np.linspace(-1 + 1 / ns, 1 - 1 / ns, ns)
    y = x.copy()

    # flattened spatial grid of length ns2
    xv, yv = np.meshgrid(x, y, indexing='xy')
    xy_flat = np.column_stack([xv.ravel(), yv.ravel()])
    ns2 = xy_flat.shape[0]

    # time grid
    t = np.linspace(0.0, t_max, n_t + 1)[1:]
    nt = len(t)

    # intensities on flattened grid
    int0 = np.zeros((nt, ns2), dtype=float)
    int1 = np.zeros((nt, ns2), dtype=float)

    seq_t = seq_txy[:, 0]
    seq_s = seq_txy[:, 1:]

    for k, tk in enumerate(t):
        mask = seq_t < tk
        hist_t = seq_t[mask]
        hist_s = seq_s[mask]
        hist_m = seq_types[mask]

        for j, (xx, yy) in enumerate(xy_flat):
            s = np.array([xx, yy], dtype=float)
            int0[k, j] = lam.value(tk, hist_t, s, hist_s, 0, hist_m)
            int1[k, j] = lam.value(tk, hist_t, s, hist_s, 1, hist_m)

    # pack into df_np
    t_rep = np.repeat(t, ns2)
    XY_tile = np.tile(xy_flat, (nt, 1))
    df_np = np.column_stack([t_rep, XY_tile, int0.reshape(-1), int1.reshape(-1)])

    return df_np, ns, ns2, t


def save_cumulative_timeavg_maps_true(df_np, ns, ns2, tgrid,
                                      type_idx,
                                      n_slices=10,
                                      save_root="figs/spatial_timeavg_true",
                                      cmap_name="viridis",
                                      dpi=300,
                                      label_size=20,
                                      tick_size=18):
    """
    TRUE cumulative time-averaged spatial intensity maps for a single type.

    For type_idx ∈ {0,1}, we:
      - reconstruct λ_k(t_i, s_j) from df_np,
      - build cumulative time-averages up to a sequence of times τ_1, ..., τ_{n_slices},
      - save one figure per τ_m (so n_slices figures),
      - use a *global* vmin/vmax across all slices for this type,
      - return (vmin_global, vmax_global) so you can reuse for FITTED.

    df_np columns: [t, x, y, intensity0, intensity1]
    """

    # basic shapes
    nt = len(tgrid)
    assert df_np.shape[0] == nt * ns2, "df_np shape inconsistent with ns and tgrid"

    # extract intensities for this type as (nt, ns2)
    col = 3 + type_idx
    int_flat = df_np[:, col].reshape(nt, ns2)   # each row: one time, flattened grid

    # choose slice times (cumulative endpoints)
    t_min = float(tgrid[0])
    t_max = float(tgrid[-1])
    # e.g. for [0,100] and n_slices=10 → approx 10,20,...,100
    slice_times = np.linspace(t_max / n_slices, t_max, n_slices)

    # determine indices in tgrid for each slice time
    slice_indices = [int(np.searchsorted(tgrid, tau, side='right') - 1)
                     for tau in slice_times]
    slice_indices = [max(0, min(nt - 1, idx)) for idx in slice_indices]

    # precompute cumulative averages and track global vmin/vmax
    cum_maps = []  # list of (idx, tau, map_ns_ns)
    vmin_global = np.inf
    vmax_global = -np.inf

    for idx, tau in zip(slice_indices, slice_times):
        # cumulative average up to index idx (inclusive)
        cum_mean_flat = int_flat[:idx+1, :].mean(axis=0)  # shape (ns2,)
        cum_map = cum_mean_flat.reshape(ns, ns)

        cum_maps.append((idx, tau, cum_map))

        vmin_global = min(vmin_global, float(cum_map.min()))
        vmax_global = max(vmax_global, float(cum_map.max()))

    if vmax_global <= vmin_global:
        vmax_global = vmin_global + 1e-12

    type_label = f"type{type_idx + 1}"

    print(f"[TRUE cum-avg] {type_label}: global vmin={vmin_global:.6g}, vmax={vmax_global:.6g}")
    print(f"[TRUE cum-avg] slice times (approx): {', '.join(f'{tau:.1f}' for tau in slice_times)}")
    print("You can reuse these vmin/vmax for FITTED cumulative maps for this type.")

    # set up colormap and global norm for colorbar
    cmap = mpl.colormaps[cmap_name]
    norm_global = colors.Normalize(vmin=vmin_global, vmax=vmax_global)

    vmin_type = vmin_global
    vmax_type = vmax_global

    # grid coordinates
    x = df_np[0:ns2, 1].reshape(ns, ns)
    y = df_np[0:ns2, 2].reshape(ns, ns)

    # output directory
    save_root = Path(save_root)
    save_dir = save_root / type_label
    save_dir.mkdir(parents=True, exist_ok=True)

    # ticks on bar
    n_ticks = 5
    ticks = np.linspace(vmin_global, vmax_global, n_ticks)

    print(f"[TRUE cum-avg] saving {len(cum_maps)} frames → {save_dir}")

    for m, (idx, tau, cum_map) in enumerate(cum_maps):
        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.pcolormesh(x, y, cum_map,
                           shading='auto', cmap=cmap, norm=norm_global)

        # title: cumulative average up to time tau (integer-ish)
        tau_int = int(round(tau))
        ax.set_title(f"{type_label}  t={tau_int}", fontsize=label_size)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        cbar = plt.colorbar(im, ax=ax, location='right', format="%.3f")
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=tick_size)

        savename = f"{m:02d}.jpg"   # 00, 01, ..., 09 for n_slices=10
        plt.savefig(save_dir / savename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"[TRUE cum-avg] finished cumulative maps for {type_label} → {save_dir}")
    return vmin_type, vmax_type


def main():
    ap = argparse.ArgumentParser("True spatial intensity frames from first train sequence")
    ap.add_argument("--data_dir", type=str, required=False,
                    default='./data/bistpp_bivariate_1',
                    help="Folder containing train.pkl (e.g., data/bistpp_bivariate_1)")
    ap.add_argument("--tag", type=str, default="bivariate_1",
                    choices=["bivariate_1", "bivariate_2", "bivariate_3", "bivariate_4"],
                    required=False)
    ap.add_argument("--mu0", type=float, default=0.1)
    ap.add_argument("--mu1", type=float, default=0.1)
    ap.add_argument("--n_grid", type=int, default=40)
    ap.add_argument("--t_max", type=float, default=100.0)
    ap.add_argument("--n_t",   type=int, default=1000)
    ap.add_argument("--frame_factor", type=float, default=1.0,
                    help="Fraction of frames to render (0–1)")
    args = ap.parse_args()

    train_pkl = Path(args.data_dir) / "train.pkl"
    if not train_pkl.exists():
        raise FileNotFoundError(f"Could not find {train_pkl}")

    seq_txy, seq_types = load_first_sequence(train_pkl)
    kernel = build_kernel_set(args.tag)
    lam = BivariateHawkesLam(mu=[args.mu0, args.mu1], kernel=kernel, maximum=5e2)

    # build df_np
    df_np, ns, ns2, tgrid = build_spatial_df_np(
        seq_txy, seq_types, lam,
        n_grid=args.n_grid,
        t_max=args.t_max,
        n_t=args.n_t
    )

    # TRUE cumulative time-averaged spatial maps (10 slices per type)
    save_root_cum = f"figs/bistpp_{args.tag}_true_spatial_timeavg"

    # save frames for type 1 (index 0)
    vmin0_cum, vmax0_cum = save_cumulative_timeavg_maps_true(
        df_np, ns, ns2, tgrid,
        type_idx=0,
        n_slices=10,
        save_root=save_root_cum,
        cmap_name="viridis",
        dpi=300,
        label_size=20,
        tick_size=18
    )

    # save frames for type 2 (index 1)
    vmin1_cum, vmax1_cum = save_cumulative_timeavg_maps_true(
        df_np, ns, ns2, tgrid,
        type_idx=1,
        n_slices=10,
        save_root=save_root_cum,
        cmap_name="viridis",
        dpi=300,
        label_size=20,
        tick_size=18
    )

    print(f"[TRUE cum-avg] type1 cumulative vmin={vmin0_cum:.6g}, vmax={vmax0_cum:.6g}")
    print(f"[TRUE cum-avg] type2 cumulative vmin={vmin1_cum:.6g}, vmax={vmax1_cum:.6g}")
    print("Reuse these vmin/vmax for FITTED cumulative maps.")


    # plot_temporal_average(
    #     seq_txy, seq_types, lam,
    #     out_path=f"figs/bistpp_{args.tag}_true.pdf",
    #     ylim=(0.4, 2.4),
    #     yq=0.99,
    #     label_size=20, tick_size=18, legend_size=18,
    #     rug_base=0.025, rug_height=0.020, rug_lw=1.0,
    #     line_lw=1.6, figsize=(10, 4), dpi=300
    # )


if __name__ == "__main__":
    main()

# /project/jun/ccchukw2/STNHP/examples/



 # sbatch -J plot_biv4 -o outputs/plot_biv4_%j.txt -N1 -n1 -t 05:00:00 --mem=8G --export=ALL,MPLBACKEND=Agg \
 #   --wrap='mkdir -p outputs; python avg_true_intensity.py --data_dir ./data/bistpp_bivariate_4 --tag bivariate_4 --mu0 0.1 --mu1 0.1'



# scp 0013.jpg 0024.jpg 0033.jpg 0044.jpg 0053.jpg ../true/type1
