
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
import numpy as np


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


#  Save spatial frames

# def save_spatial_frames_true(df_np, ns, ns2,
#                              type_idx,
#                              vmin, vmax,
#                              save_root="figs/spatial_frames_true",
#                              frame_factor=1.0,
#                              CMAP_NAME="viridis",
#                              USE_LOG=False,
#                              GAMMA=0.7,
#                              dpi=200):
#     """
#     Save true spatial intensity frames for a single type (type_idx in {0,1}).
#
#     - df_np: output from build_spatial_df_np (columns: t, x, y, intensity0, intensity1)
#     - ns, ns2: spatial grid resolution and ns^2
#     - type_idx: 0 or 1
#     - vmin, vmax: global z-scale for this type (over all frames)
#     """
#
#     # subfolder: figs/bistpp_{tag}_true_spatial_frames/type1 or type2
#     type_label = f"type{type_idx + 1}"  # type1 for index 0, type2 for index 1
#     save_root = Path(save_root)
#     save_dir = save_root / type_label
#     save_dir.mkdir(parents=True, exist_ok=True)
#
#     max_frame = int(df_np.shape[0] // ns2)
#     frame_factor = float(max(0.0, min(1.0, frame_factor)))
#     n_frames_to_draw = int(max_frame * frame_factor)
#
#     cmap = mpl.colormaps[CMAP_NAME]
#
#     # single norm reused for all frames of this type
#     if USE_LOG:
#         norm = colors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, 1e-6))
#     else:
#         norm = colors.PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax)
#
#     col = 3 + type_idx  # intensity column for this type
#
#     print(f"[TRUE] {type_label}: using global z-scale vmin={vmin:.6g}, vmax={vmax:.6g}")
#     print(f"[TRUE] saving {n_frames_to_draw} frames to {save_dir}")
#
#     for i in range(n_frames_to_draw):
#         start, stop = i * ns2, (i + 1) * ns2
#         if stop > df_np.shape[0]:
#             break
#
#         x_block = df_np[start:stop, 1].reshape(ns, ns)
#         y_block = df_np[start:stop, 2].reshape(ns, ns)
#         frame = df_np[start:stop, col].reshape(ns, ns)
#
#         t_cur = float(df_np[start, 0])
#         t_label_int = int(round(t_cur))  # integer time in the title
#
#         fig, ax = plt.subplots(figsize=(5, 4))
#
#         p = ax.pcolormesh(x_block, y_block, frame,
#                           shading='auto', cmap=cmap, norm=norm)
#
#         ax.set_title(f"{type_label}  t={t_label_int}", fontsize=20)
#
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.tick_params(axis='both', which='both', length=0)
#         for s in ax.spines.values():
#             s.set_visible(False)
#
#         cbar = plt.colorbar(p, ax=ax, location='right', format='%.3f')
#         cbar.ax.tick_params(labelsize=18)
#
#         savename = f"{i:04d}.jpg"
#         plt.savefig(save_dir / savename, dpi=dpi, bbox_inches='tight')
#         plt.close(fig)
#
#     print(f"[TRUE] finished saving frames for {type_label} → {save_dir}")



# CLI

def save_spatial_frames_true(df_np, ns, ns2,
                             type_idx,
                             vmin, vmax,
                             save_root="figs/spatial_frames_true",
                             frame_factor=1.0,
                             CMAP_NAME="viridis",
                             USE_LOG=False,
                             GAMMA=0.7,
                             dpi=200):
    """
    Save true spatial intensity frames for a single type (type_idx in {0,1}).

    - Image uses *per-frame* vmin/vmax for good contrast.
    - Colorbar uses *global* vmin/vmax (same for all frames of this type),
      so you can reuse the same bar for fitted vs true.

    df_np columns: [t, x, y, intensity0, intensity1]
    """

    # global scale for this type (for the colorbar)
    vmin_global = float(vmin)
    vmax_global = float(vmax)

    type_label = f"type{type_idx + 1}"  # type1 for index 0, type2 for index 1
    save_root = Path(save_root)
    save_dir = save_root / type_label
    save_dir.mkdir(parents=True, exist_ok=True)

    max_frame = int(df_np.shape[0] // ns2)
    frame_factor = float(max(0.0, min(1.0, frame_factor)))
    n_frames_to_draw = int(max_frame * frame_factor)

    cmap = mpl.colormaps[CMAP_NAME]

    # global norm ONLY for the colorbar
    if USE_LOG:
        norm_global = colors.LogNorm(vmin=max(vmin_global, 1e-6),
                                     vmax=max(vmax_global, 1e-6))
    else:
        # linear mapping on the bar, ticks equally spaced in value
        norm_global = colors.Normalize(vmin=vmin_global, vmax=vmax_global)

    # intensity column for this type
    col = 3 + type_idx

    print(f"[TRUE] {type_label}: global z-scale vmin={vmin_global:.6g}, vmax={vmax_global:.6g}")
    print(f"[TRUE] saving {n_frames_to_draw} frames to {save_dir}")

    # pre-build ScalarMappable for colorbar (global scale)
    sm = mpl.cm.ScalarMappable(norm=norm_global, cmap=cmap)
    sm.set_array([])  # needed for colorbar to work

    # choose nice, equally spaced ticks in value-space
    n_ticks = 5
    ticks = np.linspace(vmin_global, vmax_global, n_ticks)

    for i in range(n_frames_to_draw):
        start, stop = i * ns2, (i + 1) * ns2
        if stop > df_np.shape[0]:
            break

        x_block = df_np[start:stop, 1].reshape(ns, ns)
        y_block = df_np[start:stop, 2].reshape(ns, ns)
        frame = df_np[start:stop, col].reshape(ns, ns)

        # per-frame limits for the IMAGE
        vmin_frame = float(frame.min())
        vmax_frame = float(frame.max())
        if vmax_frame <= vmin_frame:
            vmax_frame = vmin_frame + 1e-12

        if USE_LOG:
            norm_frame = colors.LogNorm(vmin=max(vmin_frame, 1e-6),
                                        vmax=max(vmax_frame, 1e-6))
        else:
            norm_frame = colors.PowerNorm(gamma=GAMMA,
                                          vmin=vmin_frame, vmax=vmax_frame)

        t_cur = float(df_np[start, 0])
        t_label_int = int(round(t_cur))  # integer time in the title

        fig, ax = plt.subplots(figsize=(6, 5))

        # IMAGE uses per-frame norm
        p = ax.pcolormesh(x_block, y_block, frame,
                          shading='auto', cmap=cmap, norm=norm_frame)

        ax.set_title(f"{type_label}  t={t_label_int}", fontsize=20)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        # COLORBAR uses global norm + global ticks
        cbar = plt.colorbar(sm, ax=ax, location='right', format='%.3f')
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=18)

        savename = f"{i:04d}.jpg"
        plt.savefig(save_dir / savename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    print(f"[TRUE] finished saving frames for {type_label} → {save_dir}")


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

    # compute global vmin/vmax for each type over ALL frames
    int0_vals = df_np[:, 3]
    int1_vals = df_np[:, 4]

    vmin0, vmax0 = float(int0_vals.min()), float(int0_vals.max())
    vmin1, vmax1 = float(int1_vals.min()), float(int1_vals.max())

    print(f"[TRUE] type1 (index 0) global vmin={vmin0:.6g}, vmax={vmax0:.6g}")
    print(f"[TRUE] type2 (index 1) global vmin={vmin1:.6g}, vmax={vmax1:.6g}")
    print("We will reuse these for the fitted spatial maps.")

    save_root = f"figs/bistpp_{args.tag}_true_spatial_frames"

    # save frames for type 1 (index 0)
    save_spatial_frames_true(
        df_np, ns, ns2,
        type_idx=0,
        vmin=vmin0, vmax=vmax0,
        save_root=save_root,
        frame_factor=args.frame_factor,
        CMAP_NAME="viridis",
        USE_LOG=False,
        GAMMA=0.7,
        dpi=200
    )

    # save frames for type 2 (index 1)
    save_spatial_frames_true(
        df_np, ns, ns2,
        type_idx=1,
        vmin=vmin1, vmax=vmax1,
        save_root=save_root,
        frame_factor=args.frame_factor,
        CMAP_NAME="viridis",
        USE_LOG=False,
        GAMMA=0.7,
        dpi=200
    )


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
 #   --wrap='mkdir -p outputs; python true_intensity.py --data_dir ./data/bistpp_bivariate_4 --tag bivariate_4 --mu0 0.1 --mu1 0.1'



# scp 0013.jpg 0024.jpg 0033.jpg 0044.jpg 0053.jpg ../true/type1
