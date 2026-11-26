"""Replot the Figure 8(a) full-grid temporal intensity using a revision checkpoint."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from _figure8_inputs import figure_inputs


@torch.inference_mode()
def main():
    model, records, output = figure_inputs("STNHP")
    def tensor(field, dtype):
        return torch.tensor([r[field] for r in records], dtype=dtype).unsqueeze(0)

    times = tensor("time_since_start", torch.float32)
    gaps = tensor("time_since_last_event", torch.float32)
    locations = tensor("location_from_origin", torch.float32)
    displacements = tensor("location_from_last_event", torch.float32)
    types = tensor("type_event", torch.long)
    _, decays, states = model.forward((times, gaps, locations, displacements, types, None, None, None))

    ns = model.spatial_npoints
    cells = ns ** 2
    intervals, = model.make_multiple_equal_dtimespace_samples(
        times, [0, 366], locations, num_sample_t=1000)
    sample_times, curves = [], []
    for i, points in enumerate(intervals):
        if points.numel() == 0:
            continue
        absolute = points[:, 0, 0]
        elapsed = absolute - times[0, i]
        # Manuscript spike guard: shift the first sampled frame in this interval.
        if elapsed.any() and elapsed[0] <= 0.01:
            elapsed[:cells] += 0.01
        distance = torch.norm(points[:, :, 1:], p=2, dim=-1)
        hidden = model.compute_states_at_sample_points(
            decays[:, i:i+1], states[:, i:i+1], elapsed, distance)
        rates = model.layer_intensity(hidden).reshape(-1, cells, model.num_event_types).numpy()
        curves.append(rates.sum(axis=1) * (2.0 / ns) ** 2)
        sample_times.append(absolute[::cells].numpy())

    values = np.concatenate(curves)
    t = np.concatenate(sample_times)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_yscale("log", base=10)
    ymin = max(1e-6, float(np.nanmin(values[values > 0])))
    ymax = (10**0.6) * 1.2
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.logspace(np.log10(ymin), np.log10(ymax), num=6))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    for k, label in enumerate(["TTP", "BRA", "BLA", "BLF"]):
        line, = ax.plot(t, values[:, k], label=label, linewidth=1.5)
        ax.vlines(times[0, types[0] == k].numpy(), .025, .043, color=line.get_color(),
                  linewidth=.08, alpha=.95, capstyle="butt", zorder=3, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Day of year", fontsize=18)
    ax.set_ylabel("λ(t)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15, length=6)
    ax.legend(loc="upper right", fontsize=15, frameon=False)
    fig.subplots_adjust(left=.12, bottom=.20, right=.98, top=.95)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
