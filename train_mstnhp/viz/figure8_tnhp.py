"""Replot the temporal-only manuscript curves using a Pakistan revision checkpoint."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from _figure8_inputs import figure_inputs


@torch.inference_mode()
def main():
    model, records, output = figure_inputs("NHP")
    times = torch.tensor([[r["time_since_start"] for r in records]], dtype=torch.float32)
    gaps = torch.tensor([[r["time_since_last_event"] for r in records]], dtype=torch.float32)
    types = torch.tensor([[r["type_event"] for r in records]], dtype=torch.long)
    mask = torch.ones_like(types, dtype=torch.bool)
    type_mask = torch.nn.functional.one_hot(types, num_classes=model.num_event_types_pad).float()
    _, states = model.forward((times, gaps, types, mask, None, type_mask))
    elapsed = model.make_dtime_loss_samples(gaps[:, 1:])
    t = (elapsed + times[:, :-1, None]).numpy().reshape(-1)
    hidden = model.compute_states_at_sample_times(states, elapsed)
    values = model.layer_intensity(hidden).reshape(-1, model.num_event_types).numpy()
    # Preserve the ordering of plotted samples used by the original TNHP plotter.
    order = np.argsort(t)
    t, values = t[order], values[order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_yscale("log")
    positive = values[values > 0]
    ymin = max(1e-6, float(np.nanmin(positive)))
    ymax = float(np.quantile(positive, .99))
    if not np.isfinite(ymax) or ymax <= ymin:
        ymax = max(ymin * 10, 1.0)
    ax.set_ylim(ymin, (10**0.6) * .24)
    ax.set_yticks(np.logspace(np.log10(ymin), np.log10(ymax), num=6))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:.3g}"))
    for k, label in enumerate(["TTP", "BRA", "BLA", "BLF"]):
        line, = ax.plot(t, values[:, k], lw=1.5, label=label)
        ax.vlines(times[0, types[0] == k].numpy(), .025, .043, color=line.get_color(),
                  linewidth=1.0, alpha=.95, capstyle="butt", zorder=3, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Day of year", fontsize=18)
    ax.set_ylabel("λ(t)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    fig.subplots_adjust(left=.12, bottom=.20, right=.98, top=.95)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
