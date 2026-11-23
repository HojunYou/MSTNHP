#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

GROUPS = [(k, method) for k in (0, 1) for method in ("MSTNHP", "background")]
METRICS = [
    ("temporal_rmse", "Temporal RMSE", 1, 4),
    ("temporal_relative_l2", r"Temporal relative $L^2$ (\%)", 100, 2),
    ("spacetime_rmse", "Space-time RMSE", 1, 4),
    ("spacetime_relative_l2", r"Space-time relative $L^2$ (\%)", 100, 2),
    ("spatial_tv", "Spatial total variation", 1, 4),
]
RELATIVE_PANELS = [
    ("temporal_relative_l2", "biv1_temporal.pdf"),
    ("spacetime_relative_l2", "biv1_spacetime.pdf"),
]
RMSE_PANELS = [
    ("temporal_rmse", "biv1_temporal_rmse.pdf"),
    ("spacetime_rmse", "biv1_spacetime_rmse.pdf"),
]


def read_groups(folder):
    scores = {g: {} for g in GROUPS}
    with (folder / "replicate_scores.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["event_type"]), row["method"])
            rep = int(row["replicate"])
            if key not in scores or rep in scores[key]:
                raise ValueError(f"Unexpected or duplicate score row for {key}, replicate {rep}")
            values = {m: float(row[m]) for m, *_ in METRICS}
            if not all(math.isfinite(v) and v >= 0 for v in values.values()):
                raise ValueError(f"Invalid metric for {key}, replicate {rep}")
            scores[key][rep] = values

    summary = {}
    with (folder / "replicate_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["event_type"]), row["method"])
            if key not in scores or key in summary or int(row["replicates"]) != 50:
                raise ValueError(f"Invalid or duplicate summary for {key}")
            summary[key] = row
    for key in GROUPS:
        if set(scores[key]) != set(range(1, 51)) or key not in summary:
            raise ValueError(f"Expected all 50 replicates and a summary for {key}")
        for metric, *_ in METRICS:
            values = [v[metric] for v in scores[key].values()]
            for fn, suffix in ((mean, "mean"), (stdev, "sd")):
                recorded = float(summary[key][f"{metric}_{suffix}"])
                if not math.isclose(fn(values), recorded, rel_tol=1e-8, abs_tol=1e-12):
                    raise ValueError(f"Scores disagree with summary for {key}, {metric}, {suffix}")
    return scores, summary


def write_table(summary, out):
    lines = [r"\begin{table}[htbp]", r"\centering\small\color{blue}",
             r"\caption{Biv1 intensity recovery. Entries are mean $\pm$ sample standard deviation across 50 independently fitted datasets. Each replicate score averages 50 held-out sequence errors. Background uses the constant intensity $0.1$ for each event type.}",
             r"\label{tab:biv1-recovery}", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Type 0} & \multicolumn{2}{c}{Type 1} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Metric & MSTNHP & Background & MSTNHP & Background \\", r"\midrule"]
    for metric, label, scale, digits in METRICS:
        entries = []
        for key in GROUPS:
            row = summary[key]
            avg = scale * float(row[metric + "_mean"])
            sd = scale * float(row[metric + "_sd"])
            entries.append(fr"${avg:.{digits}f}\pm{sd:.{digits}f}$")
        lines.append(" & ".join([label] + entries) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "biv1_table.tex").write_text("\n".join(lines) + "\n")


def write_panels(scores, out):
    # Pad the common axis by at least one percentage point above its largest dot.
    highest = max(100 * row[m] for group in scores.values()
                  for row in group.values() for m, _ in RELATIVE_PANELS)
    percent_top = max(5, 5 * math.ceil((highest + 1) / 5))
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    for metric, filename in RELATIVE_PANELS + RMSE_PANELS:
        relative = (metric, filename) in RELATIVE_PANELS
        scale = 100 if relative else 1
        if relative:
            top = percent_top
        else:
            peak = max(row[metric] for group in scores.values() for row in group.values())
            top = peak * 1.08 if peak > 0 else 1.0
        fig, ax = plt.subplots(figsize=(3.25, 2.45))
        fig.subplots_adjust(left=.23, right=.97, top=.97, bottom=.24)
        for x, key in zip((1, 1.7, 3, 3.7), GROUPS):
            vals = [scale * scores[key][rep][metric] for rep in range(1, 51)]
            # The same replicate has the same horizontal offset in every group.
            offsets = [0.33 * ((rep * 0.61803398875) % 1 - .5) for rep in range(1, 51)]
            ax.scatter([x + dx for dx in offsets], vals, s=9, color="0.43",
                       alpha=.65, edgecolors="none")
            ax.scatter([x], [mean(vals)], marker="D", s=28, c="black",
                       edgecolors="white", linewidths=.5, zorder=3)
        ax.set(xlim=(.5, 4.2), ylim=(0, top))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=relative))
        ax.set_xticks([1, 1.7, 3, 3.7])
        ax.set_xticklabels(["M", "B", "M", "B"])
        for x, label in ((1.35, "Type 0"), (3.35, "Type 1")):
            ax.text(x, -.14, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=.6)
        fig.savefig(out / filename)
        plt.close(fig)


def write_figure_files(out):
    captions = {
        "temporal_relative_l2": r"Temporal relative $L^2$ error (\%)",
        "spacetime_relative_l2": r"Space-time relative $L^2$ error (\%)",
        "temporal_rmse": "Temporal RMSE (events per unit time)",
        "spacetime_rmse": "Space-time RMSE (events per unit time per unit area)",
    }
    for metric, filename in RELATIVE_PANELS + RMSE_PANELS:
        lines = [
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.55\linewidth]{{figures/biv1/{filename}}}",
            rf"\caption{{{captions[metric]} on the Biv1 test data. "
            r"Within each event type, M denotes MSTNHP and B denotes the constant-background benchmark. "
            r"Each dot is one replicate's mean error over 50 held-out sequences. "
            r"Each diamond is the mean across 50 independently fitted replicates.}",
            rf"\label{{fig:biv1-{metric.replace('_', '-')}}}",
            r"\end{figure}",
        ]
        (out / filename).with_suffix(".tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "train_mstnhp/evaluation_results/biv1_replicates/v1"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    scores, summary = read_groups(args.results)
    out = args.out or args.results / "paper"
    out.mkdir(parents=True, exist_ok=True)
    write_panels(scores, out)
    write_table(summary, out)
    write_figure_files(out)
    print(f"Verified 200 replicate scores and four summary rows. Wrote table and four individual figures to {out}.")


if __name__ == "__main__":
    main()
