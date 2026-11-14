# Visualization scripts

All 14 entry points initialize paths through `_paths.py` before importing
project modules. They deliberately use `train_mstnhp/` as their working
directory, preserving existing relative paths in scripts and YAML configs.

| Path | Resolved location |
| --- | --- |
| `configs/...` | `train_mstnhp/configs/...` |
| `data/...` | `train_mstnhp/data/...` |
| `checkpoints/...` | `train_mstnhp/checkpoints/...` |
| `logs/...` | `train_mstnhp/logs/...` |
| `../Meeting/...` | `Meeting/...` at the repository root |

Relative command-line paths also use this convention. Use absolute paths
for external data, checkpoints, or output folders. Config filenames,
checkpoint IDs, polygon names, and experiment IDs remain dataset-specific;
the path fix cannot establish that those inputs match each other.

From the repository root:

```bash
python train_mstnhp/viz/mstnhp_manuscript.py --config_dir configs/experiment_config_gtd_pakistan.yaml --ckpt_dir checkpoints/YOUR_RUN/models/saved_model
```

From `train_mstnhp/`:

```bash
python viz/mstnhp_manuscript.py --ckpt_dir checkpoints/YOUR_RUN/models/saved_model
```

From `train_mstnhp/viz/`:

```bash
python mstnhp_manuscript.py --ckpt_dir checkpoints/YOUR_RUN/models/saved_model
```

Module invocation from the repository root is also supported:

```bash
python -m train_mstnhp.viz.mstnhp_manuscript --ckpt_dir checkpoints/YOUR_RUN/models/saved_model
```

The local repository root and `train_mstnhp/` are added to Python's import
path. The training-side `stpp_simulator/` takes precedence over the root
copy. `mstnhp_visualization.py` now respects its existing `--save_root`
argument instead of using a hard-coded output directory.

These are legacy plotting/prediction programs. Some select the first training
batch and adjust time samples for display. They are not the quantitative
held-out evaluation pipeline proposed in `../../docs/revision_plan.md`.
