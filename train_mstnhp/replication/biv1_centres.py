"""Find a two-centre Biv1 example from truth alone, then compare its matched fit."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label, maximum_filter

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent)]
from evaluation.intensity_metrics import BIV1_PARAMETERS, biv1_reference, spatial_grid

BASE = BIV1_PARAMETERS['baseline'][0]  # type 0 throughout
HORIZON = BIV1_PARAMETERS['horizon']


def snapshot_time(times, i):
    end = times[i + 1] if i + 1 < len(times) else HORIZON
    return times[i] + min(0.01, (end - times[i]) / 2)


def peak_pair(field, grid):
    """Two interior maxima in distinct 8-connected high-intensity regions."""
    peaks = (field == maximum_filter(field, size=3))
    peaks[[0, -1], :] = False
    peaks[:, [0, -1]] = False
    indices = np.flatnonzero(peaks)
    flat = field.ravel()
    indices = indices[np.argsort(-flat[indices], kind="stable")]
    if len(indices) < 2:
        return None
    a = indices[0]
    for b in indices[1:]:
        if np.linalg.norm(grid[a] - grid[b]) < 0.5 or flat[b] - BASE < 0.01:
            continue
        threshold = BASE + 0.9 * (flat[b] - BASE)
        regions, _ = label(field >= threshold, structure=np.ones((3, 3)))
        if regions.flat[a] != regions.flat[b]:
            return [grid[a].tolist(), grid[b].tolist()]
    return None


def find_case(sequences):
    grids = {n: spatial_grid(n)[0] for n in (40, 80)}
    for seq, events in enumerate(sequences):
        times = np.array([e['time_since_start'] for e in events])
        xy = np.array([e['location_from_origin'] for e in events])
        marks = np.array([e['type_event'] for e in events])
        for i in range(1, len(times) - 1):
            t = snapshot_time(times, i)
            for n, grid in grids.items():
                field = biv1_reference(t, grid, times, xy, marks)[:, 0].reshape(n, n)
                pair = peak_pair(field, grid)
                if pair is None:
                    break
            else:
                return seq, i, t, pair
    return None


def draw(sequences, seq, i, data, fit, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import yaml
    from evaluation.model_intensity import (
        distances_to_history, event_states, fitted_intensity, load_model, sequence_arrays,
    )
    torch.set_num_threads(1)
    record = json.loads((fit / 'completed.json').read_text())
    if record['simulation_seed'] != data['seed']:
        raise ValueError('Fit and simulation seeds differ.')
    cfg = yaml.safe_load((fit / 'train.yaml').read_text())
    exp = cfg['STNHP_train']
    specs = cfg['data'][exp['base_config']['dataset_id']]['data_specs']
    model, dims = load_model(fit / record['best_checkpoint'], exp['model_config'], specs)
    times, xy, marks = sequence_arrays(sequences[seq], 2, HORIZON)
    states = event_states(model, times, xy, marks)
    grid = spatial_grid(80)[0]
    query, hist = torch.from_numpy(grid), torch.from_numpy(xy)
    fields = []
    for j in (i - 1, i, i + 1):
        t = snapshot_time(times, j)
        truth = biv1_reference(t, grid, times, xy, marks)[:, 0]
        distances = distances_to_history(query, hist[:j + 1], model.n_comps)
        fit_values = fitted_intensity(model, states[j], t - times[j], distances)[:, 0]
        fields.append((truth.reshape(80, 80), fit_values.reshape(80, 80)))
    print("Middle snapshot, fitted peak pair under the truth screening rule:",
          peak_pair(fields[1][1], grid), flush=True)
    lo, hi = np.min(fields), np.max(fields)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for col, (j, pair) in enumerate(zip((i - 1, i, i + 1), fields)):
        axes[0, col].set_title(f't = {snapshot_time(times, j):.4f}')
        for ax, field in zip(axes[:, col], pair):
            img = ax.imshow(field, origin='lower', extent=(-1, 1, -1, 1), vmin=lo,
                            vmax=hi, cmap='viridis', interpolation='nearest')
            ax.plot(*xy[j], marker='+', markersize=8, color='white')
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
    axes[0, 0].set_ylabel('True')
    axes[1, 0].set_ylabel('MSTNHP')
    fig.colorbar(img, ax=axes, label='Type 0 intensity', shrink=.8)
    fig.savefig(out / 'biv1_centres.pdf')
    plt.close(fig)
    print('Checkpoint:', fit / record['best_checkpoint'], '| n_comps:', dims['n_comps'])
    print(out / 'biv1_centres.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicate', type=int, default=1)
    parser.add_argument('--truth-only', action='store_true')
    args = parser.parse_args()
    name = f'rep_{args.replicate:03d}'
    folder = TRAIN / 'data/biv1_replicates' / name
    data = json.loads((folder / 'generation.json').read_text())
    if data['setting'] != 'bivariate_1' or data['seed'] != 10000 + args.replicate:
        raise ValueError('Expected the matching Biv1 dataset.')
    with (folder / 'test.pkl').open('rb') as stream:
        sequences = pickle.load(stream)['test']
    found = find_case(sequences)
    if found is None:
        print(f'No qualifying interior type 0 peak pair across {len(sequences)} sequences.')
        return
    seq, i, t, pair = found
    print(f'{name}, type 0, sequence {seq}, event {i} (zero-based), t={t:.8f}, centres={pair}', flush=True)
    if not args.truth_only:
        out = TRAIN / 'evaluation_results/biv1_centres' / name
        out.mkdir(parents=True, exist_ok=True)
        draw(sequences, seq, i, data, TRAIN / 'checkpoints/biv1_replicates' / name, out)


if __name__ == '__main__':
    main()
