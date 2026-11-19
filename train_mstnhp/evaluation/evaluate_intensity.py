"""CPU pilot: Biv 1 raw temporal and space--time intensity errors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = TRAINING_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(TRAINING_DIR))

import numpy as np
import torch
import yaml
from evaluation.intensity_metrics import BIV1_PARAMETERS, IntensityErrors, biv1_reference, spatial_grid, time_quadrature
from evaluation.model_intensity import load_model, sequence_arrays, event_states, distances_to_history, fitted_intensity


def local_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (TRAINING_DIR/path).resolve()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment_config_bistpp_bivariate_1.yaml')
    parser.add_argument('--experiment', default='STNHP_eval')
    parser.add_argument('--checkpoint', help='Override the checkpoint path; dimensions are inferred from its weights.')
    parser.add_argument('--data', help='Override the test pickle path (must still be Biv 1).')
    parser.add_argument('--output', default='evaluation_results/biv1_pilot')
    parser.add_argument('--start-sequence', type=int, default=0)
    parser.add_argument('--max-sequences', type=int, default=1, help='0 evaluates all remaining test sequences.')
    parser.add_argument('--ns', type=int, default=20, help='Spatial grid cells per axis.')
    parser.add_argument('--max-time-step', type=float, default=0.5)
    parser.add_argument('--time-order', type=int, default=2, help='Gauss nodes per time subinterval.')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.ns < 2 or args.max_time_step <= 0 or not np.isfinite(args.max_time_step) or args.time_order < 1 or args.threads < 1:
        parser.error('Invalid grid, time quadrature, or thread count.')
    if args.start_sequence < 0 or args.max_sequences < 0:
        parser.error('Sequence indices/counts must be nonnegative.')
    torch.set_num_threads(args.threads)
    config_path = local_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    experiment = config[args.experiment]
    dataset_id = experiment['base_config']['dataset_id']
    if dataset_id != 'bistpp_bivariate_1':
        raise ValueError('This first evaluator implements the Biv 1 reference only.')
    dataset_config = config['data'][dataset_id]
    checkpoint_value = args.checkpoint or experiment['model_config'].get('pretrained_model_dir')
    if not checkpoint_value:
        raise ValueError('Supply --checkpoint or a pretrained_model_dir in the config.')
    checkpoint = local_path(checkpoint_value)
    data_path = local_path(args.data or dataset_config['test_dir'])
    out = local_path(args.output)
    if any((out/name).exists() for name in ('sequence_metrics.csv', 'summary.json')):
        raise FileExistsError('Results already exist. Choose a new --output directory.')
    model, dimensions = load_model(checkpoint, experiment['model_config'], dataset_config['data_specs'])
    if dimensions['num_event_types'] != 2:
        raise ValueError('Biv 1 requires two event types; hidden size is unrestricted.')
    with data_path.open('rb') as stream:
        data = pickle.load(stream)
    sequences = data['test']
    if not sequences or args.start_sequence >= len(sequences):
        raise ValueError('No selected test sequences.')
    stop = len(sequences) if args.max_sequences == 0 else min(len(sequences), args.start_sequence+args.max_sequences)
    grid, spatial_weights = spatial_grid(args.ns)
    query_tensor = torch.as_tensor(grid, dtype=torch.float64)
    horizon = BIV1_PARAMETERS['horizon']
    print('Checkpoint dimensions:', dimensions, flush=True)
    print('Test sequences {}..{} of {}; CPU float64; {}x{} spatial grid.'
          .format(args.start_sequence, stop-1, len(sequences), args.ns, args.ns), flush=True)
    print('Window: immediately after first event through T=100, including the final interval.', flush=True)
    rows = []
    started = time.perf_counter()
    for index in range(args.start_sequence, stop):
        times, locations, marks = sequence_arrays(sequences[index], dimensions['num_event_types'], horizon)
        states = event_states(model, times, locations, marks)
        history_tensor = torch.as_tensor(locations, dtype=torch.float64)
        errors = IntensityErrors(dimensions['num_event_types'], spatial_weights)
        count = 0
        for i, left in enumerate(times):
            right = times[i+1] if i+1 < len(times) else horizon
            distances = distances_to_history(query_tensor, history_tensor[:i+1], model.n_comps)
            for query_time, weight in time_quadrature(left, right, args.max_time_step, args.time_order):
                fitted = fitted_intensity(model, states[i], query_time-left, distances)
                reference = biv1_reference(query_time, grid, times, locations, marks)
                errors.update(fitted, reference, weight)
                count += 1
        if not np.isclose(errors.duration, horizon-times[0]):
            raise AssertionError('Quadrature weights do not cover the evaluation window.')
        metrics = errors.results()
        for k in range(dimensions['num_event_types']):
            row = dict(sequence_index=index, event_type=k, num_events=len(times),
                       window_start=float(times[0]), window_end=horizon, num_time_queries=count)
            row.update({name: float(value[k]) for name, value in metrics.items()})
            rows.append(row)
            print('seq={} type={} temporal RMSE={:.6g} relL2={:.6g}; space-time RMSE={:.6g} relL2={:.6g}'
                  .format(index, k, row['temporal_rmse'], row['temporal_relative_l2'],
                          row['spacetime_rmse'], row['spacetime_relative_l2']), flush=True)
    summary = dict(
        setting='bivariate_1', reference_parameters=BIV1_PARAMETERS,
        dimensions=dimensions, split='test', sequences_evaluated=stop-args.start_sequence,
        total_test_sequences=len(sequences), first_sequence=args.start_sequence,
        window='condition on first event; evaluate (t_first, 100), including post-last-event tail',
        ns=args.ns, max_time_step=args.max_time_step, time_order=args.time_order,
        dtype='float64', device='cpu', torch_version=str(torch.__version__), threads=args.threads,
        config=str(config_path), config_sha256=digest(config_path),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        data=str(data_path), data_sha256=digest(data_path),
        model_source_sha256=digest(REPO_DIR/'easy_tpp/model/torch_model/torch_stnhp.py'),
        elapsed_seconds=time.perf_counter()-started,
        interpretation='Means over test sequences from ONE fitted model; not independent dataset replications.',
        mean_over_sequences={},
    )
    for k in range(dimensions['num_event_types']):
        selected = [row for row in rows if row['event_type'] == k]
        summary['mean_over_sequences'][str(k)] = {
            name: float(np.mean([row[name] for row in selected])) for name in metrics
        }
    # Biv 1 has positive baseline, so all four errors should be defined.
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'sequence_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/'summary.json').write_text(serialized+'\n')
    print('Saved results:', out, flush=True)


if __name__ == '__main__':
    main()
