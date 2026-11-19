"""Form one score per replicate, method, and event type, then summarize replicates."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

METRICS = tuple(f'{domain}_{suffix}' for domain in ('temporal', 'spacetime')
                for suffix in ('mae', 'mse', 'rmse', 'bias', 'relative_l1', 'relative_l2',
                               'iae', 'ise', 'max_abs')) + ('spatial_tv', 'spatial_hellinger')
ROOT = Path(__file__).resolve().parents[1] / 'evaluation_results/biv1_replicates'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(folder, count):
    scores = []
    settings = None
    for r in range(1, count+1):
        run = folder / f'rep_{r:03d}'
        meta = json.loads((run / 'summary.json').read_text())
        protocol = [meta[k] for k in ('setting', 'reference_parameters', 'ns', 'max_time_step',
                                       'time_order', 'window', 'dimensions', 'training_seed')]
        if settings is not None and protocol != settings:
            raise ValueError(f'Different evaluation or training settings in {run}')
        settings = protocol
        n = meta['total_test_sequences']
        if meta['replicate'] != r or meta['sequences_evaluated'] != n or not n:
            raise ValueError(f'Incomplete or mismatched replicate in {run}')
        for method_index, (method, filename) in enumerate((
                ('MSTNHP', 'sequence_metrics.csv'), ('background', 'background.csv'))):
            with (run/filename).open(newline='') as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 2*n:
                raise ValueError(f'Expected two event types per sequence in {run/filename}')
            for k in (0, 1):
                selected = [row for row in rows if int(row['event_type']) == k]
                if sorted(int(row['sequence_index']) for row in selected) != list(range(n)):
                    raise ValueError(f'Missing or duplicated test sequences in {run/filename}')
                x = 1 + 2*k + method_index
                record = dict(replicate=r, method=method, event_type=k, x=x)
                # Deterministic horizontal offsets for display only.
                record['plot_x'] = x + 0.28 * ((r*0.61803398875) % 1 - 0.5)
                record.update({m: mean(float(v[m]) for v in selected) for m in METRICS})
                scores.append(record)
    summary = []
    for x in (1, 2, 3, 4):
        group = [r for r in scores if r['x'] == x]
        row = {k: group[0][k] for k in ('method', 'event_type', 'x')}
        row['replicates'] = count
        for m in METRICS:
            vals = [v[m] for v in group]
            row[m+'_mean'] = mean(vals)
            row[m+'_sd'] = stdev(vals) if count > 1 else ''
        summary.append(row)
    write_csv(folder/'replicate_scores.csv', scores)
    write_csv(folder/'replicate_summary.csv', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='v1')
    parser.add_argument('--replicates', type=int, default=50, help='Require complete replicates 1 through N.')
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error('replicates must be positive')
    summarize(ROOT/args.run, args.replicates)
