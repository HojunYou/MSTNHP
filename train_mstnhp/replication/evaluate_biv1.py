"""Score the validation-selected fit and the background on the replicate's test split."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

TRAIN = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicate', type=int, required=True)
    parser.add_argument('--run', default='v1', help='Choose a new name for revised evaluations.')
    parser.add_argument('--ns', type=int, default=40)
    parser.add_argument('--max-time-step', type=float, default=0.5)
    args = parser.parse_args()
    if args.replicate < 1 or not args.run or Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Expected a positive replicate ID and a simple run name.')
    name = f'rep_{args.replicate:03d}'
    fit = TRAIN / 'checkpoints/biv1_replicates' / name
    data = TRAIN / 'data/biv1_replicates' / name
    record = json.loads((fit/'completed.json').read_text())
    generation = json.loads((data/'generation.json').read_text())
    if record['simulation_seed'] != generation['seed'] or generation['seed'] != 10000 + args.replicate:
        raise ValueError('Dataset and fit seeds do not match the replicate.')
    output = TRAIN / 'evaluation_results/biv1_replicates' / args.run
    output.mkdir(parents=True, exist_ok=True)
    destination = output / name
    if destination.exists():
        raise FileExistsError(f'{destination} exists. Choose a new run name to rescore.')
    with tempfile.TemporaryDirectory(dir=output) as temp:
        stage = Path(temp) / name
        cmd = [sys.executable, str(TRAIN/'evaluation/evaluate_intensity.py'),
               '--config', str(fit/'train.yaml'), '--experiment', 'STNHP_train',
               '--checkpoint', str(fit/record['best_checkpoint']), '--data', str(data/'test.pkl'),
               '--max-sequences', '0', '--ns', str(args.ns), '--max-time-step', str(args.max_time_step),
               '--compare-background', '--output', str(stage)]
        subprocess.run(cmd, check=True)
        summary = json.loads((stage/'summary.json').read_text())
        if summary['sequences_evaluated'] != generation['splits'][2]:
            raise ValueError('Test split size differs from generation record.')
        summary.update(replicate=args.replicate, simulation_seed=generation['seed'],
                       training_seed=record['training_seed'])
        (stage/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        stage.rename(destination)
    print('Published:', destination)


if __name__ == '__main__':
    main()
