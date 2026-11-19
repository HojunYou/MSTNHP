"""Generate one seeded Biv1 dataset or train the existing model on it."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

TRAIN = Path(__file__).resolve().parents[1]
REPO = TRAIN.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TRAIN))


def save_atomic(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        partial = Path(output.name)
        pickle.dump(value, output, protocol=4)
    os.replace(partial, path)


def generate(folder, seed, split_sizes=(900, 50, 50), chunk_size=25):
    import numpy as np
    import pandas as pd
    from final_mstpp_simulator import bivariate_simulation as legacy
    from final_mstpp_simulator.biv1_thinning import Biv1Thinning

    manifest = dict(setting="bivariate_1", seed=seed, splits=list(split_sizes),
                    chunk_size=chunk_size, min_events=5, bound=500, horizon=100,
                    spatial_domain=[[-1, 1], [-1, 1]], numpy=np.__version__,
                    simulation_version=1)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "generation.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        # RNG resume needs the recorded NumPy version. Completed datasets
        # can also be consumed on a different machine.
        if (folder / "READY").is_file():
            manifest["numpy"] = previous.get("numpy")
        if previous != manifest:
            raise ValueError(f"Existing simulation parameters differ in {folder}")
    else:
        # No old unlabelled files should be silently adopted.
        if any(folder.iterdir()):
            raise ValueError(f"Unidentified files already exist in {folder}")
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if (folder / "READY").is_file():
        for name in ("train", "dev", "test", "points"):
            if not (folder / (name + ".pkl")).is_file():
                raise FileNotFoundError(folder / (name + ".pkl"))
        return

    simulator = Biv1Thinning(legacy.BivariateHawkesLam(
        [.1, .1], legacy.build_kernel_set("bivariate_1"), maximum=500))
    chunks = folder / "chunks"
    chunks.mkdir(exist_ok=True)
    np.random.seed(seed)
    sequences = []
    types = []
    total = sum(split_sizes)
    for start in range(0, total, chunk_size):
        path = chunks / f"{start:04d}.pkl"
        if path.exists():
            with path.open("rb") as stream:
                points, marks, sizes, rng = pickle.load(stream)
            np.random.set_state(rng)
        else:
            points, marks, sizes = simulator.generate(
                [0., 100.], [[-1., 1.], [-1., 1.]],
                batch_size=min(chunk_size, total - start), min_n_points=5, verbose=False)
            save_atomic(path, (points, marks, sizes, np.random.get_state()))
        for i, count in enumerate(sizes):
            sequences.append(points[i, :count])
            types.append(marks[i, :count])
        print(f"Seed {seed}: {len(sequences)}/{total} sequences", flush=True)

    max_events = max(map(len, sequences))
    padded = np.zeros((total, max_events, 3))
    labels = np.zeros((total, max_events), dtype=np.int64)
    formatted = []
    for i, (events, marks) in enumerate(zip(sequences, types)):
        padded[i, :len(events)] = events
        labels[i, :len(events)] = marks
        frame = pd.DataFrame(events, columns=["time", "x", "y"])
        frame["event_type"] = marks
        formatted.append(legacy.convert_sequence(frame[frame.time > 0]))

    with tempfile.TemporaryDirectory(dir=folder) as stage:
        stage = Path(stage)
        offset = 0
        for split, count in zip(("train", "dev", "test"), split_sizes):
            legacy.dump_easytpp_pickle(stage, split, formatted[offset:offset + count])
            offset += count
        save_atomic(stage / "points.pkl", [padded, labels])
        for path in stage.iterdir():
            os.replace(path, folder / path.name)
    (folder / "READY").touch()
    for path in chunks.glob("*.pkl"):
        path.unlink()
    chunks.rmdir()


def train(folder, checkpoint_root):
    import yaml
    if not (folder / "READY").is_file():
        raise RuntimeError(f"Incomplete dataset in {folder}")
    template = TRAIN / "configs/experiment_config_bistpp_bivariate_1.yaml"
    config = yaml.safe_load(template.read_text())
    run = config["STNHP_train"]
    dataset = config["data"][run["base_config"]["dataset_id"]]
    for key, name in (("train_dir", "train"), ("valid_dir", "dev"), ("test_dir", "test")):
        dataset[key] = str(folder / (name + ".pkl"))
    run["base_config"]["base_dir"] = str(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    completed = checkpoint_root / "completed.json"
    if completed.exists():
        previous = json.loads(completed.read_text())
        saved_model = checkpoint_root / previous["best_checkpoint"]
        if not saved_model.is_file():
            raise FileNotFoundError(saved_model)
        print(f"Already trained: {saved_model}")
        return
    existing = set(checkpoint_root.glob("*/models/saved_model"))
    config_path = checkpoint_root / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, str(TRAIN / "train_mstnhp.py"), "--config_dir",
                    str(config_path), "--experiment_id", "STNHP_train"],
                   cwd=TRAIN, env=env, check=True)
    new_checkpoints = set(checkpoint_root.glob("*/models/saved_model")) - existing
    if len(new_checkpoints) != 1:
        raise RuntimeError(f"Expected one new validation-selected checkpoint under {checkpoint_root}")
    best = new_checkpoints.pop()
    record = dict(best_checkpoint=str(best.relative_to(checkpoint_root)),
                  training_seed=run["trainer_config"]["seed"],
                  simulation_seed=json.loads((folder / "generation.json").read_text())["seed"])
    with tempfile.NamedTemporaryFile(dir=checkpoint_root, mode="w", delete=False) as stream:
        json.dump(record, stream, indent=2)
    os.replace(stream.name, completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "train"))
    parser.add_argument("--replicate", type=int, required=True)
    args = parser.parse_args()
    seed = 10000 + args.replicate
    if args.replicate < 1 or seed >= 2**32:
        parser.error("Use a positive replicate number with a valid NumPy seed.")
    name = f"rep_{args.replicate:03d}"
    data = TRAIN / "data" / "biv1_replicates" / name
    if args.stage == "generate":
        generate(data, seed)
    else:
        train(data, TRAIN / "checkpoints" / "biv1_replicates" / name)
