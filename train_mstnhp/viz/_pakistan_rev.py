"""Shared data loading and temporal-intensity figure for the Point 4 comparison."""
import argparse
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from _paths import configure_paths

LABELS = ["TTP", "BRA", "BLA", "BLF"]


def draw(times, rates, events, types, path, yscale):
    """Four type-specific intensity curves and an event rug, as in pakistan_4process."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, name in enumerate(LABELS):
        line, = ax.plot(times, rates[:, k], label=name, lw=1.5)
        ax.vlines(events[types == k], .025, .043, color=line.get_color(),
                  linewidth=.8, transform=ax.get_xaxis_transform())
    ax.set(xlabel="Days since first recorded event", ylabel="Intensity (events/day)",
           yscale=yscale)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run(model_id, make_curve):
    train_dir = configure_paths()
    parser = argparse.ArgumentParser(description=f"Plot held-out temporal intensities from {model_id}.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Run's models/saved_model file.")
    parser.add_argument("--config", default="configs/experiment_config_gtd_pakistan.yaml", type=Path)
    parser.add_argument("--yscale", default="log", choices=["linear", "log"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import torch
    from easy_tpp.config_factory.model_config import ModelConfig
    from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
    from easy_tpp.preprocess.data_loader import STPPDataLoader, TPPDataLoader
    from easy_tpp.config_factory.data_config import DataConfig

    # The training YAML supplies the polygon and n_comps, which older saved run
    # configs may omit. Evaluation uses the same architecture on CPU.
    config = yaml.safe_load(args.config.read_text())
    exp = config[f"{model_id}_pakistan_rev_train"]
    data = DataConfig.parse_from_yaml_config(config["data"][exp["base_config"]["dataset_id"]])
    specs = data.data_specs
    model_options = dict(exp["model_config"], gpu=-1, thinning=None, model_id=model_id,
                         num_event_types=specs.num_event_types, num_event_types_pad=specs.num_event_types_pad,
                         event_pad_index=specs.pad_token_id)
    model = TorchBaseModel.generate_model_from_config(ModelConfig(**model_options))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(1)

    loader = TPPDataLoader if model_id == "NHP" else STPPDataLoader
    batch, = loader(data, "torch", batch_size=1, shuffle=False).test_loader()
    with open(data.test_dir, "rb") as source:
        records, = pickle.load(source)["test"]
    for field, key in (("time_seqs", "time_since_start"), ("time_delta_seqs", "time_since_last_event"),
                       ("type_seqs", "type_event")):
        mask = batch["seq_non_pad_mask"][0].bool()
        if not bool(mask.all()) or batch[field][0].tolist() != [r[key] for r in records]:
            raise ValueError(f"Unexpected padding, order, or truncation in the test batch: {field}")
    events = batch["time_seqs"][0]
    types = batch["type_seqs"][0]
    gaps = events.diff()
    if len(events) != 103 or int((gaps == 0).sum()) != 24 or (gaps < 0).any():
        raise ValueError("Expected the 2018 revision split with 103 events and 24 tied gaps.")

    # Endpoints are one-sided limits. Leave same-day events in model.forward(),
    # and plot only intervals with positive elapsed time.
    indices = torch.where(gaps > 0)[0]
    dt = gaps[indices, None] * torch.linspace(0., 1., 20)
    with torch.inference_mode():
        rates = make_curve(model, batch, indices, dt).reshape(-1, len(LABELS)).cpu().numpy()
    times = (events[indices, None] + dt).reshape(-1).numpy()
    if not np.isfinite(rates).all():
        raise ValueError("Nonfinite fitted rates.")
    folder = "tnhp" if model_id == "NHP" else "mstnhp"
    path = args.out or train_dir / "evaluation_results/pakistan_rev" / f"{folder}_test_2018.pdf"
    draw(times, rates, events.numpy(), types.numpy(), path, args.yscale)
    print(f"{folder}: {len(events)} test events, 24 tied gaps. Saved {path.resolve()}")
