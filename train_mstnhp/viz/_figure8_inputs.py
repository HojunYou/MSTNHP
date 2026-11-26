"""Select a tied-event Pakistan sequence and its trained checkpoint for Figure 8."""
import argparse
import pickle
from pathlib import Path

import torch
import yaml

from _paths import configure_paths


def figure_inputs(model_id):
    root = configure_paths()
    from easy_tpp.config_factory.data_config import DataConfig
    from easy_tpp.config_factory.model_config import ModelConfig
    from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", required=True, type=int, choices=(2014, 2018))
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Directory containing models/, relative to train_mstnhp or absolute.")
    parser.add_argument("--weights", required=True, choices=("best", "final"))
    args = parser.parse_args()
    label = {"STNHP": "mstnhp", "NHP": "tnhp"}[model_id]
    filename = {"best": "saved_model", "final": "last_epoch_model"}[args.weights]
    checkpoint = (args.run_dir / "models" / filename).resolve()
    cfg = yaml.safe_load((root / "configs/experiment_config_gtd_pakistan.yaml").read_text())
    experiment = cfg[f"{model_id}_pakistan_rev_train"]
    data = DataConfig.parse_from_yaml_config(cfg["data"][experiment["base_config"]["dataset_id"]])
    specs = data.data_specs
    options = dict(experiment["model_config"], gpu=-1, thinning=None, model_id=model_id,
                   num_event_types=specs.num_event_types, num_event_types_pad=specs.num_event_types_pad,
                   event_pad_index=specs.pad_token_id)
    if model_id == "STNHP":
        # Figure 8(a) sums the entire plotting grid. The polygon is used separately
        # for a count diagnostic in the original script.
        options.update(polygon_path=None, spatial_npoints=40)

    split, index, counts = {2014: ("train", 6, (331, 133)), 2018: ("test", 0, (103, 24))}[args.year]
    with open(data.get_data_dir(split), "rb") as source:
        records = pickle.load(source)[split][index]
    times = [r["time_since_start"] for r in records]
    gaps = [b - a for a, b in zip(times, times[1:])]
    observed = (len(records), gaps.count(0))
    if observed != counts or min(gaps) < 0 or gaps != [r["time_since_last_event"] for r in records[1:]]:
        raise ValueError(f"Unexpected sequence for {args.year}: {observed} events/tied gaps.")
    torch.set_num_threads(1)
    model = TorchBaseModel.generate_model_from_config(ModelConfig(**options))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    output = root / "evaluation_results/pakistan_rev/figure8" / f"{label}_{split}_{args.year}_{args.weights}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label} {split} {args.year}: {counts[0]} events, {counts[1]} tied gaps; weights={checkpoint}")
    return model, records, output
