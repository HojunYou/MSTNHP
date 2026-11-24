"""Polygon-integrated temporal intensity from the four-process MSTNHP fit."""
from _pakistan_rev import run


def temporal_rates(model, batch, indices, elapsed):
    import torch
    from shapely.geometry import Point
    from evaluation.model_intensity import distances_to_history

    ns = 40
    axis = -1 + (torch.arange(ns, dtype=torch.float32) + .5) * 2 / ns
    grid = torch.cartesian_prod(axis, axis)
    inside = [model.polygon.contains(Point(x, y)) for x, y in grid.tolist()]
    grid = grid[torch.tensor(inside)]
    if len(grid) == 0:
        raise ValueError("No spatial grid cells inside the normalized polygon.")

    history = batch["space_seqs"][0]
    _, decays, outputs = model.forward(batch.values())
    curves = []
    for i, dt in zip(indices.tolist(), elapsed):
        distances = distances_to_history(grid, history[:i + 1], model.n_comps)
        time = dt[:, None].expand(-1, len(grid)).reshape(-1)
        space = distances.repeat(len(dt), 1)
        hidden = model.compute_states_at_sample_points(
            decays[:, i:i + 1], outputs[:, i:i + 1], time, space)
        local_rates = model.layer_intensity(hidden).reshape(len(dt), len(grid), model.num_event_types)
        curves.append(local_rates.sum(dim=1) * (2 / ns) ** 2)
    return torch.stack(curves)


if __name__ == "__main__":
    run("STNHP", temporal_rates)
