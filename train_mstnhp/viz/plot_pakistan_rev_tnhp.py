"""Temporal NHP version of the four-process Pakistan intensity plot."""
from _pakistan_rev import run


def temporal_rates(model, batch, indices, elapsed):
    _, states = model.forward(batch.values())
    hidden = model.compute_states_at_sample_times(states[:, indices], elapsed[None, :, :])
    return model.layer_intensity(hidden)[0]


if __name__ == "__main__":
    run("NHP", temporal_rates)
