"""Evaluate the existing STNHP without changing model or training code."""

from types import SimpleNamespace
import numpy as np
import torch


def load_model(checkpoint, model_options, data_specs):
    from easy_tpp.model.torch_model.torch_stnhp import STNHP

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or 'layer_type_emb.weight' not in state:
        raise ValueError("Expected a plain STNHP state_dict checkpoint.")
    embedding = state['layer_type_emb.weight']
    readout = state['layer_intensity.0.weight']
    decay = state['rnn_cell.layer_decay.0.weight']
    hidden = embedding.shape[1]
    types = readout.shape[0]
    if readout.shape[1] != hidden or decay.shape[1] != 2*hidden:
        raise ValueError("Inconsistent hidden dimensions in checkpoint.")
    if decay.shape[0] % hidden:
        raise ValueError("Decay-layer output is not divisible by hidden size.")
    n_comps = decay.shape[0]//hidden - 1
    if n_comps < 1:
        raise ValueError("Checkpoint has no spatial components.")
    dimensions = dict(hidden_size=hidden, num_event_types=types, n_comps=n_comps)
    for key in ('hidden_size', 'n_comps', 'num_event_types'):
        configured = model_options.get(key)
        if configured is not None and int(configured) != dimensions[key]:
            raise ValueError("Config {}={} disagrees with checkpoint {}. Use the matching run config."
                             .format(key, configured, dimensions[key]))
    if int(data_specs['num_event_types']) != types:
        raise ValueError("Dataset event-type count disagrees with checkpoint.")
    if model_options.get('polygon_path'):
        raise ValueError("This Biv 1 evaluator uses a square, not a polygon domain.")
    specs = dict(model_options.get('model_specs') or {})
    has_bias = 'layer_intensity.0.bias' in state
    if 'bias' in specs and bool(specs['bias']) != has_bias:
        raise ValueError("Config intensity bias disagrees with checkpoint.")
    specs['bias'] = has_bias
    # beta is not stored in state_dict: retain the actual run configuration.
    options = SimpleNamespace(
        hidden_size=hidden, num_event_types=types, num_event_types_pad=embedding.shape[0],
        pad_token_id=int(data_specs['pad_token_id']), n_comps=n_comps,
        loss_integral_num_sample_per_step=model_options.get('loss_integral_num_sample_per_step', 50),
        spatial_npoints=model_options.get('spatial_npoints', 10),
        model_specs=specs, thinning=None, gpu=-1, polygon_path=None,
    )
    model = STNHP(options)
    model.load_state_dict(state, strict=True)
    model = model.double().eval()
    return model, dict(**dimensions, embedding_rows=embedding.shape[0],
                       intensity_bias=has_bias, softplus_beta=specs.get('beta', 1.0))


def sequence_arrays(sequence, num_types, horizon):
    if not sequence:
        raise ValueError("Empty sequence has no first-event conditioning time.")
    times = np.asarray([e['time_since_start'] for e in sequence], dtype=float)
    locations = np.asarray([e['location_from_origin'] for e in sequence], dtype=float)
    raw_marks = np.asarray([e['type_event'] for e in sequence], dtype=float)
    if locations.shape != (len(sequence), 2):
        raise ValueError("Expected planar event locations.")
    if not all(np.all(np.isfinite(x)) for x in (times, locations, raw_marks)):
        raise ValueError("Nonfinite event data.")
    if np.any(raw_marks != np.floor(raw_marks)):
        raise ValueError("Event types must be integers.")
    marks = raw_marks.astype(np.int64)
    if np.any(marks < 0) or np.any(marks >= num_types):
        raise ValueError("Event type outside model range.")
    if times[0] < 0 or times[-1] >= horizon or np.any(np.diff(times) <= 0):
        raise ValueError("Require strictly increasing synthetic times in [0,T), without ties.")
    if np.any(locations < -1) or np.any(locations > 1):
        raise ValueError("Locations outside the Biv 1 generating square [-1,1]^2.")
    supplied = np.asarray([e['time_since_last_event'] for e in sequence])
    if len(times) > 1 and not np.allclose(supplied[1:], np.diff(times), rtol=1e-5, atol=1e-7):
        raise ValueError("Stored time differences disagree with timestamps.")
    return times, locations, marks


def distances_to_history(query, history, n_comps):
    """Match forward(): left zero padding, then oldest-to-newest centres."""
    count = min(len(history), n_comps)
    result = torch.zeros((len(query), n_comps), dtype=query.dtype, device=query.device)
    if count:
        result[:, -count:] = torch.linalg.vector_norm(query[:, None, :] - history[-count:][None, :, :], dim=-1)
    return result


@torch.no_grad()
def event_states(model, times, locations, marks):
    """Cache the state after EVERY event, including the final event.

    Uses the original embedding, cell update and decay methods. Matches the
    training forward pass on its N-1 intervals, extending it through event N.
    """
    dtype = next(model.parameters()).dtype
    places = torch.as_tensor(locations, dtype=dtype)
    h, c, cbar = model.init_state(1)
    h, c, cbar = h.to(dtype), c.to(dtype), cbar.to(dtype)
    states = []
    for i, mark in enumerate(marks):
        embedding = model.layer_type_emb(torch.tensor([int(mark)], dtype=torch.long))
        c, cbar, decay, output = model.rnn_cell(embedding, h, c, cbar)
        states.append((c, cbar, decay, output))
        if i+1 < len(times):
            dt = torch.tensor([[times[i+1]-times[i]]], dtype=dtype)
            ds = distances_to_history(places[i+1:i+2], places[:i+1], model.n_comps)
            c, h = model.rnn_cell.decay(c, cbar, decay, output, dt, ds)
    return states


@torch.no_grad()
def fitted_intensity(model, state, delta_time, distances):
    cell, baseline, decay, output = state
    dt = torch.full((len(distances), 1), float(delta_time), dtype=distances.dtype)
    _, hidden = model.rnn_cell.decay(cell, baseline, decay, output, dt, distances)
    return model.layer_intensity(hidden).cpu().numpy()
