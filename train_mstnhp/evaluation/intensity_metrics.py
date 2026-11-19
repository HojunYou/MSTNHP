"""Weighted errors on raw conditional intensities."""

import numpy as np


class IntensityErrors:
    def __init__(self, num_types, spatial_weights):
        self.weights = np.asarray(spatial_weights, dtype=float)
        if self.weights.ndim != 1 or np.any(self.weights <= 0):
            raise ValueError("Spatial weights must be a positive vector.")
        self.area = float(self.weights.sum())
        self.duration = 0.0
        self.temporal_squared_error = np.zeros(num_types)
        self.temporal_reference_energy = np.zeros(num_types)
        self.spatial_squared_error = np.zeros(num_types)
        self.spatial_reference_energy = np.zeros(num_types)

    def update(self, fitted, reference, time_weight):
        fitted = np.asarray(fitted, dtype=float)
        reference = np.asarray(reference, dtype=float)
        expected = (len(self.weights), len(self.temporal_squared_error))
        if fitted.shape != expected or reference.shape != expected:
            raise ValueError("Intensity arrays must have shape (spatial points, types).")
        if time_weight <= 0 or not np.isfinite(time_weight):
            raise ValueError("Invalid time weight.")
        if not np.all(np.isfinite(fitted)) or not np.all(np.isfinite(reference)):
            raise ValueError("Nonfinite intensity encountered.")
        if np.any(fitted < 0) or np.any(reference < 0):
            raise ValueError("Negative intensity encountered; no clipping is applied.")
        fitted_total = self.weights @ fitted
        true_total = self.weights @ reference
        self.duration += time_weight
        self.temporal_squared_error += time_weight * (fitted_total - true_total)**2
        self.temporal_reference_energy += time_weight * true_total**2
        self.spatial_squared_error += time_weight * (self.weights @ ((fitted-reference)**2))
        self.spatial_reference_energy += time_weight * (self.weights @ (reference**2))

    def results(self):
        if self.duration <= 0:
            raise ValueError("No positive evaluation window.")
        def relative(error, energy):
            result = np.full_like(error, np.nan)
            np.divide(error, energy, out=result, where=energy > 0)
            return np.sqrt(result)
        return dict(
            temporal_rmse=np.sqrt(self.temporal_squared_error/self.duration),
            temporal_relative_l2=relative(self.temporal_squared_error, self.temporal_reference_energy),
            spacetime_rmse=np.sqrt(self.spatial_squared_error/(self.duration*self.area)),
            spacetime_relative_l2=relative(self.spatial_squared_error, self.spatial_reference_energy),
        )


def spatial_grid(ns):
    """Equal-area midpoints on the generating domain [-1,1]^2."""
    if ns < 2:
        raise ValueError("ns must be at least 2.")
    x = -1 + (np.arange(ns)+0.5)*2/ns
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel())), np.full(ns*ns, 4/(ns*ns))


def time_quadrature(left, right, max_step, order):
    """Interior Gauss nodes; caller splits at every event."""
    if right <= left or max_step <= 0 or order < 1:
        raise ValueError("Invalid quadrature interval/settings.")
    edges = np.linspace(left, right, int(np.ceil((right-left)/max_step))+1)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for a, b in zip(edges[:-1], edges[1:]):
        for node, weight in zip(nodes, weights):
            yield (a+b)/2+(b-a)*node/2, (b-a)*weight/2


BIV1_PARAMETERS = dict(
    baseline=[0.1, 0.1], alpha=[[0.25, 0.1], [0.1, 0.25]],
    beta=0.3, sigma2=0.5, spatial_bounds=[[-1, 1], [-1, 1]], horizon=100.0,
)


def biv1_reference(query_time, query_locations, times, locations, marks):
    """Biv 1: normalized Gaussian space, exponential time, all strictly past events.

    alpha[target_type, source_type]; no coordinate transformation is applied.
    Parameters match final_mstpp_simulator/bivariate_simulation.py Biv 1.
    """
    mask = times < query_time
    history_t, history_s, history_k = times[mask], locations[mask], marks[mask]
    p = BIV1_PARAMETERS
    values = np.tile(p['baseline'], (len(query_locations), 1)).astype(float)
    if len(history_t):
        squared_distance = ((query_locations[:, None, :] - history_s[None, :, :])**2).sum(-1)
        spatial = np.exp(-squared_distance/(2*p['sigma2']))/(2*np.pi*p['sigma2'])
        temporal = p['beta']*np.exp(-p['beta']*(query_time-history_t))
        coefficients = np.asarray(p['alpha'])[:, history_k].T
        values += (spatial*temporal[None, :]) @ coefficients
    return values
