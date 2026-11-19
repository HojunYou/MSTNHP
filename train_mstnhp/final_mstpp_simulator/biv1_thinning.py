"""Batched evaluation of the legacy Biv1 thinning loop."""
import numpy as np

from .mstppg import BivariateSpatialTemporalPointProcess
from .stppg import SeparableExpTimeGaussianSpaceKernel


class Biv1Thinning(BivariateSpatialTemporalPointProcess):
    def __init__(self, lam):
        super().__init__(lam)
        kernels = [kernel for row in lam.kernel for kernel in row]
        if not all(isinstance(k, SeparableExpTimeGaussianSpaceKernel) for k in kernels):
            raise ValueError("Expected exponential time and Gaussian space kernels.")
        self.beta = kernels[0].beta_t
        self.variance = kernels[0].sigma2
        if any(k.beta_t != self.beta or k.sigma2 != self.variance or k.alpha < 0
               for k in kernels):
            raise ValueError("This path requires shared decay, shared variance and positive kernels.")
        self.alpha = np.array([[k.alpha for k in row] for row in lam.kernel])

    def _rates(self, candidates, history, marks):
        dt = candidates[:, 0, None] - history[:, 0]
        dx = candidates[:, 1, None] - history[:, 1]
        dy = candidates[:, 2, None] - history[:, 2]
        temporal = self.beta * np.exp(-self.beta * dt) * (dt > 0)
        spatial = (1 / (2 * np.pi * self.variance)
                   * np.exp(-(dx * dx + dy * dy) / (2 * self.variance)))
        return np.column_stack([
            self.lam.mu[k] + ((self.alpha[k, marks] * temporal) * spatial).sum(axis=1)
            for k in (0, 1)
        ])

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose=False):
        # Retain the legacy candidate stream and RNG consumption, including the
        # special randint draw for equal mark rates. Recompute after each event.
        bound = self.lam.upper_bound()
        accepted = []
        marks = []
        pos = 0
        while pos < len(homo_points):
            if not accepted:
                r0, r1 = self.lam.mu
                mark = (np.random.randint(2) if r0 == r1 else
                        np.random.choice(2, p=np.array([r0, r1]) / (r0 + r1)))
                draw = np.random.uniform()
                if r0 + r1 > bound:
                    return None, None
                if r0 + r1 >= draw * bound:
                    accepted.append(pos)
                    marks.append(mark)
                pos += 1
                continue

            candidates = homo_points[pos:pos + 512]
            history = homo_points[accepted]
            rates = self._rates(candidates, history, np.asarray(marks))
            # In near-ties, use the scalar legacy routine to determine exactly
            # which RNG call the legacy version would make.
            near = np.flatnonzero(np.abs(rates[:, 0] - rates[:, 1]) <=
                                  32 * np.finfo(float).eps * rates.max(axis=1))
            for j in near:
                t, x, y = candidates[j]
                rates[j] = [self.lam.value(t, history[:, 0], np.array([x, y]),
                                           history[:, 1:], k, marks) for k in (0, 1)]
            equal = rates[:, 0] == rates[:, 1]
            if equal[0]:
                mark = np.random.randint(2)
                draw = np.random.uniform()
                total = rates[0].sum()
                if total > bound:
                    return None, None
                if total >= draw * bound:
                    accepted.append(pos)
                    marks.append(mark)
                pos += 1
                continue

            # For unequal rates, the legacy choice and acceptance draw each
            # consume one uniform. Speculate only up to the next equal-rate row.
            equal_rows = np.flatnonzero(equal)
            count = int(equal_rows[0]) if equal_rows.size else len(rates)
            totals = rates[:count].sum(axis=1)
            initial_rng = np.random.get_state()
            draws = np.random.random_sample((count, 2))
            stops = np.flatnonzero((totals >= draws[:, 1] * bound) | (totals > bound))
            if not stops.size:
                pos += count
                continue
            j = int(stops[0])
            # Discard unconsumed look-ahead draws.
            if j + 1 < count:
                np.random.set_state(initial_rng)
                np.random.random_sample((j + 1, 2))
            if totals[j] > bound:
                return None, None
            cdf = np.cumsum(rates[j] / totals[j])
            mark = int(draws[j, 0] >= cdf[0] / cdf[-1])
            accepted.append(pos + j)
            marks.append(mark)
            pos += j + 1
        return homo_points[np.asarray(accepted, dtype=int)], np.asarray(marks, dtype=np.int32)
