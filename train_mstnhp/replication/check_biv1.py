"""Compare the two thinning implementations with the same seeded proposals."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_mstpp_simulator.bivariate_simulation import build_kernel_set
from final_mstpp_simulator.biv1_thinning import Biv1Thinning
from final_mstpp_simulator.mstppg import BivariateHawkesLam, BivariateSpatialTemporalPointProcess


def simulate(cls, seed, horizon):
    lam = BivariateHawkesLam([0.1, 0.1], build_kernel_set("bivariate_1"), maximum=500)
    pp = cls(lam)
    np.random.seed(seed)
    start = perf_counter()
    candidates = pp._homogeneous_poisson_sampling([0, horizon], [[-1, 1], [-1, 1]])
    points, marks = pp._inhomogeneous_poisson_thinning(candidates, verbose=False)
    return points, marks, np.random.get_state(), perf_counter() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--horizon", type=float, default=100.)
    args = parser.parse_args()
    fast = simulate(Biv1Thinning, args.seed, args.horizon)
    legacy = simulate(BivariateSpatialTemporalPointProcess, args.seed, args.horizon)
    for left, right in zip(fast[:2], legacy[:2]):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(fast[2], legacy[2]):
        np.testing.assert_array_equal(left, right)
    print(f"{len(fast[0])} matching events and matching final RNG state. "
          f"Original {legacy[3]:.3f} s, optimized {fast[3]:.3f} s.")
