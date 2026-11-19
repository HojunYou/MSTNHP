from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from intensity_metrics import IntensityErrors


class ReplicateErrorsTest(unittest.TestCase):
    def test_spatial_redistribution_and_constant_error(self):
        scores = IntensityErrors(2, [1., 1.])
        # First type doubles throughout; second moves all mass between cells.
        scores.update([[2., 0.], [2., 2.]], [[1., 2.], [1., 0.]], 3.)
        m = scores.results()
        expected = {
            'temporal_mae':[2., 0.], 'spacetime_mae':[1., 2.],
            'temporal_mse':[4., 0.], 'spacetime_mse':[1., 4.],
            'temporal_rmse':[2., 0.], 'spacetime_rmse':[1., 2.],
            'temporal_bias':[2., 0.], 'spacetime_bias':[1., 0.],
            'temporal_iae':[6., 0.], 'spacetime_iae':[6., 12.],
            'temporal_ise':[12., 0.], 'spacetime_ise':[6., 24.],
            'temporal_relative_l1':[1., 0.], 'spacetime_relative_l1':[1., 2.],
            'temporal_relative_l2':[1., 0.], 'spacetime_relative_l2':[1., np.sqrt(2.)],
            'temporal_max_abs':[2., 0.], 'spacetime_max_abs':[1., 2.],
            'spatial_tv':[0., 1.], 'spatial_hellinger':[0., 1.]}
        self.assertEqual(set(m), set(expected))
        for name, value in expected.items():
            with self.subTest(metric=name):
                np.testing.assert_allclose(m[name], value)
