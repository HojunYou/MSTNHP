"""Run: python -m unittest discover -s train_mstnhp/evaluation -p test_evaluation.py"""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from easy_tpp.model.torch_model.torch_stnhp import STNHP
from evaluation.intensity_metrics import IntensityErrors, spatial_grid, time_quadrature, biv1_reference
from evaluation.model_intensity import load_model, event_states, fitted_intensity, distances_to_history, sequence_arrays


def new_model(hidden, components):
    options = SimpleNamespace(hidden_size=hidden, num_event_types=2,
        num_event_types_pad=3, pad_token_id=2, n_comps=components,
        loss_integral_num_sample_per_step=10, spatial_npoints=4,
        model_specs={}, thinning=None, gpu=-1, polygon_path=None)
    return STNHP(options).double().eval()


class EvaluationTests(unittest.TestCase):
    def test_constant_intensity_and_relative_error(self):
        grid, weights = spatial_grid(4)
        self.assertAlmostEqual(weights.sum(), 4)
        errors = IntensityErrors(2, weights)
        truth = np.ones((len(grid), 2))*2
        errors.update(truth+0.5, truth, 3)
        result = errors.results()
        np.testing.assert_allclose(result['temporal_rmse'], 2)
        np.testing.assert_allclose(result['spacetime_rmse'], .5)
        np.testing.assert_allclose(result['temporal_relative_l2'], .25)
        np.testing.assert_allclose(result['spacetime_relative_l2'], .25)

    def test_identical_and_zero_reference(self):
        _, w = spatial_grid(3)
        errors = IntensityErrors(2, w)
        values = np.ones((len(w), 2))
        errors.update(values, values, 2)
        for result in errors.results().values():np.testing.assert_allclose(result, 0)
        errors = IntensityErrors(2, w)
        errors.update(values*0, values*0, 2)
        self.assertTrue(np.isnan(errors.results()['temporal_relative_l2']).all())

    def test_time_weights_and_tail(self):
        intervals = [(.3, 1.2), (1.2, 2.9), (2.9, 100)]
        points = []
        for left, right in intervals:
            q = list(time_quadrature(left, right, .7, 2))
            self.assertTrue(all(left < t < right for t, _ in q))
            self.assertAlmostEqual(sum(w for _, w in q), right-left)
            points.extend(q)
        self.assertAlmostEqual(sum(w for _, w in points), 99.7)
        self.assertAlmostEqual(sum(t*w for t, w in points), (100**2-.3**2)/2)

    def test_reference_excludes_future_and_event_itself(self):
        t = np.array([1., 3.]); s = np.array([[0., 0.], [.4, .2]]); k = np.array([0, 1])
        query = np.array([[0., 0.]])
        np.testing.assert_allclose(biv1_reference(1., query, t, s, k), .1)
        expected = np.array([[.1, .1]]) + np.array([[.25, .1]])*.3*np.exp(-.3)/np.pi
        np.testing.assert_allclose(biv1_reference(2., query, t, s, k), expected)

    def test_forward_equivalence_and_dynamic_dimensions(self):
        torch.set_num_threads(1)
        times = np.array([.4, 1.1, 1.8, 3.2])
        places = np.array([[0.,0.],[.3,.2],[-.2,.6],[.7,-.1]])
        marks = np.array([0, 1, 0, 1])
        for hidden in (32, 64):
            for components in (1, 3):
                with self.subTest(hidden=hidden, components=components):
                    model = new_model(hidden, components)
                    time_tensor = torch.tensor(times)[None, :]
                    place_tensor = torch.tensor(places)[None, :, :]
                    dt = torch.tensor(np.r_[times[0], np.diff(times)])[None, :]
                    ds = torch.zeros_like(place_tensor)
                    batch = (time_tensor, dt, place_tensor, ds, torch.tensor(marks)[None, :], None, None, None)
                    with torch.no_grad():
                        h, decay, outputs = model.forward(batch)
                        states = event_states(model, times, places, marks)
                        for i in range(len(times)-1):
                            c, cb, d, o = states[i]
                            torch.testing.assert_close(d[0], decay[0,i])
                            torch.testing.assert_close(torch.stack((c[0],cb[0],o[0])), outputs[0,i])
                            distances = distances_to_history(torch.tensor(places[i+1:i+2]), torch.tensor(places[:i+1]), components)
                            val = fitted_intensity(model, states[i], times[i+1]-times[i], distances)
                            np.testing.assert_allclose(val, model.layer_intensity(h[:,i]).numpy(), rtol=1e-10, atol=1e-12)
                        tail_distances = distances_to_history(torch.tensor([[0.,0.]]), torch.tensor(places), components)
                        self.assertTrue(np.isfinite(fitted_intensity(model, states[-1], 2., tail_distances)).all())
                    with tempfile.TemporaryDirectory() as folder:
                        checkpoint = Path(folder)/'model.pt'
                        torch.save(model.state_dict(), checkpoint)
                        loaded, dims = load_model(checkpoint, {}, {'num_event_types':2,'pad_token_id':2})
                        self.assertEqual(dims['hidden_size'], hidden)
                        self.assertEqual(dims['n_comps'], components)
                        with self.assertRaises(ValueError):
                            load_model(checkpoint, {'hidden_size':hidden+1}, {'num_event_types':2,'pad_token_id':2})

    def test_distances_order_and_padding(self):
        query = torch.tensor([[0., 0.]])
        history = torch.tensor([[1.,0.],[0.,2.]])
        torch.testing.assert_close(distances_to_history(query,history,3),torch.tensor([[0.,1.,2.]]))
        torch.testing.assert_close(distances_to_history(query,history,1),torch.tensor([[2.]]))

    def test_reject_ties(self):
        event = dict(time_since_start=1., location_from_origin=[0.,0.],type_event=0,time_since_last_event=0.)
        with self.assertRaises(ValueError):sequence_arrays([event,event],2,100.)


if __name__ == '__main__':
    unittest.main()
