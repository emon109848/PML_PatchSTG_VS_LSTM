"""Replay selection, exact history/target alignment and safe standalone export."""

import json
from pathlib import Path
import unittest

import numpy as np

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows
from traffic_benchmark.evaluation.replay import select_windows, replay_payload, render_replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(240, dtype=float).reshape(120, 2, 1)
        self.dataset = TrafficWindows(self.values, (60, 120), GlobalStandardScaler(1., 2.))
        self.metadata = [{'ID': str(i), 'Lat': '32.5', 'Lng': '-117.0', 'Fwy': 'I5'} for i in range(2)]

    def test_selection_is_chronological_bounded_and_includes_requested_last(self):
        self.assertEqual(select_windows(self.dataset, 1, 3, 2), [1, 3, 5])
        for args in ((-1, 2, 1), (0, 0, 1), (0, 1, 0), (0, 50, 1), (True, 1, 1)):
            with self.assertRaises(ValueError):
                select_windows(self.dataset, *args)

    def test_history_forecast_and_actual_align_for_first_last_and_strided_origins(self):
        indices = select_windows(self.dataset, 0, 3, 5)
        predictions = np.stack([self.dataset[i]['y'] for i in indices])
        payload = replay_payload(self.values, self.metadata, self.dataset, indices, predictions, {})
        for frame, index in enumerate(indices):
            offset = payload['frames'][frame]['observation_offset']
            obs = np.asarray(payload['observations'])
            sample = self.dataset[index]
            np.testing.assert_array_equal(obs[offset-12:offset], self.values[sample['target_start']-12:sample['target_start'], :, 0])
            np.testing.assert_array_equal(obs[offset:offset+12], sample['y'][..., 0])
            np.testing.assert_array_equal(np.asarray(payload['predictions'][frame]).T, sample['y'][..., 0])
        self.assertEqual(payload['frames'][0]['issued_at'], '2019-01-01T14:45')
        self.assertEqual(payload['frames'][0]['first_target_at'], '2019-01-01T15:00')

    def test_preserve_zero_observations_negative_predictions_and_round_only_display(self):
        self.values[60, 0, 0] = 0
        forecasts = np.full((1, 12, 2, 1), -1.123456, dtype=np.float32)
        before = forecasts.copy()
        data = replay_payload(self.values, self.metadata, self.dataset, [0], forecasts, {})
        self.assertEqual(data['observations'][12][0], 0)
        self.assertEqual(data['predictions'][0][0][0], -1.123)
        np.testing.assert_array_equal(forecasts, before)

    def test_invalid_prediction_shapes_and_nonchronological_origins_fail(self):
        for indices, pred in (([1, 0], np.zeros((2, 12, 2, 1))), ([0], np.zeros((1, 2, 12, 1))),
                              ([0], np.full((1, 12, 2, 1), np.nan))):
            with self.assertRaises(ValueError):
                replay_payload(self.values, self.metadata, self.dataset, indices, pred, {})

    def test_inert_json_escaping_and_template_contract(self):
        payload = {'label': '</script><script>alert(1)</script>&'}
        html = render_replay('<script type="application/json">__REPLAY_DATA__</script>', payload)
        encoded = html.split('>', 1)[1].rsplit('</script>', 1)[0]
        self.assertNotIn('</script>', encoded)
        self.assertEqual(json.loads(encoded), payload)
        for template in ('missing', '__REPLAY_DATA____REPLAY_DATA__'):
            with self.assertRaises(ValueError):
                render_replay(template, {})
        template = Path('demos/patchstg_replay.html').read_text()
        self.assertNotIn('__REPLAY_DATA__', render_replay(template, {'schema': 1}))


if __name__ == '__main__':
    unittest.main()
