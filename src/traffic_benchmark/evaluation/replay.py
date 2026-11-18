"""Small, explicit contracts for chronological historical forecast replay."""

import json
from datetime import datetime, timedelta

import numpy as np


def select_windows(dataset, start_window=0, frames=96, frame_stride=1):
    """Select by chronological index only, never forecast error or target values."""
    for name, value in (('start_window', start_window), ('frames', frames), ('frame_stride', frame_stride)):
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == 'start_window' else 1):
            raise ValueError(f'Invalid {name}.')
    indices = list(range(start_window, start_window + frames * frame_stride, frame_stride))
    if indices[-1] >= len(dataset):
        raise ValueError('Requested replay extends beyond eligible test windows.')
    return indices


def replay_payload(values, metadata, dataset, indices, predictions, provenance):
    """Store contiguous raw observations plus (frame,sensor,lead) predictions.

    Prediction input shape is (frames,horizon,sensors,1), in original units.
    Only exported predictions are rounded (3 decimals); benchmark reports stay exact.
    """
    if not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError('Replay indices must be nonempty and strictly chronological.')
    if indices[0] < 0 or indices[-1] >= len(dataset):
        raise ValueError('Replay index outside dataset.')
    starts = [dataset.target_starts[i] for i in indices]
    horizon, history, nodes = dataset.horizon, dataset.history, values.shape[1]
    predictions = np.asarray(predictions)
    if predictions.shape != (len(indices), horizon, nodes, 1) or not np.isfinite(predictions).all():
        raise ValueError('Expected finite (frames,horizon,sensors,1) predictions.')
    if len(metadata) != nodes:
        raise ValueError('Sensor metadata count mismatch.')
    base = starts[0] - history
    stop = starts[-1] + horizon
    observations = values[base:stop, :, 0]
    if not np.isfinite(observations).all() or (observations < 0).any():
        raise ValueError('Expected finite nonnegative observations.')
    def timestamp(index):
        stamp = dataset.assumed_start + timedelta(minutes=index * 15)
        return stamp.isoformat(timespec='minutes')
    return {
        'schema': 1, 'title': 'PatchSTG historical forecast replay',
        'history': history, 'horizon': horizon, 'interval_minutes': 15,
        'calendar_note': 'Assumed 2019 calendar, 15-minute cadence, no timezone; not observed timestamps.',
        'units': 'source flow units', 'prediction_decimal_places': 3,
        'zero_policy': 'Zero targets excluded from MAE; zero observations remain visible as reported.',
        'selection': 'Chronological test-window indices; no filtering by forecast quality.',
        'provenance': provenance,
        'sensors': [{'id': row['ID'], 'latitude': float(row['Lat']), 'longitude': float(row['Lng']),
                     'road': row.get('Fwy', ''), 'direction': row.get('Direction', '')} for row in metadata],
        'frames': [{'window_index': index, 'target_start': start, 'observation_offset': start - base,
                    'issued_at': timestamp(start - 1), 'first_target_at': timestamp(start)}
                   for index, start in zip(indices, starts)],
        'observations': observations.tolist(),
        'predictions': np.round(predictions[..., 0].transpose(0, 2, 1).astype(np.float64), 3).tolist(),
    }


def render_replay(template, payload):
    """Embed JSON safely in an inert script block; no network requests required."""
    marker = '__REPLAY_DATA__'
    if template.count(marker) != 1:
        raise ValueError('Replay template must have exactly one data marker.')
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    encoded = encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    return template.replace(marker, encoded)
