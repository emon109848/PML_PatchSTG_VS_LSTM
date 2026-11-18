# Implementation: comparison_dashboard.py

Source: [comparison_dashboard.py](../../src/traffic_prediction/comparison_dashboard.py).
Command: [export_comparison_dashboard.py](../../scripts/export_comparison_dashboard.py).

This combines the existing PatchSTG forecast cache with frozen LSTM predictions
at the same origins. It does not import/construct PatchSTG or run any training.

## Functions

- `load_cached_payload`: verify saved HTML against its manifest, extract the inert
  JSON payload, and check protocol/checkpoint identity.
- `canonical_groups`: map selected indices to their original full-test batches,
  including final partial batches, rather than changing neighbor grouping.
- `predict_origins`: process those groups in eval/no-grad, derive only historical
  calendar, and retain requested origin predictions.
- `comparison_payload`: copy cache metadata/observations/frames/sensors, preserve
  PatchSTG predictions exactly, and add rounded LSTM predictions/model metrics.
- `main`: verify completed LSTM selection, source/data/cache/evaluation hashes,
  training scale, sensor order and observation alignment; generate forecasts;
  assert frozen state; render self-contained HTML and manifest.

The shared integrated `render_replay` safely embeds JSON by escaping HTML-significant
characters, allowing the dashboard to open as a local file without fetching data.
The template uses canvas for charts/maps and ordinary controls for keyboard access.

The manifest distinguishes recorded full-test benchmark metrics from illustrative
window-local errors. Exporting 96 origins is not another full-test evaluation.
LSTM predictions are original units before rounding; targets are never rescaled.

The original PatchSTG replay and all protected checkpoints/reports remain unchanged.
See [dashboard controls, provenance and verification](../comparison_dashboard.md).
