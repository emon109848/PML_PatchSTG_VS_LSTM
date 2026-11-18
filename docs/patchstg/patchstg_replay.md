# Saved PatchSTG forecast replay

The integrated [dashboard](../../runs/patchstg_replay_v1/index.html) is an existing
standalone export of frozen PatchSTG forecasts. Open it in a browser; no server,
training, or inference is needed to view the saved predictions.

It shows 716 sensors, 96 chronological test origins, 12 forecast leads, sensor
selection, a coordinate view, history/forecast/revealed-future charts and error
summaries. Export selection is chronological rather than selected by low error.
This is a replay of recorded traffic, not live data or a road simulator.

The accompanying manifest records identity and provenance. Display values are
rounded; benchmark test metrics retain their original precision.

Included source:

- [Replay payload utilities](../../src/traffic_benchmark/evaluation/replay.py).
- [HTML template](../../demos/patchstg_replay.html).
- [Exporter](../../scripts/export_patchstg_replay.py).
- [Saved export record](../../reports/patchstg/patchstg_replay_v1.json).

The exporter is retained for completeness, but was not run for this integration.
The saved replay was copied unchanged. Do not imply it displays the newly trained
contextual LSTM: the dashboard currently contains PatchSTG forecasts only.
