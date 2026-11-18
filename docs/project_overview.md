# Project goal and implemented architecture

This course project compares contextual LSTM and PatchSTG for processed SD
traffic forecasting, with persistence and seasonal-average baselines. LSTM
applies a method learned in class; PatchSTG provides a spatial-attention system.
BigST and the unfinished LSTM width-384 run are deferred.

## Shared task

The array contains 35,040 time steps, 716 sensors and one flow feature. Cadence
is assumed 15 minutes from January 1, 2019. Each example uses 12 observed steps
to forecast the next 12. Chronological splits, train-only normalization and
nonzero target masks are shared. Timestamp and sensor-column provenance limits
remain documented in the audit.

## Integrated models

- Contextual LSTM: two recurrent layers, traffic plus calendar/sensor embeddings,
  direct 12-step head, 848,644 parameters for selected width 256. It processes
  each sensor history independently. Best validation epoch is 37.
- PatchSTG: temporal projection, sensor/calendar embeddings, coordinate/similarity
  patch preparation, five depth/breadth attention blocks and forecast projection,
  2,284,844 parameters. Best validation epoch is 50.

LSTM source lives in src/traffic_prediction. PatchSTG and its recorded-run
infrastructure live in src/traffic_benchmark. Both include models, supporting
preparation, runnable tooling, checks and saved experiments in this project.

## Results and reuse

Both selected models have completed frozen test measurements. LSTM MAE is
18.1696; PatchSTG MAE is 17.1943. The comparison reuses PatchSTG's completed
50-epoch run, checkpoint and saved evaluation. Integration performed integrity
checks, not new PatchSTG training or inference.

[The final report](../reports/final_comparison/PROJECT_REPORT.md) contains matched
scores, learning curves, forecast-lead analysis, sensor variation and limitations.
The [saved PatchSTG replay](../runs/patchstg_replay_v1/index.html) displays cached
forecasts without running a model. It does not display contextual LSTM forecasts.

Formal uncertainty estimates, causal spatial ablations, new datasets and
controlled cross-model runtime benchmarks are outside the completed scope.
Read [the model guide](patchstg/README.md), [results](results_and_status.md), and
[fairness discussion](concepts/17_comparison_and_fairness.md).
