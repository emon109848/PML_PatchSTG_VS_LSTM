# PatchSTG model guide

PatchSTG is a main integrated model in this project. Its implementation, spatial
preprocessing, training/evaluation code, configs, tests, full saved run, selected
checkpoint and replay are included. The comparison reuses the completed 50-epoch
run and saved test measurements without retraining or reevaluation.

Start at [the PatchSTG documentation index](patchstg/README.md), then read
[architecture](patchstg/architecture.md), [spatial preparation](patchstg/spatial_preparation.md),
[implementation](patchstg/patchstg_implementation.md), [training](patchstg/patchstg_training.md),
and [evaluation](patchstg/patchstg_evaluation.md).

| Split | Best epoch | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| Validation | 50 | 14.9936 | 25.4044 | 9.0571 |
| Test | 50 | 17.1943 | 29.1370 | 11.1706 |

The shared-protocol adaptation has five width-192 attention blocks, 16×64 padded
sensor positions, and 2,284,844 parameters. It uses traffic, historical calendar,
sensor identity, coordinates and training-derived similarity. Width-256 contextual
LSTM has 848,644 parameters and no sensor-to-sensor traffic exchange.

Compare matching splits and disclose different optimizer/information/capacity
budgets. This is an adapted implementation, not an official author release;
[attribution](../THIRD_PARTY_NOTICES.md) and method deviations are retained.
