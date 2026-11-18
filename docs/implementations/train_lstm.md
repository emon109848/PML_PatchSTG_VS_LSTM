# Implementation: train_lstm.py

Source: [train_lstm.py](../../src/traffic_prediction/train_lstm.py).

This original script starts one fresh CPU epoch, not 50 epochs or resume. It
refuses an existing output directory. Training uses Adam 0.001, batch four,
seed 2023, two CPU threads, deterministic algorithms and shuffled windows.

## Functions

`validate` sets evaluation mode and no-grad, pools original-unit errors in
float64, and returns MAE/RMSE/MAPE, target count and windows. `sha256` fingerprints
bytes. `main` builds data/model/optimizer, records provenance, trains, validates,
saves, and checks reload.

## Saved first run

`runs/lstm_epoch1_v1/` contains config.json, best.pt, last.pt, report.json.
Every one of 21,001 training windows was processed in 5,251 updates. Validation
scored 6,997 windows and 58,367,674 included targets. Epoch 1 was both best and
last; validation MAE was 46.0824. Training plus first validation took about
296 seconds on CPU, excluding setup/saves/reload verification.

The script reloads best.pt into a new model and repeats complete validation,
requiring exact agreement. It records model/Adam/RNG state, source/data hashes,
software versions, settings and counts. No test dataset is constructed.

The original default output exists; rerunning it is unnecessary. For continuation,
use continue_lstm.py, not another fresh run. This file's source hash belongs to
existing checkpoints, so documentation work deliberately leaves it unchanged.

See [continuation](continue_lstm.md) and [original training guide](../../temp/notes/LSTM_TRAINING_GUIDE.md).
