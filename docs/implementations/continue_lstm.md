# Implementation: continue_lstm.py

Source: [continue_lstm.py](../../src/traffic_prediction/continue_lstm.py).

This continues the original small LSTM/Adam run to a chosen total epoch count.
It is not the contextual suite trainer. Original protocol remains historical
provenance; the actual continuation execution device is recorded separately.

## Functions and lifecycle

- `restore_training`: reconstruct model, restore weights, place on selected device,
  restore Adam state and CPU/shuffle/CUDA RNG state after initialization.
- `device_batches` / `validate_device`: move x/y/mask consistently for scoring.
- `metrics_match`: coverage exact, floating metrics within rtol 1e-5/atol 1e-4
  to handle CPU/CUDA differences.
- `train_epoch`: all training windows, empty-mask skips, checks, original-unit loss.
- `save_checkpoint` / `save_report`: temporary writes followed by replacement.
- `main`: verify source/data/splits/scaler, validate loaded checkpoint, then
  continue and save best/last/history reports.

The check-only mode performs full frozen validation AND one disposable Adam
update; it does not save weights. After validation, model.train() must precede
the probe forward: cuDNN RNN backward needs training-mode forward intermediates.
This fixed an observed host error.

## Existing completed run

Epochs 2–50 ran on the user's RTX 5060 after CPU epoch 1. Best is epoch 49,
validation MAE 41.1316. The interrupted first CPU attempt at epoch 2 was discarded.
The GPU continuation completed 257,299 new updates, with exact best-reload scoring
recorded. This older run did not explicitly enforce the contextual suite's IEEE
RNN settings; do not retrospectively label all experiments identical precision.

Prior best.pt must accompany last.pt for this older continuation format. The
new contextual suite instead embeds its best/history in last for easier recovery.
See [checkpoints](../concepts/13_checkpoints_and_resume.md).
