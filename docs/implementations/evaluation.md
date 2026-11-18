# Implementation: evaluation.py

Source: [evaluation.py](../../src/traffic_prediction/evaluation.py).

## masked_mae(prediction, target, mask)

Checks matching shapes, boolean masks, finite predictions/targets, and at least
one included target. It returns the mean absolute error after boolean indexing.
This scalar retains gradients when called during training; it is not detached.

## evaluate_persistence

Runs the persistence rule under `torch.no_grad()`. Across loader batches it
accumulates absolute errors, squared errors, relative errors, target count, and
window count. It calculates pooled metrics at the end and returns a metric
dictionary plus valid-count/window-count values. An empty evaluation fails.

## evaluate_seasonal

For each example, adds `[0,...,11]` to target_start, maps those future indices
to calendar keys, and indexes the already fitted seasonal table. It accumulates
NumPy errors/counts with the same mask/metric definitions. It does not refit the
table using validation observations.

These functions are baseline-specific CPU evaluators, not one universal neural
API. Small LSTM validation lives in train_lstm.py; contextual train/validation
lives in train_context_lstm.py. They share the same original-unit metric rules.

## Important distinction

Loss averages one batch for gradients. Reporting must pool all included targets,
not average batch means without weights. The last partial batch and different
zero coverage make that distinction observable. Mask before percentage division.

See [formulas and aggregation](../concepts/09_metrics_and_aggregation.md).
