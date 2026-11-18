# Implementation: data_pipeline.py

Source: [data_pipeline.py](../../src/traffic_prediction/data_pipeline.py).

This module provides reusable preparation functions. Importing it does not load
the dataset or run evaluations.

## Functions in reading order

1. `load_traffic(path)` opens NPZ with `allow_pickle=False`, extracts `data`, and
   closes the archive. The returned array stays in memory. This minimal loader
   does not itself enforce dataset shape/value/metadata checks; the audit does.
2. `split_boundaries(total_steps)` calculates training end at 60% and test start
   before the final 20%. Validation receives the remainder.
3. `fit_normalization(training_values)` computes one float64 mean/population std,
   including zeros. It rejects invalid statistics or nonpositive std. The caller
   must pass training observations only.
4. `make_calendar_keys` maps assumed chronological indices to weekday/slot groups
   for the seasonal baseline. It returns keys, not source timestamps.
5. `make_window` selects past/future, normalizes past, casts copies to float32,
   and returns x, original-unit y, and `y != 0` mask. It checks window bounds.
6. `TrafficDataset` stores references/settings and a range of eligible starts.
   `__len__` counts starts; `__getitem__` calls make_window and adds target_start.

## Contract

One example has x/y/mask `(12,716,1)`; a DataLoader adds the batch dimension.
The calendar is not currently part of Dataset output. Contextual runners derive
it from target_start, avoiding a change to this original module during training.
All split datasets share the underlying traffic array and training scale.

## Verification and practical limits

The walkthrough checks first/last source correspondence, all split target bounds,
normalization reversal and independent window storage. The audit handles richer
source validation. This is intentionally small educational infrastructure,
not a generalized production loader for arbitrary datasets and cadences.

See [splits](../concepts/04_temporal_splits_and_leakage.md),
[windows](../concepts/05_sliding_windows.md), and
[normalization/masks](../concepts/06_normalization_and_masks.md).
