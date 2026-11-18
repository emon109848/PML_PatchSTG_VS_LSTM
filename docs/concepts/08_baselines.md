# Persistence and seasonal-average baselines

A baseline is a simple reference predictor. A neural model is useful only in
relation to meaningful alternatives, not merely because it has learned weights.

## Persistence

Repeat each sensor's latest observed value for all 12 future steps. It learns
nothing. If the last reading is 106, all 12 forecasts are 106. Our implementation
extracts the last normalized input step, restores original units, and repeats it.

This can work for slowly changing near-term traffic but cannot anticipate a
change from the last level.

## Seasonal average

Learn average nonzero training traffic for each weekday, 15-minute slot, and
sensor. There are `7*96=672` calendar groups; table shape is `(672,716,1)`.
Future calendar groups are known without observing future traffic.

If a group's sensor mean has no observations, use that sensor's overall nonzero
training mean. If the sensor is entirely zero during training, use the global
nonzero training mean. An entirely zero training array cannot fit this baseline.

The seasonal table is fitted once on training and is never updated on validation.
It has no neural layers or optimizer, but it does estimate statistics.

## Observed validation reference

| Metric | Persistence | Seasonal average |
| --- | ---: | ---: |
| MAE | 64.1336 | 19.7442 |
| RMSE | 99.3846 | 36.4470 |
| MAPE (%) | 41.5436 | 12.4536 |

These are validation results. Do not compare them directly with a test score.
They show that recurring temporal structure is a strong reference on this task;
they do not establish why a neural model succeeds or fails.

See [baseline code](../implementations/baselines.md).
