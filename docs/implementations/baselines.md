# Implementation: baselines.py

Source: [baselines.py](../../src/traffic_prediction/baselines.py).

## persistence_forecast

Input: normalized x `(B,12,N,1)`, mean/std, horizon. It selects `x[:,-1:,:,:]`,
converts to float64, reverses normalization, and repeats across the time axis.
Output is `(B,horizon,N,1)` in original traffic units. Reversing float32 input
normalization can produce tiny differences from the raw integer-valued readings.
There are no learned parameters or optimizer.

## fit_seasonal_average

Input: training traffic and aligned calendar keys. It counts nonzero observations,
calculates the global nonzero mean, then sensor nonzero means. The initial
`(672,N,1)` table repeats sensor means as fallback. For each weekday/slot key,
it replaces positions with group means where group counts are nonzero.

`np.divide(..., out=..., where=...)` avoids dividing by zero and retains fallback
values where no group observations exist. A sensor with no nonzero training
readings uses the global mean. Entirely zero training input raises an error.

The function returns a lookup table, not an object with a prediction method.
`evaluate_seasonal` indexes that table using the future forecast calendar keys.
The runner is responsible for training-only fitting and valid aligned inputs.

## Verification

Persistence is compared with last raw source readings. A small two-sensor example
checks weekday/slot averaging and both seasonal fallback levels. Full validation
counts match between the two baselines; persistence also has an independent
full raw-array metric calculation.

See [baseline concepts](../concepts/08_baselines.md) and
[evaluation implementation](evaluation.md).
