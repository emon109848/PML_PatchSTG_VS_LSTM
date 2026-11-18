# Normalization and zero-target masks

These are separate rules: normalization controls input scale; the mask controls
which future observations contribute to errors.

## One global training scale

```text
z = (value - mean) / std
value = z * std + mean
```

The training mean is 247.19353447084683; population standard deviation is
184.27468320611501. One pair is fitted across training frames and sensors,
including zeros. Validation/test reuse it; no per-split refitting occurs.
The value 29 becomes approximately -1.1841. A negative normalized reading simply
lies below the mean; it does not mean negative traffic.

Inputs become float32 after normalization. Targets remain in original units,
also float32. Both LSTM classes inverse-normalize forecasts inside their forward
pass, so predictions and targets are compared on the same original-unit scale.
Inverse normalization remains differentiable.

## Mask policy

```python
target_mask = y != 0
```

True includes the target; False excludes it. Zero inputs are preserved and
normalized, not filled. There is no imputation or interpolation.
An all-masked training batch is skipped. An evaluation with no valid targets
raises an error rather than reporting perfect accuracy.

Zero exclusion follows the benchmark convention, not proven source validity.
Seasonal averages exclude zero training readings, whereas the scaler includes
them. These different rules are intentional. Scores can exclude whole sensors
when their targets are all zero; report coverage with accuracy.

See [metric implementation](../implementations/evaluation.md).
