# MAE, RMSE, MAPE, and correct aggregation

Let `e = prediction - target` and let `K` be included forecast-target pairs:

```text
MAE  = sum(abs(e)) / K
RMSE = sqrt(sum(e²) / K)
MAPE = 100 * sum(abs(e) / abs(target)) / K
```

MAE measures average magnitude in traffic units. RMSE also uses traffic units
but emphasizes large errors. MAPE expresses relative error as a percentage.
For prediction 120 and target 100, absolute error is 20, squared error 400,
and percentage error 20%.

Zero targets are masked before division. There is no extra near-zero threshold
or epsilon in our implementation; small nonzero targets can inflate MAPE.

## Why not average batch scores?

A batch with 2 included targets and error sum 10 has MAE 5. Another with 6 targets
and sum 18 has MAE 3. Correct pooled MAE is `(10+18)/(2+6)=3.5`, not 4.
Accumulate sums/counts first, then calculate the metric. Likewise, overall RMSE
comes from pooled squared errors, not a mean of per-batch or per-lead RMSEs.

Our training score pools predictions produced by evolving weights. Validation
uses one frozen set of weights. They are not two scores for the identical model
state, even though they share formulas.

## Forecast leads and coverage

Lead 1 is assumed 15 minutes ahead; lead 12 is 180 minutes. Per-lead metrics
expose errors hidden by an overall score. The same physical reading can appear
as different forecast-target pairs because origins overlap.
Validation has 58,367,674 included pairs; test has 58,158,396 under this protocol.

See [metric code](../implementations/evaluation.md) and
[test scoring](../implementations/evaluate_lstm.md).
