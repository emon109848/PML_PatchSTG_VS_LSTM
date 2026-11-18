# Implementation: finalize_context_lstm.py

Source: [finalize_context_lstm.py](../../src/traffic_prediction/finalize_context_lstm.py).

This completes the reduced two-candidate project scope without training width
384. It preserves the original suite/checkpoints and writes a separate frozen
evaluation directory, not a misleading completed three-width suite summary.

## Functions

- `select_completed`: verify completed 128/256 training reports, checkpoint
  selection, metadata, source identities and coverage. Select lowest validation
  MAE and freeze a manifest. No test scores participate. Width 384 is deferred.
- `metric_values`: derive pooled MAE/RMSE/MAPE or null values for zero coverage.
- `score_frozen`: eval/no-grad contextual forecasts, with historical calendar,
  pooling overall, per-horizon and per-sensor errors without storing all forecasts.
- `check_reference`: require matching integrated PatchSTG data/scaler/windows and
  lead-wise target counts before comparison.
- `build_comparison`: verify old reference reports and PatchSTG checkpoint hash,
  generate learning curves, lead curves, sensor-error distribution, machine-readable
  comparison and an interpreted Markdown project report. Old reference values
  are explicitly historical; no new PatchSTG inference is claimed.
- `main`: verify paths/data/scaler/splits, reproduce selected validation, run test,
  independently count masks, assert frozen model/checkpoint integrity, and write.

## Execution

```bash
python -u scripts/finalize_context_lstm.py --device cuda --output-dir runs/final_comparison_v1
```

The directory must not exist. `--prepare-only` can freeze and check selection
without GPU access or any forward passes, using a separate output directory.
The agent executed that preparation in reports/context_lstm_final_selection_v1.

Validation tolerates small CPU/GPU metric differences (rtol 1e-5, atol 1e-4),
while window/mask counts match exactly. No optimizer or gradients are created.
Mean/std, weights and checkpoint bytes remain unchanged. Global/matmul/cuDNN
RNN precision uses the contextual runner's explicit IEEE settings.

## Interpretation limits

Only the two completed candidates participate. The scope revision occurred after
interim validation observations; it must be disclosed. Previously observed test
results and unequal capacity/input/optimizer/search budgets remain limitations.
Per-sensor analysis is LSTM-only because old reports lack paired sensor-level
forecasts. Do not infer a spatial causal explanation from those figures.

Actual test output is pending host execution; synthetic metric and temporary
report-generation checks have passed. See
[finalization status](../../reports/finalization_status.md).
