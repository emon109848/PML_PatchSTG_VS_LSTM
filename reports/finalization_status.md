# Reduced project scope and frozen LSTM selection

The user explicitly deferred the unfinished 384-unit candidate to finish the
project. Its checkpoints and partial results are preserved. It is not a completed
candidate, and the original three-width suite is not relabeled complete.

Selection is now among the two completed 50-epoch candidates, by lowest best
full-validation MAE:

| Width | Parameters | Best epoch | Validation MAE |
| --- | ---: | ---: | ---: |
| 128 | 234,244 | 39 | 15.3385 |
| **256** | **848,644** | **37** | **15.0656** |

Frozen manifest: [selection.json](context_lstm_final_selection_v1/selection.json).
Both report/checkpoint pairs, all 50 epoch records, source hashes, and dataset
identity were checked. No contextual test prediction was calculated during
selection/preparation. The decision followed observation of interim validation
results, including partial width-384 results; disclose this scope revision.

## Current action

No evaluation needs to be repeated. Refresh the written report from tracked saved
measurements with `python scripts/build_project_report.py`. Refresh slides with
`python scripts/build_presentation.py`. Neither executes a forecasting model.
The frozen evaluator remains available for deliberate reproduction into a new
output directory; see [commands](../docs/commands.md).

## Completed host execution

Frozen width 256, epoch 37 evaluation completed successfully on CUDA. Full
validation reproduced before test scoring. All 6,997 test windows and 58,158,396
included targets were scored; model tensors and checkpoint bytes stayed unchanged.
Test MAE **18.1696**, RMSE **31.9485**, MAPE **12.0646%**.

PatchSTG test MAE is 17.1943, so LSTM MAE is 5.67% higher. Both neural
models outperform the recorded seasonal-average and persistence references.
These are single-seed system comparisons with documented input/optimizer/capacity
differences, not isolated causal evidence or statistical significance.

[Final project report](final_comparison/PROJECT_REPORT.md),
[measured result](final_comparison/report.json), and
[comparison evidence](final_comparison/comparison.json) are available.
No further model tuning should be based on these observed test scores.
