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

## Finishing command

Stop any existing contextual suite process first, in the terminal that can see it:

```bash
pkill -TERM -f '^python -u scripts/train_context_lstm.py'
```

Then run the frozen evaluation/report builder on the user's GPU:

```bash
python -u scripts/finalize_context_lstm.py --device cuda --output-dir runs/final_comparison_v1
```

This verifies full validation, evaluates the selected test checkpoint without
updates, checks frozen-state/coverage integrity, and generates overall,
per-horizon and per-sensor LSTM metrics plus comparison figures/report.
It reads historical baseline/PatchSTG reports and verifies matching data/protocol/
coverage; it also verifies the integrated PatchSTG checkpoint's recorded hash.
It does not retrain or freshly evaluate PatchSTG.

Output: `runs/final_comparison_v1/PROJECT_REPORT.md`, `report.json`,
`selection.json`, `comparison.json`, `learning_curves.png`, `test_horizons.png`,
and `sensor_errors.png`. Existing output directories are refused.

Synthetic checks passed for pooled/per-lead/per-sensor scoring, zero masks,
empty-sensor metrics and partial batches. The report/figure builder passed a
temporary fixture check, including the old PatchSTG checkpoint identity. Those
fixtures were discarded and were not published as new LSTM measurements.

CUDA is inaccessible to the agent. A real CPU forward timing was about 0.4 s per
batch, suggesting a much longer CPU validation/test pass. The GPU completion
command is prepared for the user's terminal. Final contextual test results and
generated comparison artifacts remain pending that execution.

## Completed host execution

Frozen width 256, epoch 37 evaluation completed successfully on CUDA. Full
validation reproduced before test scoring. All 6,997 test windows and 58,158,396
included targets were scored; model tensors and checkpoint bytes stayed unchanged.
Test MAE **18.1696**, RMSE **31.9485**, MAPE **12.0646%**.

PatchSTG test MAE is 17.1943, so LSTM MAE is 5.67% higher. Both neural
models outperform the recorded seasonal-average and persistence references.
These are single-seed system comparisons with documented input/optimizer/capacity
differences, not isolated causal evidence or statistical significance.

[Final project report](../runs/final_comparison_v1/PROJECT_REPORT.md),
[measured result](../runs/final_comparison_v1/report.json), and
[comparison evidence](../runs/final_comparison_v1/comparison.json) are available.
No further model tuning should be based on these observed test scores.
