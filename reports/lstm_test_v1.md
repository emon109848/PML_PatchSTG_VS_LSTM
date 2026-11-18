# Frozen LSTM test evaluation

Evaluated November 17, 2026 after user approval. The checkpoint was selected
solely by validation MAE from 50 total training epochs; selected epoch is 49.
These test results are observed and must not be used to tune this model.

## Protocol and results

Loaded `runs/lstm_50epochs_gpu_v1/best.pt` using
[evaluate_lstm.py](../temp/legacy_scripts/evaluate_lstm.py). Evaluated on CPU because CUDA was not
available in the agent environment. Batch size remained 4, history/horizon 12,
and inputs used the saved global training normalization. The same frozen
checkpoint first reproduced validation on CPU within the documented CPU/CUDA
tolerance (`rtol=1e-5`, `atol=1e-4`). The original training used CPU epoch 1
and CUDA epochs 2–50.

| Measurement | Result |
| --- | ---: |
| Test target interval | `[28032, 35040)` |
| Test windows | 6,997 |
| Included forecast-target pairs | 58,158,396 |
| MAE | **39.7079** |
| RMSE | **62.9506** |
| MAPE (%) | **27.4996** |

Zero targets were excluded before scoring or division. Metrics pool errors
across windows, sensors and leads. Each lead's target count was independently
checked against nonzero source observations. Overlapping windows count distinct
forecast-target pairs, not only unique physical readings.

The test scoring pass took 10.09 seconds on CPU, including dataset construction,
batching, model execution, and metric accumulation. This is not isolated
inference latency or a controlled efficiency benchmark.

## Forecast distance

Minutes assume the benchmark's 15-minute sampling cadence.

| Lead | Minutes | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 15 | 15.3701 | 23.9490 | 9.8023 |
| 2 | 30 | 19.8898 | 31.1394 | 12.2146 |
| 3 | 45 | 24.1454 | 37.4956 | 14.8266 |
| 4 | 60 | 28.3938 | 43.6568 | 17.8913 |
| 5 | 75 | 33.4387 | 50.8887 | 21.2647 |
| 6 | 90 | 38.0517 | 57.4497 | 24.4131 |
| 7 | 105 | 42.3251 | 63.2513 | 28.1047 |
| 8 | 120 | 46.4641 | 68.8964 | 31.7400 |
| 9 | 135 | 50.9933 | 75.1207 | 35.7758 |
| 10 | 150 | 55.2414 | 81.0186 | 39.9054 |
| 11 | 165 | 59.1864 | 86.3697 | 44.4856 |
| 12 | 180 | 62.9966 | 91.6197 | 49.5728 |

Errors rise with forecast distance in this run. This is evidence that overall
scores conceal a substantial short-/long-lead difference. It does not alone
establish the reason: limited history, capacity, optimization, and traffic
variation are possible explanations requiring further controlled analysis.
The model is a shared per-sensor temporal LSTM; it has no explicit sensor
interaction or calendar features. Do not attribute differences from PatchSTG
to spatial modeling alone without controlling other input/training differences.

## Verification and artifacts

- Completed training report and checkpoint agree on selected epoch/validation.
- Original source hashes, dataset hash, split boundaries and train-only scaler match.
- Model evaluated in evaluation mode with no gradient tracking or optimizer.
- Every state-dictionary tensor remained exactly unchanged; no parameter gradients
  were created and the source checkpoint hash remained unchanged.
- Synthetic tests checked hand-calculated overall and per-lead metrics,
  zero masking, partial batches, and invariance to evaluation batch grouping.
- Full result, all horizon counts, checkpoint/source identity, versions, and
  validation recheck are in [the machine-readable report](../temp/experiments/lstm_test_v1/report.json).

One seed and one network are evaluated. Metadata alignment, calendar provenance,
zero validity meanings, and uneven nonzero sensor coverage retain the limits
described in [the data audit](data_audit_v1/README.md). No new tuning, extra
training, or PatchSTG evaluation was performed in this milestone.

The executed command was:

```bash
python evaluate_lstm.py
```

An existing output directory is preserved rather than overwritten. Rerunning
requires a new `--output-dir`; repeating evaluation is unnecessary unless there
is a specific verification reason.
