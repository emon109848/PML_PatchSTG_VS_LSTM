# Implementation: evaluate_lstm.py

Source: [evaluate_lstm.py](../../reproducibility/original_sources/evaluate_lstm.py).

This evaluates the original traffic-only LSTM checkpoint, not ContextLSTM.
Default input is runs/lstm_50epochs_gpu_v1/best.pt. Existing output directories
are preserved rather than overwritten.

## Preconditions

Confirm checkpoint epoch/validation match the completed training report. Verify
original source hashes, dataset hash, chronological boundaries and train-only
normalization. Reconstruct TrafficLSTM, load strictly, and recheck full validation
on the evaluation device before allowing test scoring.

## score_test

Sets eval/no-grad, predicts sequential test batches, and accumulates errors and
counts separately for each of 12 horizons in float64. Overall RMSE comes from
all squared errors, not averaging horizon RMSE. Independently count each lead's
source nonzero targets. Assert all model-state tensors and checkpoint bytes
remain unchanged, and that no parameter gradients were created.

## Executed result

CPU evaluation of epoch 49 covered 6,997 test windows and 58,158,396 valid pairs:
MAE 39.7079, RMSE 62.9506, MAPE 27.4996%. Lead MAE rose from 15.3701 at assumed
15 minutes to 62.9966 at 180 minutes. The test pass took about 10.09 seconds,
including loading/batching/metrics; it is not isolated GPU inference latency.

[The report](../../reports/lstm_test_v1.md) contains all lead results and limits.
Synthetic examples checked masked overall/per-lead metrics and partial batches.
No additional training or optimizer was used. These observed test scores must
not be used for checkpoint selection or represented as newly unseen afterward.

Contextual evaluation remains separate future work; checkpoint schemas differ.
