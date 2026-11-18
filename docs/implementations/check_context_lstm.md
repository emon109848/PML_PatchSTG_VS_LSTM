# Implementation: check_context_lstm.py

Source: [check_context_lstm.py](../../src/traffic_prediction/check_context_lstm.py).

This probes all contextual widths sequentially on the first real four-window
training batch. Each starts fresh with seed 2023 and Adam 0.001. One warm-up
update is excluded from timing; by default five more updates are measured.
Weights are discarded at the end; no checkpoint/validation/test score is created.

## What is checked

- Correct prediction shape and finite masked loss, gradients and weights.
- A recurrent forward pre-hook captures actual LSTM input sequences.
- Selected traffic, calendar embedding and sensor embedding positions match
  intended sources with zero tolerance, detecting axis/identity mistakes.
- Isolated-sensor predictions match batched positions within stated tolerance.
- Changing one input sensor leaves all other sensor predictions unchanged.

## GPU behavior and output

Synchronize CUDA around updates, record mean timed-step duration and peak allocated/
reserved memory. Explicit IEEE FP32 matmul and cuDNN RNN settings resolved the
previous host parity failure in the rerun without loosening output tolerance.
Actual cause was not conclusively isolated.

The completed host report is
[context_lstm_gpu_ieee_probe.json](../../reports/context_lstm_gpu_ieee_probe.json).
Widths 128/256/384 averaged 34.5/43.1/95.8 ms per diagnostic step, with peak
allocated memory approximately 734/843/1,221 MiB.

Output JSON is written only if --output is supplied and all probes finish.
The early failed attempt therefore did not produce a completed all-width report.
Repeated warmed-batch timing is not a full training-epoch estimate.

See [precision/timing](../concepts/15_gpu_precision_and_timing.md).
