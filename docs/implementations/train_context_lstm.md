# Implementation: train_context_lstm.py

Source: [train_context_lstm.py](../../src/traffic_prediction/train_context_lstm.py).

This runs widths 128 → 256 → 384 sequentially for 50 epochs each. It constructs
training/validation datasets only. The original traffic-only model is not reused.

## Functions

`setup(device)` configures deterministic algorithms, two CPU threads and explicit
IEEE FP32 matmul/cuDNN RNN precision; unavailable requested CUDA fails.

`run_epoch(model,loader,optimizer=None)` shares the actual train/validation path.
An optimizer enables train mode and gradients; otherwise eval mode and disabled
gradients. It reconstructs historical calendar, computes original-unit masked
MAE, performs updates when training, and pools reporting metrics in float64.
Invalid gradients/weights or empty full scoring sets fail. It prints progress
at every 1,000 training updates.

`restore` reloads weights, Adam, shuffle and CPU/CUDA RNG states after model
construction. CUDA-loaded RNG tensors are explicitly moved to CPU where needed.

`train_candidate` handles one width, fresh or resumed. It records each epoch,
checks expected target/window coverage, chooses strict lowest validation MAE,
and saves authoritative last state before derived best/report files. Last embeds
best and history, so interrupted derived writes can be repaired. After epoch 50,
it reloads the saved best and requires exact full-validation reproduction.

`main` validates paths and suite identity, records hashes/settings/environment,
runs all candidates, writes summary after each completion, then selects lowest
validation MAE with smaller-width exact tie breaking. No early stopping or test
selection occurs.

## Saved layout

```text
runs/context_lstm_suite_v1/
  config.json
  summary.json
  hidden_128/{last.pt,best.pt,report.json}
  hidden_256/{last.pt,best.pt,report.json}
  hidden_384/{last.pt,best.pt,report.json}
```

`last.pt` includes optimizer/RNG and is for resume. `best.pt` stores selected
weights and metadata, not the complete optimizer state; it is for final selected
inference. Reports contain epoch metrics/times for later learning-curve plots.
A status 'running' in a report means not finalized; it does not prove the process
is alive. Check the terminal/log separately.

On resume, matching source/data/settings/device/software metadata is required.
Completed candidates are reverified; incomplete ones restart after the saved
boundary. An interrupted partial epoch is repeated. Failures stop the suite
rather than silently changing settings or skipping a candidate.

See [commands](../commands.md), [current snapshot](../results_and_status.md),
and [selection concepts](../concepts/14_validation_selection_and_overfitting.md).
