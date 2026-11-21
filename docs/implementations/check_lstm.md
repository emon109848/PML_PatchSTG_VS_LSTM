# Implementation: check_lstm.py

Source: [check_lstm.py](../../reproducibility/original_sources/check_lstm.py).

This is a real-data CPU diagnostic, not training or evaluation of generalization.
It sets seed 2023 and two CPU threads, fits the training scale, and loads the
first four chronological training windows.

## Checks

- Forward output matches target shape `(4,12,716,1)` and is finite.
- Masked MAE backward reaches every trainable parameter tensor with finite,
  nonzero gradients.
- Parameter values remain exactly unchanged because no optimizer step is called.
- Fixed mean/std buffers do not receive gradients.
- Individual selected sensor calls match their batched positions within tolerance.
- Changing one sensor does not affect other sensors' forecasts.
- Repeated calls use fresh states, and a batch of one works.

Historical command from the original layout (not a current root entry point):

```text
python check_lstm.py
```

Observed initial first-batch MAE was 193.0234. Random initialized predictions
are not evidence of model quality. Another diagnostic can have a different
initial score even with the same seed when construction/iterator order differs,
because those operations consume RNG state.

No artifacts, checkpoints, or test scores are created. These checks support
correct computation, not faithfulness to every possible LSTM formulation.

See [the model](lstm_model.md) and [the update diagnostic](train_lstm_diagnostic.md).
