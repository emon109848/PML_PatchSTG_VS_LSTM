# Implementation: train_lstm_diagnostic.py

Source: [train_lstm_diagnostic.py](../../temp/legacy_scripts/train_lstm_diagnostic.py).

This adds actual Adam updates to the small LSTM check. It starts fresh and runs
exactly 20 successful CPU updates at learning rate 0.001 and batch size four.
Seed 2023 and a separate shuffle generator define training-window order.

## Flow

Load training data and scaler, build TrafficLSTM/Adam, and record initial weights.
Measure a fixed first-four-window training probe before updates. For each shuffled
batch: skip an empty mask; zero gradients; forward; masked MAE; backward;
check gradients; Adam step; check finite weights; print the pre-update batch loss.
Stop after update 20, assert weights changed, and score the same fixed probe.

Observed fixed-probe MAE decreased from 198.1274 to 187.3881. All six trainable
parameter tensors changed and no batches were skipped. Pooled diagnostic
training MAE was 145.1866 across evolving weights and different examples.

```bash
python train_lstm_diagnostic.py
```

This is a repeated fresh diagnostic, not a continuation command. No weights are
saved, no validation/test datasets are scored, and lower probe error does not
establish generalization. Mean/std are fixed rather than learned by Adam.

See [gradient/optimizer concepts](../concepts/12_training_gradients_and_adam.md).
