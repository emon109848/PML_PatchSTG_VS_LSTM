# Training: loss, gradients, and Adam

Training adjusts learned parameters using training examples. It does not adjust
the fixed global normalization statistics.

```python
model.train()
optimizer.zero_grad(set_to_none=True)
prediction = model(inputs)
loss = masked_mae(prediction, target, mask)
loss.backward()
optimizer.step()
```

`zero_grad` clears previous gradients because PyTorch otherwise accumulates them.
The forward pass calculates predictions. Masked MAE gives one differentiable
scalar. `backward` calculates gradients via the chain rule; it does not update
weights. Adam's `step` uses gradients and running first/second-moment estimates
to update parameters. Our learning rate is 0.001.

## What gets learned?

Small LSTM: recurrent/input weights, biases, and linear head. Contextual LSTM:
those parameters plus sensor and calendar embedding tables. The output's inverse
normalization is differentiable, so original-unit loss still reaches all layers.

All-masked batches are skipped; missing or nonfinite gradients and nonfinite
weights stop training. Loss may rise between batches because the batches differ.
A short diagnostic establishes usable gradients/updates, not generalization.

## Our fixed contextual protocol

Two layers; widths 128/256/384; no dropout; Adam with no weight decay, scheduler,
or clipping; batch size four; seed 2023; 50 epochs per candidate. Training shuffles
windows and keeps their internal time order. The original PatchSTG used different
optimizer settings, so equal epochs do not mean identical optimization.

See [diagnostic](../implementations/train_lstm_diagnostic.md) and
[candidate runner](../implementations/train_context_lstm.md).
