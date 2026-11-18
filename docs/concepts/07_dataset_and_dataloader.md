# Dataset and DataLoader

A Dataset explains how to obtain one example. A DataLoader asks for examples,
converts them to tensors, and groups them into batches.

`TrafficDataset.__init__` stores array references, normalization, dimensions,
and eligible starts. `__len__` reports windows. `__getitem__` creates one
normalized `x`, original-unit `y`, boolean mask, and integer `target_start`.

```python
loader = DataLoader(dataset, batch_size=4, shuffle=True,
                    generator=generator, num_workers=0)
```

Training is shuffled; validation is sequential. The seeded generator makes
training order reproducible and its state is saved for resume. `num_workers=0`
uses the current Python process, not background loading workers.

```python
batch = next(iter(loader))
```

`iter` creates a batch iterator; `next` gets one batch. For full batches:

| Entry | Shape | Type |
| --- | --- | --- |
| x | `(4,12,716,1)` | float32 |
| y | `(4,12,716,1)` | float32 |
| target_mask | `(4,12,716,1)` | bool |
| target_start | `(4,)` | int64 |

The Dataset does not currently return calendar features. Contextual model
runners reconstruct historical calendar from `target_start`. There are 5,251
training batches; the last has one example because `drop_last` remains False.
Four windows contain 2,864 per-sensor sequences for a shared sensor-wise LSTM.

Iterator creation can consume random-generator state. Correct resume therefore
restores RNG state after model initialization and preserves it around probes.

See [checkpoint/RNG concepts](13_checkpoints_and_resume.md).
