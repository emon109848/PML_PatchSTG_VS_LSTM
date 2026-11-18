# Parameters, hidden width, and capacity

A trainable parameter is a learned scalar weight or bias. Input observations,
hidden activations, Adam moments, and fixed mean/std buffers are not counted as
trainable parameters. Sensor count increases computation; shared recurrent
weights are not replicated into 716 separate models.

For one PyTorch LSTM layer with input width d and hidden width h:

```text
parameters = 4h(d + h + 2)
```

There are four gate/candidate calculations, each using input/recurrent weights
and two biases. A linear head from h to 12 adds `12h+12`.

Small model: `4*32*(1+32+2) + 32*12+12 = 4,876`.

Contextual model adds 13,048 embedding parameters and two recurrent layers:

```text
13,048 + 4h(41+h+2) + 4h(h+h+2) + 12h+12
= 12h² + 192h + 13,060
```

| Hidden width | Parameters |
| --- | ---: |
| 128 | 234,244 |
| 256 | 848,644 |
| 384 | 1,856,260 |

Earlier conversation estimates were approximate; these are exact code counts.
Increasing hidden width has a roughly quadratic parameter cost and also increases
activation memory and per-update work. It may improve representation, overfit,
or simply require different optimization. Size alone is not accuracy.

PatchSTG has 2,284,844 parameters. Similar counts do not equate spatial
information, architectures, or optimization budgets. Report all those differences.

See [comparison fairness](17_comparison_and_fairness.md).
