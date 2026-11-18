# Arrays, tensors, indexing, and shapes

NumPy arrays store the dataset. PyTorch tensors feed the model and support
automatic differentiation and GPU execution.

| Object | Shape | Axes |
| --- | --- | --- |
| Source | `(35040,716,1)` | time, sensor, feature |
| One history or target | `(12,716,1)` | window time, sensor, feature |
| Batch | `(4,12,716,1)` | example, window time, sensor, feature |
| LSTM sequences | `(4*716,12,F)` | sensor sequence, time, input features |

`F=1` for the original LSTM; `F=41` for the contextual version.

## Python indexing

```python
traffic[0, 0, 0]     # one scalar: first reading, first sensor
traffic[:12, 0, 0]   # steps 0 through 11 for sensor 0
traffic[0, :, 0]     # all sensors at step 0
x[:, -1:, :, :]     # last input step, keeping a length-one time axis
```

Indexing starts at zero. Slice endpoints are excluded. `:` selects a complete
axis; `-1` selects its last entry. Keeping dimensions can make later broadcasting
and repetition easier to understand.

## Reordering is different from reshaping

For sensor-wise LSTMs, first reorder `(B,T,N,F)` to `(B,N,T,F)` with `permute`,
then reshape to `(B*N,T,F)`. A direct reshape of the original layout would mix
sensor and time positions. The output must reverse this organization.

A raw NumPy slice can share memory with its source. Our constructed windows
are new arrays, so editing a returned window does not edit source observations.
Source data is float64. Model windows are float32; masks are boolean; calendar
and sensor indices are int64. Small precision differences are normal when
converting float64 values to float32 and reversing normalization.

See [the contextual forward pass](../implementations/context_lstm.md).
