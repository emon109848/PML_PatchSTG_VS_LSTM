# Learned calendar and sensor embeddings

An embedding is a trainable lookup table. Given an integer ID, it returns that
row's vector. Training gradients update the relevant rows. The row is not a
precomputed traffic measurement or a guaranteed interpretable feature.

| Table | Rows | Vector width | Trainable values |
| --- | ---: | ---: | ---: |
| Time of day | 96 | 16 | 1,536 |
| Weekday | 7 | 8 | 56 |
| Sensor identity | 716 | 16 | 11,456 |

At each historical step, concatenate 1 normalized traffic value, 16 slot values,
8 weekday values, and 16 sensor values: 41 inputs. The sensor vector stays the
same across that sensor's history. Calendar vectors follow the historical steps.

## Calendar reconstruction

For absolute time index t:

```text
slot = t % 96
weekday = (1 + t // 96) % 7
```

Monday is zero; the assumed January 1, 2019 start is Tuesday, so the offset is
one. `%` wraps a cycle; `//` counts complete days. `input_calendar` constructs
indices `target_start-12` through `target_start-1`, excluding the target start.
Original timestamps/timezone remain unverified.

## Why context may help

The same recent flow can imply different futures at different times or sensors.
Context lets the shared model learn conditional patterns. Sensor identity is
not coordinates or neighboring traffic; contextual LSTMs still process each
sensor's sequence independently.

We changed context, layers, and capacity together relative to the small LSTM.
Their combined improvement cannot identify each factor's separate contribution.

See [context model implementation](../implementations/context_lstm.md).
