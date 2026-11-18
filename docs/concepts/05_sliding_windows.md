# Sliding windows, history, horizon, and stride

A model learns from input/target examples rather than one giant sequence.
`target_start` is the first future index, not the first historical index.

```python
past = values[target_start - history:target_start]
future = values[target_start:target_start + horizon]
```

For history/horizon 12 and `target_start=12`, input is `[0,12)` and target is
`[12,24)`. At `target_start=13`, both move forward one step. Stride is one.

## Eligible origins

```python
range(max(split_start, history), split_end - horizon + 1)
```

The lower bound ensures enough history. The upper bound ensures every forecast
target lies inside its split. The extra `+1` includes the last valid start because
Python range excludes its stop.

| Split | Windows | First target | Last target start |
| --- | ---: | ---: | ---: |
| Training | 21,001 | 12 | 21,012 |
| Validation | 6,997 | 21,024 | 28,020 |
| Test | 6,997 | 28,032 | 35,028 |

Training count is `21024 - 12 - 12 + 1`. Validation/test can use earlier history,
so each count is `7008 - 12 + 1`.

Windows are built on demand; we do not allocate one expanded overlapping-window
array. The underlying NPZ is still fully loaded into memory, not streamed from
disk. Overlap also means one observed value can be scored at multiple forecast
origins/leads; metric counts refer to forecast-target pairs.

See [make_window and TrafficDataset](../implementations/data_pipeline.md).
