# Implementation: context_lstm.py

Source: [context_lstm.py](../../src/traffic_prediction/context_lstm.py). Class: ContextLSTM.

This is one configurable architecture with hidden widths 128/256/384, two layers,
no dropout, and a direct 12-step head. It adds historical calendar and sensor
identity to the traffic-only reference.

## Modules and shapes

Slot embedding `(96,16)`, weekday embedding `(7,8)`, and sensor embedding
`(716,16)` concatenate with traffic to form 41 inputs per historical step.

```text
x:                              (B,12,N,1)
calendar:                       (B,12,2)
expanded calendar/sensor data:   (B,12,N,40)
combined features:              (B,12,N,41)
reordered LSTM sequences:        (B*N,12,41)
top-layer final hidden:         (B*N,H)
linear head:                    (B*N,12)
restored original-unit output:  (B,12,N,1)
```

The recurrent output at the last input time is used, not a future target.
`input_calendar(target_starts,history)` builds the historical positions only.

Optional `sensor_ids` enables isolated-sensor diagnostic calls while preserving
that sensor's original embedding row. Normal training omits it and requires all
716 sensors in original order. All states start fresh per window. Identity does
not introduce traffic communication between sensors.

Dimensions, calendar dtype/shape and basic device compatibility are checked.
Embedding lookups enforce index bounds. Models are not automatically robust to
arbitrary malformed external arrays; the data pipeline/audit define the contract.

Exact parameter counts are 234,244 / 848,644 / 1,856,260. These are fresh models,
not expanded/reused small-LSTM checkpoints.

See [embeddings](../concepts/11_embeddings_and_context.md) and
[GPU probe](check_context_lstm.md).
