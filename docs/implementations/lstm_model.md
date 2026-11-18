# Implementation: lstm_model.py

Source: [lstm_model.py](../../src/traffic_prediction/lstm_model.py). Class: `TrafficLSTM`.

This original traffic-only reference uses one LSTM layer with hidden width 32
and a linear head producing 12 steps. It has 4,876 trainable parameters and no
calendar, sensor embedding, or spatial interactions.

## Construction

Validate dimensions and scale, register fixed float32 mean/std buffers, create
`nn.LSTM(1,32,num_layers=1,batch_first=True)`, and `nn.Linear(32,12)`.
Buffers move with the model and appear in state_dict but are not optimizer weights.

## Forward path

1. Validate input `(B,12,N,1)` and basic type/nonempty dimensions.
2. Permute to `(B,N,12,1)` and reshape to `(B*N,12,1)`.
3. Run LSTM with fresh default zero states.
4. Select last-step hidden output `(B*N,32)`.
5. Apply linear head `(B*N,12)`.
6. Restore `(B,12,N,1)` and inverse-normalize.

The output head is unconstrained; predictions are not clipped to nonnegative
values in current training/evaluation. Adding clipping would change the method
and should not be silently introduced after seeing test results.

This model remains intact as an educational reference. ContextLSTM has a different
architecture and checkpoint shape; its weights cannot be directly loaded here.

See [LSTM mechanics](../concepts/10_lstm_gates_and_states.md) and
[parameter counts](../concepts/16_parameter_counts_and_capacity.md).
