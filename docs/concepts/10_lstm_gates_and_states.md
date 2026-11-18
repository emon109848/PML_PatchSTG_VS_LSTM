# How the LSTM processes history

At time t, an LSTM combines the current feature vector x with its previous hidden
state h and cell memory c. Gates are learned vectors, not hand-coded traffic rules.

```text
f = sigmoid(Wf [x,h_previous] + bf)    # retain previous cell memory
 i = sigmoid(Wi [x,h_previous] + bi)  # admit candidate memory
 g = tanh(Wg [x,h_previous] + bg)     # propose candidate content
 o = sigmoid(Wo [x,h_previous] + bo)  # expose updated memory
 c = f * c_previous + i * g
 h = o * tanh(c)
```

Here `*` is element-wise multiplication and `[x,h]` concatenates vectors.
Sigmoid lies between zero and one; tanh between minus one and one. A forget
component near 0.9 retains most of the corresponding old memory component.
The equations summarize conceptual biases; PyTorch stores separate input and
recurrent bias vectors for each gate.

Hidden width is the number of components in h and c, not the number of sensors
or forecast steps. These components have learned numerical meanings; we cannot
assume one means morning traffic or another means congestion.

## One layer versus two

The first layer processes chronological input features and outputs a hidden
representation at every step. A second LSTM layer processes that sequence, not
just its final state. Our head uses the last step's top-layer hidden output.

Both implemented LSTMs begin each independent window with zero hidden/cell
states. They do not carry state between shuffled windows. All 12 forecasts are
produced together by a linear head; no future traffic is supplied.

Shared parameters process every sensor. This is not 716 separately trained
models, and it does not by itself provide communication between sensors.

See [small LSTM](../implementations/lstm_model.md) and
[contextual LSTM](../implementations/context_lstm.md).
