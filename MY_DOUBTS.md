# My Doubts

## 1. Do the contextual LSTM and PatchSTG use the same input features?

| What they use | What’s similar | What’s different |
|---|---|---|
| **Past traffic flow** | Both take the last 12 readings per sensor and predict the next 12. | The **LSTM** uses each sensor’s own readings to forecast that sensor. **PatchSTG** can also use readings from other sensors through attention. |
| **Calendar** | Both use time of day and day of week from the input window. | The **LSTM** uses learned calendar vectors at all 12 steps. **PatchSTG** uses numeric calendar values at all 12 steps and adds learned calendar vectors for the last step. |
| **Sensor ID** | Both learn a vector for each sensor so they can account for its typical behavior. | The **LSTM** appends the vector to every step in that sensor’s sequence. **PatchSTG** adds it once, after summarizing the 12 input steps. |
| **Spatial information** | Both produce a separate forecast for each sensor. | **PatchSTG** uses sensor coordinates to form groups called *patches*. It uses similarity in training traffic to fill extra positions in those groups. The **LSTM** has no spatial grouping. |

For example, both models can learn that **sensor A is usually busy on Monday mornings**. PatchSTG can also learn that **a recent change at sensor B helps forecast sensor A**. That’s the difference between knowing a sensor’s **identity** and using information from **other sensors**.

[Contextual LSTM](src/traffic_prediction/context_lstm.py) · [PatchSTG](src/traffic_benchmark/models/patchstg.py) · [Spatial preparation](docs/patchstg/spatial_preparation.md)

## 2. What does `forward` do, and what should I look for in each model?

**To revisit after I understand the code.**

When I call `model(traffic, calendar)`, PyTorch runs the model’s `forward` method to turn the inputs into a forecast.

- **Contextual LSTM.** Follow the line that reshapes the inputs to `(batch * sensors, steps, 41)`, then the call to `self.lstm(sequences)`. Each sensor’s history becomes its own sequence, and the same LSTM processes each sequence independently. [Contextual LSTM source](src/traffic_prediction/context_lstm.py#L29)
- **PatchSTG.** Follow the line that uses `gather_indices` to arrange sensor representations into patches, then the loop over `self.blocks`. The attention blocks let information from one sensor influence another sensor’s forecast. [PatchSTG source](src/traffic_benchmark/models/patchstg.py#L118)

When I come back to this, I want to understand why changing only sensor B’s readings leaves sensor A’s LSTM forecast unchanged but can change its PatchSTG forecast.

## 3. How do I calculate the number of trainable parameters in each model?

A parameter is one learned number, such as a weight or a bias. Multiply the dimensions of each trainable tensor to get its parameter count, then add the counts. In PyTorch, `numel()` counts the entries in a tensor.

```python
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name, tuple(parameter.shape), parameter.numel())

total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total:,}")
```

### Contextual LSTM with 256 hidden units

| Component | Calculation | Parameters |
|---|---:|---:|
| Time-of-day, weekday, and sensor embeddings | `96×16 + 7×8 + 716×16` | 13,048 |
| First LSTM layer | `4×256×(41 + 256 + 2)` | 306,176 |
| Second LSTM layer | `4×256×(256 + 256 + 2)` | 526,336 |
| Forecast layer | `256×12 + 12` | 3,084 |
| **Total** | | **848,644** |

The first layer receives 41 values at each step, made up of 1 traffic value and embedding vectors of sizes 16, 8, and 16. The second layer receives 256 values from the first layer. Each LSTM layer has four gates. For each gate, PyTorch learns input weights, recurrent weights, and two bias vectors. [Contextual LSTM definition](src/traffic_prediction/context_lstm.py#L20)

### PatchSTG

| Component | Calculation | Parameters |
|---|---:|---:|
| Traffic and calendar projection | `64×3×12 + 64` | 2,368 |
| Sensor, time-of-day, and weekday embeddings | `716×64 + 96×32 + 7×32` | 49,120 |
| Five spatial attention blocks | `5×446,208` | 2,231,040 |
| Forecast layer | `192×12 + 12` | 2,316 |
| **Total** | | **2,284,844** |

At width 192, one block has two attention layers with a combined count of `2×4×192×(192 + 1)`, two feed-forward networks with `2×2×192×(192 + 1)`, and four normalization layers with `4×2×192`. These sum to 446,208 per block. The sensor grouping indices are fixed buffers and contribute zero trainable parameters. [PatchSTG definition](src/traffic_benchmark/models/patchstg.py#L40)
