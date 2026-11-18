# Spatial attention in PatchSTG

Attention calculates a weighted combination of representations. Learned linear
projections create queries Q, keys K, and values V. One-head scaled attention is:

```text
weights = softmax(Q Kᵀ / sqrt(width))
output  = weights V
```

Queries and keys determine relative weights; values supply mixed information.
These are learned representations, not necessarily physical road connectivity.
Weights change with the input. The output projection, normalization, residual
connections and feed-forward layers are also trainable parts of the block.

PatchSTG applies depth attention among 64 positions within each patch, then
breadth attention among 16 patches at matching positions. Repeated blocks permit
information exchange without a single full sensor-pair matrix at every stage.
The one temporal patch has already encoded the 12 input readings.

Our contextual LSTM shares weights but processes sensor sequences independently.
PatchSTG mixes sensor representations; that is a functional difference, not a
guarantee of better accuracy. Capacity, context, optimizer and preprocessing
also differ, so an accuracy gap does not isolate attention as its cause.

Read [architecture](../patchstg/architecture.md) and
[model source](../../src/traffic_benchmark/models/patchstg.py), focusing on
SpatialAttention and DualAttentionBlock before the complete forward method.
