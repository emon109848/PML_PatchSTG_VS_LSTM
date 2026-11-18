# PatchSTG architecture and shapes

The integrated PatchSTG combines temporal history encoding with spatial attention.
The completed SD configuration has five blocks, one attention head per operation,
width 192, and 2,284,844 trainable parameters. The selected checkpoint is epoch 50.

## Forward path

```text
12-step normalized flow + historical input calendar
                         ↓
Temporal signal projection + learned sensor/calendar embeddings
                         ↓
Gather/pad original sensors into spatial patches
                         ↓
Within-patch depth attention + across-patch breadth attention (five blocks)
                         ↓
Restore original 716-sensor ordering
                         ↓
Projection to 12 normalized forecasts → inverse scale for loss/metrics
```

| Stage | Shape |
| --- | --- |
| Traffic | `(B,12,716,1)` |
| Shared calendar (slot, weekday) | `(B,12,2)` |
| Expanded calendar | `(B,12,716,2)` |
| Temporal projection + embeddings | `(B,1,716,192)` |
| Padded patch grid | `(B,1,16,64,192)` |
| Restored representations | `(B,1,716,192)` |
| Normalized forecast | `(B,12,716,1)` |

The one-temporal-patch SD path uses history and horizon 12. This implementation
does not claim support for arbitrary temporal patching or unequal horizons.

## Embedding components

Signal width 64 + sensor/node width 64 + slot width 32 + weekday width 32 = 192.
The temporal Conv2d consumes traffic and normalized calendar scalars across the
12 historical positions. The last input step indexes the calendar embedding
vectors; node embeddings identify sensors. No future traffic or target calendar
is supplied by this path.

## Attention block

Depth attention operates over the 64 positions within each patch. Breadth
attention operates over the 16 patches at a corresponding within-patch position.
Their combination permits exchange across the network without one full
716-by-716 attention operation at every stage.

Each block uses pre-LayerNorm, learned Q/K/V and output projections, scaled dot
products and softmax, residual additions, and GELU feed-forward layers with width
ratio one. Dropout is 0.1 at the recorded attention/projection/MLP positions.

Padding positions duplicate selected sensors for computation; restoration
recovers originals and avoids adding duplicate forecast targets.

## Method sources and adaptations

Reviewed paper: [PatchSTG v2](https://arxiv.org/abs/2412.09972v2).
Reference code commit: [8b298b2](https://github.com/LMissher/PatchSTG/tree/8b298b230227a304a082e05241c69dca3740a73d).
The SD code configuration, rather than every paper hyperparameter, defines this
adaptation. Explicit PyTorch attention replaces timm, and numerical full-upstream
checkpoint parity has not been established. See [implementation](patchstg_implementation.md).
