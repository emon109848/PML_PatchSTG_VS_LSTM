# PatchSTG implementation and integration

Source modules are included under [traffic_benchmark](../../src/traffic_benchmark/__init__.py).
The model, data preparation, training kernels, metrics and checkpoint utilities
retain their measured-run code bytes. The model namespace exports PatchSTG only;
BigST is outside the active scope.

## Model components

[patchstg.py](../../src/traffic_benchmark/models/patchstg.py) defines:

- `PatchSTGConfig`: fixed SD dimensions, embedding widths, five layers, dropout.
- `SpatialAttention`: one-head Q/K/V projection, scaled softmax attention and output.
- `FeedForward`: linear → GELU → dropout → linear → dropout, ratio one.
- `DualAttentionBlock`: depth within patches, breadth across patches, pre-normalization
  and residual updates.
- `PatchSTG`: validate traffic/calendar, encode history/context, gather, process,
  restore and decode normalized forecasts.

The calendar may be shared `(B,12,2)` or expanded `(B,12,716,2)`, with int64
slot/weekday values. Traffic must be finite float32 normalized flow on the same
device as model/calendar. The caller inverse-transforms predictions for scoring.

## Recorded checks and current integration checks

The included tests cover maps, training-only preprocessing, attention axes and
numerics, gradients, cache/checkpoint integrity, training/resume, masks and replay.
Their recorded diagnostic reports are included. Integration itself verified
copied hashes and saved checkpoint/report/map identities without rerunning
PatchSTG training, test scoring or forward inference.

The completed full-width CPU diagnostic recorded 16×64 padded positions and
finite backward/reload behavior. Those were implementation checks, not forecast
accuracy claims. The actual saved test result is documented separately.

## Faithfulness and license

This is a code-aligned shared-protocol adaptation. It substitutes explicit
PyTorch one-head attention for timm, preserves the recorded residual/dropout/MLP
layout, and documents calendar/tie/288-frame conventions. Full upstream model
checkpoint parity and initialization-order parity are not asserted.

Preserve [MIT attribution and notice](../../THIRD_PARTY_NOTICES.md).
Source references are recorded in [patchstg_sources.json](../../reproducibility/patchstg_sources.json).
The [integration manifest](../../reproducibility/patchstg_import_manifest.json)
records imported sources/artifacts and their hashes.
