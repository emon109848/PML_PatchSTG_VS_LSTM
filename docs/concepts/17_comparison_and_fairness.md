# Comparing LSTM and PatchSTG responsibly

Our active course project compares contextual temporal LSTM with integrated
PatchSTG; BigST is deferred. LSTM directly applies the course method.

## Shared protocol

The inspected integrated PatchSTG records match the dataset hash, split boundaries,
normalization, history/horizon 12, stride one, batch size four, masks and test
coverage used by the LSTM. Both use 50 training epochs and validation-based
checkpoint selection. These support a shared forecasting-task comparison.

## Remaining differences

| Aspect | Contextual LSTM | PatchSTG |
| --- | --- | --- |
| Traffic history | Per-sensor | Across sensors through spatial attention |
| Calendar and identity | Learned embeddings | Calendar projection/embeddings and sensor embeddings |
| Coordinates/similarity | Not used | Organize/pad spatial patches |
| Optimizer | Adam, fixed 0.001 | AdamW, initial 0.002, schedule/decay/clipping |
| Search budget here | Three widths | Existing configured run |
| Parameters | 234k–1.856m | 2.285m |

The old small traffic-only LSTM was an especially unequal capacity/context
comparison. The new model is a stronger reference, but the comparison still does
not isolate architecture alone. Matching epoch count aligns dataset passes,
not total computation or tuning effort.

We can claim which documented system performs better on matching validation/test
windows. We cannot attribute a difference solely to spatial attention without
controlled ablations, or claim statistical significance from one seed.

## Interpretation plan

Examine per-lead errors, predefined traffic/time conditions, sensor-level errors,
coverage, parameters and consistently measured costs. State hypotheses as such.
Pick subgroup rules using training/validation rather than looking for favorable
test examples. PatchSTG is not guaranteed to beat LSTM.

The integrated PatchSTG is a code-aligned shared-protocol adaptation, not an
official full-paper reproduction. Its completed saved run supplies the comparison.
See [PatchSTG reference](../patchstg_reference.md).
