# Checkpoints, randomness, and resume

A checkpoint is saved experiment state. `best.pt` serves model selection;
`last.pt` serves continuation. They can refer to different epochs.

Validation MAE chooses best. We use strict improvement, so an exact tie retains
the earlier epoch. Restoring a model's weights alone is insufficient for the
same Adam trajectory: optimizer moments and random-generator states matter too.

The contextual runner's last checkpoint includes current weights, Adam state,
CPU/CUDA RNG state, shuffle-generator state, epoch history, best model so far,
and experiment metadata. Mean/std are fixed model buffers included in weights.

## Why restore RNG after initialization?

Constructing layers consumes random numbers. Model construction must finish
before restoring the saved RNG state. Shuffle uses a dedicated generator.
CUDA RNG state and CPU generator state are stored separately; RNG tensors must
be restored in the form expected by their respective APIs.

## Interruptions

Saves occur at epoch boundaries. Stopping mid-epoch loses that partial epoch;
resume repeats it from the previous completed checkpoint. The runner writes a
temporary file before replacing `last.pt`, reducing exposure to partial saves.
For the contextual suite, last owns best/history so derived files can be repaired.
No design can guarantee recovery from storage corruption; preserve artifacts.

Completed candidates are loaded and verified on resume, then the incomplete
candidate continues. Source/data hashes, settings, device and recorded versions
must match. Editing training code while a run is in progress can prevent resume.

A saved seed is not a universal cross-device reproducibility guarantee.
Deterministic operations and matching state help within a recorded environment.

See [pause/resume commands](../commands.md) and
[context training implementation](../implementations/train_context_lstm.md).
