# Implementation: integrated PatchSTG package and commands

The [PatchSTG documentation index](../patchstg/README.md) links source, exact
configuration, saved experiments and measured results.

Read the implementation in this order:

1. [data/pipeline.py](../../src/traffic_benchmark/data/pipeline.py): loader, split,
   training scaler and historical calendar/window contract.
2. [data/spatial.py](../../src/traffic_benchmark/data/spatial.py): training-only
   similarity, balanced partitions, padding, gather/restore and cache checks.
3. [models/patchstg.py](../../src/traffic_benchmark/models/patchstg.py): configuration,
   attention/MLP blocks, context encoding, patch processing and forecast head.
4. [training/trainer.py](../../src/traffic_benchmark/training/trainer.py): masked
   original-unit loss, AdamW, schedule/clipping, validation and checkpoint resume.
5. [evaluation/metrics.py](../../src/traffic_benchmark/evaluation/metrics.py) and
   [checkpoint.py](../../src/traffic_benchmark/evaluation/checkpoint.py): pooled
   errors and frozen-selection/source/map integrity.
6. [evaluation/replay.py](../../src/traffic_benchmark/evaluation/replay.py): saved
   forecast payload and standalone HTML rendering.

The package retains traffic_benchmark imports and original source paths to match
the trained checkpoint's source hashes. Mathematical modules remain unchanged.
The namespace exports PatchSTG only. The separate traffic_prediction package
contains our LSTMs; both are integrated project components.

## Runnable tooling

`python scripts/patchstg.py --help` lists check/gpu-check/train/evaluate/export.
The dispatcher adds src/ to imports and delegates to byte-preserved command
scripts. These capabilities are retained for reproduction; no such PatchSTG
execution is required for the final saved-result comparison.

`scripts/check_patchstg_artifacts.py` performs hash/metadata/tensor/map verification
without constructing/running a model. `scripts/build_project_report.py` rebuilds
the comparison from saved LSTM/PatchSTG scores without inference.

The included tests live under tests/patchstg. Integration did not run model tests
or training/evaluation; it checked saved artifact identity and preserved the
recorded measurements. Earlier diagnostic evidence accompanies the import.

## Current artifacts

Full training/report/checkpoints/maps: runs/patchstg_sd_seed2023_50ep.
Saved test evaluation: runs/patchstg_sd_seed2023_test_v1.
Saved browser replay: runs/patchstg_replay_v1.
Canonical measured results: reports/patchstg_sd_seed2023_test_v1.json.
Imported source/artifact provenance: reproducibility/patchstg_import_manifest.json.
