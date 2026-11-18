# PatchSTG: integrated model and completed experiment

PatchSTG is an integrated main model in this project, alongside contextual LSTM.
Its source, preparation, trainer/evaluator, configs, tests, selected/last
checkpoints, spatial maps, completed run reports and saved replay are included.
Comparison uses its existing measured 50-epoch run. No retraining or test
reevaluation is required or performed during integration.

## Read in this order

1. [Architecture and tensor shapes](architecture.md).
2. [Spatial preparation and sensor maps](spatial_preparation.md).
3. [Implementation and fidelity](patchstg_implementation.md).
4. [Training configuration and saved run](patchstg_training.md).
5. [Frozen evaluation and reused metrics](patchstg_evaluation.md).
6. [GPU feasibility evidence](patchstg_gpu_feasibility.md).
7. [Saved replay dashboard](patchstg_replay.md).

## Location map

| Component | Location |
| --- | --- |
| Model/attention blocks | [patchstg.py](../../src/traffic_benchmark/models/patchstg.py) |
| Spatial preparation | [spatial.py](../../src/traffic_benchmark/data/spatial.py) |
| Data pipeline | [pipeline.py](../../src/traffic_benchmark/data/pipeline.py) |
| Trainer | [trainer.py](../../src/traffic_benchmark/training/trainer.py) |
| Checkpoint validation | [checkpoint.py](../../src/traffic_benchmark/evaluation/checkpoint.py) |
| Entry point | [patchstg.py dispatcher](../../scripts/patchstg.py) |
| Full run config | [patchstg_sd_50epochs.json](../../configs/patchstg_sd_50epochs.json) |
| Saved experiment | [completed report](../../runs/patchstg_sd_seed2023_50ep/report.json) |
| Test measurements | [test report](../../reports/patchstg_sd_seed2023_test_v1.json) |
| Checks | [test_patchstg.py](../../tests/patchstg/test_patchstg.py) and neighboring test modules |
| Attribution | [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) |

Two package namespaces are retained intentionally. `traffic_prediction` contains
our LSTM implementation; `traffic_benchmark` contains integrated PatchSTG and
its infrastructure. Keeping the latter's import/source layout preserves recorded
training hashes and checkpoint behavior. They share the processed SD data and
scoring protocol, not necessarily identical Python Dataset classes.

The implementation is a code-aligned shared-protocol adaptation, not the official
author release or a full-paper-settings reproduction. That distinction remains
part of honest attribution and interpretation.
