# PatchSTG frozen test result: reuse the completed evaluation

The model was selected using validation, then evaluated on the full test period.
The project reuses that saved evaluation; no PatchSTG reevaluation is needed.

| Quantity | Result |
| --- | ---: |
| Selected epoch | 50 |
| Test windows | 6,997 |
| Included forecast-target pairs | 58,158,396 |
| MAE | **17.1943** |
| RMSE | **29.1370** |
| MAPE (%) | **11.1706** |

Canonical [test report](../../reports/patchstg_sd_seed2023_test_v1.json) contains
checkpoint/source/data hashes, device/software settings, per-horizon metrics,
counts and timing. The [saved evaluation output](../../runs/patchstg_sd_seed2023_test_v1/report.json)
is also included. The exact best checkpoint and training report are included
under runs/patchstg_sd_seed2023_50ep.

Integration verified validation-only selection, original training source hashes,
checkpoint identity and spatial cache. The final report builder checks shared
data/scaler/splits/windows and lead-wise target counts before comparing the
saved PatchSTG scores with contextual LSTM.

Checkpoint utilities: [checkpoint.py](../../src/traffic_benchmark/evaluation/checkpoint.py).
The evaluation CLI is retained for reproducibility, but was not executed during
integration and should not be run just to regenerate an already measured score.

## Interpretation

Selected contextual LSTM test MAE is 18.1696, 5.67% higher than PatchSTG's 17.1943.
Both outperform seasonal average and persistence on matching test windows.
This is a single-seed system comparison; capacity, spatial inputs, optimizer
and search budget differ. It does not isolate spatial attention as the cause.

See [final project report](../../reports/final_comparison/PROJECT_REPORT.md).
