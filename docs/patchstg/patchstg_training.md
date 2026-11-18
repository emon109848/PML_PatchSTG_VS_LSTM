# PatchSTG training configuration and completed run

The integrated saved run completed 50 epochs, seed 2023, with 262,550 optimizer
updates and no skipped empty batches. Best validation checkpoint is epoch 50:
MAE 14.9936, RMSE 25.4044, MAPE 9.0571%.

| Setting | Recorded value |
| --- | --- |
| History / horizon | 12 / 12 |
| Batch size | 4 |
| Optimizer | AdamW, foreach=False |
| Initial learning rate | 0.002 |
| Weight decay | 0.0001 |
| Gradient clipping | Global norm 5 |
| Scheduler | Epoch-end milestones 1/35/40, gamma 0.5 |
| Loss | Original-unit MAE on nonzero targets |
| Selection | Lowest full validation MAE, earliest epoch on ties |
| Input order | Seeded shuffled training windows; sequential validation |
| Numeric mode | Recorded FP32 deterministic run; no AMP/compile |

The first epoch uses 0.002; the schedule changes epoch 2 to 0.001. Configuration:
[patchstg_sd_50epochs.json](../../configs/patchstg_sd_50epochs.json).
The saved run includes resolved model/training settings and provenance, not just
this abbreviated table.

## Files

- [Full run report](../../runs/patchstg_sd_seed2023_50ep/report.json).
- [Resolved configuration](../../runs/patchstg_sd_seed2023_50ep/resolved.json).
- [Provenance](../../runs/patchstg_sd_seed2023_50ep/provenance.json).
- `best.pt`: selected weights; `last.pt`: final recovery state.
- `spatial_maps.npz`: exact training-derived spatial preparation.
- [Curated training summary](../../reports/patchstg_sd_seed2023_training.json).

The training harness is [trainer.py](../../src/traffic_benchmark/training/trainer.py).
It handles masked losses, invalid numeric values, empty-mask skips, validation,
optimizer/scheduler state, seeded shuffle and checkpoint recovery. All earlier
smoke/calibration diagnostic artifacts were imported for completeness under
`temp/patchstg_diagnostics/`; they are not substitutes for full-run scores.

Training plus validation over 50 epochs was recorded as 16,901.33 seconds,
excluding setup/checkpoint writes. It is run timing, not a controlled model-speed
benchmark. No training was launched during integration; this completed run is
used as-is for the project comparison.
