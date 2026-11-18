# Completed project results and status

Contextual LSTM and PatchSTG are integrated main models. Width 384 is deferred;
width 256, epoch 37 was selected among the two completed LSTM capacities by
validation MAE. PatchSTG uses its completed saved 50-epoch run and evaluation.
No PatchSTG rerun was performed for integration or comparison.

## Validation

| Model | Best epoch | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| Persistence | — | 64.1336 | 99.3846 | 41.5436 |
| Seasonal average | — | 19.7442 | 36.4470 | 12.4536 |
| Small traffic-only LSTM | 49 | 41.1316 | 64.7507 | 25.9854 |
| Contextual LSTM 128 | 39 | 15.3385 | 26.3129 | 9.3532 |
| Contextual LSTM 256 | 37 | 15.0656 | 25.8611 | 9.1746 |
| PatchSTG | 50 | 14.9936 | 25.4044 | 9.0571 |

Both contextual widths completed 50 epochs. The 384-unit candidate is incomplete
and excluded by the documented scope revision; it is not silently labeled complete.

## Frozen test comparison

| Model | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Persistence | 60.7825 | 94.3350 | 41.7801 |
| Seasonal average | 28.0045 | 53.3906 | 19.9964 |
| Contextual LSTM 256 | 18.1696 | 31.9485 | 12.0646 |
| PatchSTG | 17.1943 | 29.1370 | 11.1706 |

All use matching 6,997 test windows and 58,158,396 included target pairs.
LSTM MAE is 5.67% higher than PatchSTG. The contextual LSTM is substantially
stronger than the small reference and simple baselines, but this single-seed
comparison does not isolate architecture, context or optimizer effects.

## Included implementation and artifacts

Data audit/plots, pipeline, baselines, LSTM and PatchSTG models, spatial maps,
training/evaluation utilities, checks, configs, completed reports/checkpoints,
and the saved PatchSTG replay are local. The selected contextual test pass and
both models' integrity records are included. The comparison figures/report are
rebuilt from saved scores, not repeated PatchSTG inference.

See [final report](../reports/final_comparison/PROJECT_REPORT.md),
[LSTM measured result](../reports/final_comparison/report.json),
[PatchSTG measured result](../reports/patchstg_sd_seed2023_test_v1.json),
[PatchSTG run](../runs/patchstg_sd_seed2023_50ep/report.json), and
[integration guide](patchstg/README.md).

## Limitations

The test period was observed before this follow-up capacity redesign. Results
must not guide additional tuning presented as unseen-test evaluation. Calendar,
zero semantics and sensor pairing are benchmark assumptions. Model capacity,
spatial information, optimization and search budgets differ; report them.
Significance, scalability and controlled causal/spatial claims remain unestablished.
