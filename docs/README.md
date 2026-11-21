# Documentation index

Use [the root README](../README.md) to find the final deliverables. This folder
explains the methods and implementation; it does not replace the final report.

## Choose a route

- **Understand the experiment:** [overview](project_overview.md), [results/status](results_and_status.md), [fairness](concepts/17_comparison_and_fairness.md), then [final report](../reports/final_comparison/PROJECT_REPORT.md).
- **Learn from the beginning:** [reading guide](../guide.md), followed by the numbered concepts below.
- **Trace the code:** choose an implementation walkthrough below, then follow its source links.
- **Study PatchSTG:** [PatchSTG guide](patchstg/README.md), [spatial patches](concepts/19_spatial_patches.md), [attention](concepts/20_spatial_attention.md).
- **Review personal learning:** [lesson progress](learning/lessons/README.md), [saved questions](learning/questions.md), [project handoff context](learning/project_context.md).
- **Operate the project:** [commands](commands.md), [structure](project_structure.md), [GitHub assets](github.md), [dashboard](comparison_dashboard.md).

Both main models have completed saved results. Width 384 remains deferred. The
original small LSTM is educational history, rather than the final LSTM comparator.

## Concepts: one topic per file

1. [Traffic forecasting: what are we predicting?](concepts/01_forecasting_problem.md)
2. [Arrays, tensors, indexing, and shapes](concepts/02_arrays_and_tensor_shapes.md)
3. [Data quality, metadata, and provenance](concepts/03_data_quality_and_provenance.md)
4. [Chronological splits and information leakage](concepts/04_temporal_splits_and_leakage.md)
5. [Sliding windows, history, horizon, and stride](concepts/05_sliding_windows.md)
6. [Normalization and zero-target masks](concepts/06_normalization_and_masks.md)
7. [Dataset and DataLoader](concepts/07_dataset_and_dataloader.md)
8. [Persistence and seasonal-average baselines](concepts/08_baselines.md)
9. [MAE, RMSE, MAPE, and correct aggregation](concepts/09_metrics_and_aggregation.md)
10. [How the LSTM processes history](concepts/10_lstm_gates_and_states.md)
11. [Learned calendar and sensor embeddings](concepts/11_embeddings_and_context.md)
12. [Training: loss, gradients, and Adam](concepts/12_training_gradients_and_adam.md)
13. [Checkpoints, randomness, and resume](concepts/13_checkpoints_and_resume.md)
14. [Validation selection and overfitting](concepts/14_validation_selection_and_overfitting.md)
15. [GPU execution, precision, utilization, and timing](concepts/15_gpu_precision_and_timing.md)
16. [Parameters, hidden width, and capacity](concepts/16_parameter_counts_and_capacity.md)
17. [Comparing LSTM and PatchSTG responsibly](concepts/17_comparison_and_fairness.md)
18. [Frozen test evaluation](concepts/18_frozen_test_evaluation.md)
19. [Spatial patches and sensor maps](concepts/19_spatial_patches.md)
20. [Spatial attention](concepts/20_spatial_attention.md)

## Implementations: one walkthrough per Python file

Each walkthrough covers purpose, inputs/outputs, execution flow, verification,
and practical limits. Click the source column to read the implementation itself.

| Walkthrough | Source |
| --- | --- |
| [audit_data.py](implementations/audit_data.md) | [audit_data.py](../src/traffic_prediction/audit_data.py) |
| [baselines.py](implementations/baselines.md) | [baselines.py](../src/traffic_prediction/baselines.py) |
| [check_context_lstm.py](implementations/check_context_lstm.md) | [check_context_lstm.py](../src/traffic_prediction/check_context_lstm.py) |
| [check_context_training.py](implementations/check_context_training.md) | [check_context_training.py](../src/traffic_prediction/check_context_training.py) |
| [check_lstm.py](implementations/check_lstm.md) | [check_lstm.py](../reproducibility/original_sources/check_lstm.py) |
| [context_lstm.py](implementations/context_lstm.md) | [context_lstm.py](../src/traffic_prediction/context_lstm.py) |
| [continue_lstm.py](implementations/continue_lstm.md) | [continue_lstm.py](../src/traffic_prediction/continue_lstm.py) |
| [data_pipeline.py](implementations/data_pipeline.md) | [data_pipeline.py](../src/traffic_prediction/data_pipeline.py) |
| [evaluate_lstm.py](implementations/evaluate_lstm.md) | [evaluate_lstm.py](../reproducibility/original_sources/evaluate_lstm.py) |
| [finalize_context_lstm.py](implementations/finalize_context_lstm.md) | [finalize_context_lstm.py](../src/traffic_prediction/finalize_context_lstm.py) |
| [evaluation.py](implementations/evaluation.md) | [evaluation.py](../src/traffic_prediction/evaluation.py) |
| [inspect_data.py](implementations/inspect_data.md) | [inspect_data.py](../src/traffic_prediction/inspect_data.py) |
| [lstm_model.py](implementations/lstm_model.md) | [lstm_model.py](../src/traffic_prediction/lstm_model.py) |
| [train_context_lstm.py](implementations/train_context_lstm.md) | [train_context_lstm.py](../src/traffic_prediction/train_context_lstm.py) |
| [train_lstm.py](implementations/train_lstm.md) | [train_lstm.py](../src/traffic_prediction/train_lstm.py) |
| [train_lstm_diagnostic.py](implementations/train_lstm_diagnostic.md) | [train_lstm_diagnostic.py](../reproducibility/original_sources/train_lstm_diagnostic.py) |

## Supplementary references

- [Integrated PatchSTG implementation walkthrough](implementations/patchstg.md)
- [PatchSTG architecture, spatial, training and saved-run guides](patchstg/README.md)
- [Final LSTM–PatchSTG report](../reports/final_comparison/PROJECT_REPORT.md)

- [Glossary](glossary.md)
- [Original data learning guide](../temp/notes/DATA_LEARNING_GUIDE.md)
- [Project decisions and milestone record](../temp/notes/PROJECT_PLAN.md)
- [Small LSTM training guide](../temp/notes/LSTM_TRAINING_GUIDE.md)
- [Contextual capacity experiment protocol](../temp/notes/LSTM_CAPACITY_EXPERIMENT.md)
- [Source-data audit and figures](../reports/data_audit_v1/README.md)
- [Frozen small-LSTM test report](../reports/lstm_test_v1.md)
