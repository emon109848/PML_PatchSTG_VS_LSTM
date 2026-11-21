> Start with the [final report](reports/final_comparison/PROJECT_REPORT.md) for
> completed results, or [presentations/](presentations/README.md) for slides.
> This guide is the longer reading route; the [documentation index](docs/README.md)
> lets you choose one concept or implementation at a time.

# Reading guide: understand the full traffic forecasting project

Visual learning aid: [LSTM–PatchSTG dashboard](runs/lstm_patchstg_replay_v1/index.html)
overlays matching forecasts and actual traffic. Read [the dashboard guide](docs/comparison_dashboard.md)
to understand local errors versus full-test metrics, and
[the exporter walkthrough](docs/implementations/comparison_dashboard.md) for its implementation.

After cleanup, implementation code lives under src/traffic_prediction and
runnable commands under scripts/. Links below have been updated. Historical
small-model-only tools are in temp/legacy_scripts for reading, not direct execution.
See [the current layout](docs/project_structure.md) before using older commands.

Scope update: width 384 is deferred; the frozen selection is width 256, epoch 37
among the two completed runs. References below to the original three-candidate
plan are historical. Finish with
[the final evaluator/report walkthrough](docs/implementations/finalize_context_lstm.md)
and [current finalization status](reports/finalization_status.md).

Start here if you want to understand what we built, why each part exists, and
how the pieces connect. Follow the stages in order rather than starting with
the longest training script.

The project compares traffic forecasting methods on processed SD data. We began
with simple baselines, built a small LSTM, and then added contextual inputs and
three predefined LSTM capacities. PatchSTG is an integrated model with source, spatial preparation, tests and saved
training/evaluation artifacts. Its completed run is reused for comparison.

## How to use this guide

For each stage:

1. Read the concept explanation.
2. Read the implementation walkthrough.
3. Open the linked Python file and trace the named functions.
4. Answer the checkpoint questions in your own words before moving on.

Keep [the glossary](docs/glossary.md) open. The [documentation index](docs/README.md)
lists every topic and Python walkthrough. You do not need to memorize every line;
aim to explain each function's inputs, outputs, and purpose.

Reading does not require rerunning training. In particular, do not start another
suite while one is already running. Existing result files are enough for the
exercises below. Avoid editing training source during the current experiment:
resume checks compare its recorded hashes.

## Stage 1 — Understand the question and the project map

Read:

- [Project overview](docs/project_overview.md).
- [Forecasting problem](docs/concepts/01_forecasting_problem.md).

Draw this pipeline on paper:

```text
Traffic data → split → normalize inputs → windows → batches
                                                   ↓
                                                predictor
                                                   ↓
                              original-unit predictions versus targets
                                                   ↓
                                 training loss or evaluation metrics
```

Keep the three experiments distinct: the original small LSTM, the new contextual
LSTM capacity study, and the integrated PatchSTG adaptation. Their results and
checkpoint formats are not interchangeable.

**Checkpoint:** What are we predicting? What do 12 history steps and 12 forecast
steps mean? Which models were implemented in this folder? Why is BigST deferred?

## Stage 2 — Understand the data and array indexing

Read:

- [Arrays and tensor shapes](docs/concepts/02_arrays_and_tensor_shapes.md).
- [Data quality and provenance](docs/concepts/03_data_quality_and_provenance.md).
- [Audit implementation](docs/implementations/audit_data.md).
- [Audit report and figures](reports/data_audit_v1/README.md).

Open [audit_data.py](src/traffic_prediction/audit_data.py). Follow `audit_values`, `audit_metadata`, and
`plot_patterns`. Then find `load_traffic` in [data_pipeline.py](src/traffic_prediction/data_pipeline.py).

Write the meaning of each axis in `(35040,716,1)`. Explain these selections:

```python
traffic[0, 0, 0]
traffic[:12, 0, 0]
traffic[0, :, 0]
```

Look at the figures: describe daily changes and differences between sensors
without inventing a cause for an abrupt drop or an all-zero interval.

**Checkpoint:** What does an SHA-256 match establish? Does a zero necessarily
mean missing data? What do we assume about timestamps and metadata ordering?
Why is ID2 not a direct index into the 716-sensor array?

## Stage 3 — Follow splits, normalization, and masks

Read:

- [Temporal splits and leakage](docs/concepts/04_temporal_splits_and_leakage.md).
- [Normalization and masks](docs/concepts/06_normalization_and_masks.md).
- [Data pipeline walkthrough](docs/implementations/data_pipeline.md).

In [data_pipeline.py](src/traffic_prediction/data_pipeline.py), trace `split_boundaries` and
`fit_normalization`. In [inspect_data.py](src/traffic_prediction/inspect_data.py), find where their
outputs are used to form training/validation/test slices.

Write down:

```text
Training:   [0,21024)
Validation: [21024,28032)
Test:       [28032,35040)

normalized = (raw - training_mean) / training_std
raw = normalized * training_std + training_mean
target_mask = target != 0
```

Trace one raw value through normalization and back. Notice that inputs are
normalized while targets remain in original units. Training statistics include
zeros; target scoring excludes zeros. These are different rules.

**Checkpoint:** Why not fit normalization on all data? What does a negative
normalized value mean? Which quantities are fixed rather than learned by Adam?
What happens to a sensor whose targets are zero throughout a split?

## Stage 4 — Build one example, then a batch

Read:

- [Sliding windows](docs/concepts/05_sliding_windows.md).
- [Dataset and DataLoader](docs/concepts/07_dataset_and_dataloader.md).
- [Inspection walkthrough](docs/implementations/inspect_data.md).

In [data_pipeline.py](src/traffic_prediction/data_pipeline.py), follow `make_window` and then
`TrafficDataset.__init__`, `__len__`, and `__getitem__`.

Trace the first example manually:

```text
target_start = 12
input steps  = 0 through 11
target steps = 12 through 23
x, y, mask   = (12,716,1)
batch x      = (4,12,716,1)
```

Next trace the first validation example: its historical steps are 21012–21023,
and targets are 21024–21035. Earlier observations provide available context;
all prediction targets still belong to validation.

Find the `range(...)` expression and derive the 21,001 / 6,997 / 6,997 window
counts. Explain why the last training batch contains one example.

**Checkpoint:** Is target_start the beginning of history? Does shuffle reorder
time inside a window? Do lazy windows mean disk streaming? How do overlapping
windows affect the number of forecast-target pairs?

## Stage 5 — Understand baselines before neural models

Read:

- [Baseline concepts](docs/concepts/08_baselines.md).
- [Baseline implementation](docs/implementations/baselines.md).

Open [baselines.py](src/traffic_prediction/baselines.py). First trace `persistence_forecast`: select
the last input value, inverse-normalize, repeat across 12 forecast steps.
Then trace `fit_seasonal_average`: group nonzero training traffic by weekday,
time slot, and sensor, with sensor/global fallback means.

Use the saved validation results to ask why repeating the latest reading can
be worse than using recurring weekly patterns. Treat your explanation as a
hypothesis supported by patterns, not proof about every sensor.

**Checkpoint:** Which baseline requires fitting? Where does the seasonal table
get its information? Why are there 672 groups? Why do some sensors require the
global fallback? Is future calendar the same as future traffic?

## Stage 6 — Understand error metrics and reporting

Read:

- [Metrics and aggregation](docs/concepts/09_metrics_and_aggregation.md).
- [Evaluation implementation](docs/implementations/evaluation.md).

Open [evaluation.py](src/traffic_prediction/evaluation.py). Trace `masked_mae`, then the two full
baseline evaluators. Locate where masks are applied and totals are accumulated.

Calculate this example yourself: prediction 120, actual 100 gives absolute
error 20, squared error 400, and percentage error 20%.
Then calculate pooled MAE from error sums 10 and 18 over 2 and 6 valid targets.
It is 3.5; simply averaging the two batch MAEs gives the wrong answer.

**Checkpoint:** Why is RMSE sensitive to large errors? Why mask before MAPE
division? Why can a small nonzero target inflate MAPE? Why must validation be
compared with validation rather than another model's test score?

## Stage 7 — Understand the original small LSTM

Read:

- [LSTM gates and states](docs/concepts/10_lstm_gates_and_states.md).
- [Parameter counts](docs/concepts/16_parameter_counts_and_capacity.md).
- [Small LSTM implementation](docs/implementations/lstm_model.md).

Open [lstm_model.py](src/traffic_prediction/lstm_model.py). Follow construction and `forward` without
getting distracted by training code yet.

Track this complete shape path:

```text
(B,12,N,1) → (B,N,12,1) → (B*N,12,1)
                  LSTM → final hidden (B*N,32)
                Linear → (B*N,12)
               restore → (B,12,N,1)
         inverse scale → predictions in original traffic units
```

Draw forget/input/output gates, candidate memory, cell state, and hidden state.
Understand their roles before trying to memorize equations. The linear head
produces all 12 future steps together.

**Checkpoint:** Why permute before reshape? Are there 716 independent parameter
sets? Are states carried between shuffled windows? How does the parameter formula
give 4,876? Does a small untrained MAE check prove forecast quality?

## Stage 8 — Follow gradients and the first real updates

Read:

- [Training, gradients, and Adam](docs/concepts/12_training_gradients_and_adam.md).
- [Model check](docs/implementations/check_lstm.md).
- [Twenty-update diagnostic](docs/implementations/train_lstm_diagnostic.md).

Read [check_lstm.py](temp/legacy_scripts/check_lstm.py), then
[train_lstm_diagnostic.py](temp/legacy_scripts/train_lstm_diagnostic.py).

Find these steps and explain why their order matters:

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(x)
loss = masked_mae(prediction, y, mask)
loss.backward()
optimizer.step()
```

Separate three claims: finite forward computation, gradient flow, and actual
parameter changes. Backward alone establishes no weight update. The short
diagnostic uses training probes, not evidence of validation generalization.

**Checkpoint:** Why clear gradients each update? What does Adam save besides
weights? How does loss reach parameters through inverse normalization? Why can
batch losses fluctuate even when training is working?

## Stage 9 — Understand epochs, checkpoints, and the small-model result

Read:

- [Validation and overfitting](docs/concepts/14_validation_selection_and_overfitting.md).
- [Checkpoints and resume](docs/concepts/13_checkpoints_and_resume.md).
- [One-epoch implementation](docs/implementations/train_lstm.md).
- [Continuation implementation](docs/implementations/continue_lstm.md).

Open [train_lstm.py](src/traffic_prediction/train_lstm.py), beginning with `validate` and then `main`.
Then read [continue_lstm.py](src/traffic_prediction/continue_lstm.py): restore state, verify protocol,
train an epoch, validate, compare best MAE, save last/best, and verify reload.

Inspect the existing [epoch-one report](temp/experiments/lstm_epoch1_v1/report.json) and
[50-epoch report](temp/experiments/lstm_50epochs_gpu_v1/report.json). First-epoch validation
MAE was 46.0824; the best after 50 epochs was 41.1316 at epoch 49.

**Checkpoint:** Why are best.pt and last.pt different? Why restore RNG state
after constructing the model? What is lost after stopping mid-epoch? Is pooled
training MAE the score of one frozen model?

## Stage 10 — Understand GPU execution and numerical checks

Read:

- [GPU precision and timing](docs/concepts/15_gpu_precision_and_timing.md).
- [Contextual feasibility probe](docs/implementations/check_context_lstm.md).

Read [check_context_lstm.py](src/traffic_prediction/check_context_lstm.py) to identify device transfers,
explicit IEEE FP32 controls, warm-up updates, CUDA synchronization, peak memory,
and exact recurrent-input layout checks. The model itself is explained next;
here focus on how a diagnostic differs from a training experiment.

The first CPU epoch and later CUDA run had different speeds. GPU utilization
alone did not tell us whether training was efficient. The first contextual GPU
parity test failed; the revised explicit-precision probe passed all widths.

**Checkpoint:** Does CUDA availability automatically move tensors? Why synchronize
around timing? Why does deterministic mode not guarantee identical sliced/batched
outputs? Why is repeated warmed-batch timing different from epoch duration?

## Stage 11 — Understand the new contextual LSTM

Read:

- [Embeddings and context](docs/concepts/11_embeddings_and_context.md).
- [Contextual model implementation](docs/implementations/context_lstm.md).
- [Capacity experiment protocol](temp/notes/LSTM_CAPACITY_EXPERIMENT.md), especially the
  fixed candidate/selection rules.

Open [context_lstm.py](src/traffic_prediction/context_lstm.py). Trace `input_calendar` separately from
`ContextLSTM.forward`.

```text
traffic 1 + time-slot 16 + weekday 8 + sensor identity 16 = 41 features
41-feature sequence → first LSTM layer → second LSTM layer
                    → final hidden state → 12 forecasts
```

Compare with the small model: context, depth, and hidden width changed together.
Among contextual candidates, only hidden width changes. Compute counts using
`12h² + 192h + 13060` for h = 128, 256, and 384.

**Checkpoint:** Is an embedding measured traffic or learned context? Does sensor
identity provide a neighboring sensor's readings? Does input_calendar include
target_start or only earlier steps? Why can context help prediction?

## Stage 12 — Understand the overnight three-candidate experiment

Read:

- [Context training implementation](docs/implementations/train_context_lstm.md).
- [Synthetic training/resume checks](docs/implementations/check_context_training.md).
- [Command reference](docs/commands.md).

Read [train_context_lstm.py](src/traffic_prediction/train_context_lstm.py) in this order:

1. `setup`: execution device and precision.
2. `run_epoch`: shared training/validation flow, enabled by optimizer presence.
3. `restore`: weights, Adam, shuffle, and CPU/CUDA RNG state.
4. `train_candidate`: epoch loop, best selection, authoritative checkpoint save.
5. `main`: metadata, resume compatibility, sequential candidates, final selection.

Then read [check_context_training.py](src/traffic_prediction/check_context_training.py) to see how a
small synthetic experiment establishes exact serialized-resume behavior.

Each candidate gets 50 epochs with matching settings. Widths run sequentially,
not together. Best validation MAE chooses the checkpoint; completed candidate
best scores choose the width. The unfinished model must complete before the
predefined selection is finalized.

Inspect candidate report.json files rather than loading large checkpoint files
just to read metrics. Each report contains epoch history for later plots.

**Checkpoint:** Why is last.pt authoritative in this newer runner? What does
--resume-suite do to completed candidates? Does resume grant 50 extra epochs?
Why does `>> logfile` preserve logs while `> logfile` replaces them? Does a
report marked running prove a process is alive?

## Stage 13 — Understand frozen test evaluation

Read:

- [Frozen evaluation concept](docs/concepts/18_frozen_test_evaluation.md).
- [Small-model evaluator implementation](docs/implementations/evaluate_lstm.md).
- [Observed small-model test report](reports/lstm_test_v1.md).

Open [evaluate_lstm.py](temp/legacy_scripts/evaluate_lstm.py). Follow identity/selection checks,
validation reproduction, frozen test inference, per-lead metric accumulation,
coverage checks, and unchanged model/checkpoint verification.

This evaluator accepts the original small-model format. It cannot directly load
a contextual checkpoint. Contextual test scoring is still a separate step after
the width/checkpoint decision is frozen.

**Checkpoint:** Why evaluate the validation-selected checkpoint rather than the
last one? How do we prove scoring did not train the model? Why do forecast-lead
errors matter? What is the limitation of reusing a previously observed test period?

## Stage 14 — Explain the findings and comparison limits

Before interpreting the comparison, follow the integrated PatchSTG path:

1. Read [spatial patches](docs/concepts/19_spatial_patches.md) and
   [attention](docs/concepts/20_spatial_attention.md).
2. Read [PatchSTG architecture](docs/patchstg/architecture.md) and
   [spatial preparation](docs/patchstg/spatial_preparation.md).
3. Follow [the implementation walkthrough](docs/implementations/patchstg.md),
   especially the gather/restore invariant and depth/breadth attention axes.
4. Inspect [saved training settings](configs/patchstg_sd_50epochs.json) and
   [completed test metrics](reports/patchstg_sd_seed2023_test_v1.json).
5. Open [the saved replay](runs/patchstg_replay_v1/index.html) if you want to see
   cached PatchSTG forecasts. Viewing it runs no forecasting model.

PatchSTG is integrated and its saved run supplies the comparison; do not rerun it
as part of reading. Explain how its context overlaps with LSTM while its spatial
processing and optimizer differ.

Read:

- [Results and status snapshot](docs/results_and_status.md).
- [Fair comparison](docs/concepts/17_comparison_and_fairness.md).
- [PatchSTG reference](docs/patchstg_reference.md).

Prepare a short explanation that separates these observations:

- Seasonal average was much stronger than persistence on validation.
- The original small traffic-only LSTM was a weak reference.
- Context and capacity together substantially improved validation performance.
- The completed contextual candidates are competitive with integrated PatchSTG
  validation. Width 256, epoch 37 was selected among the two completed widths;
  width 384 is deferred. Frozen test MAE is 18.1696 versus PatchSTG 17.1943.
- More parameters do not guarantee better results, and PatchSTG is not
  guaranteed to beat LSTM.

The final report contains the completed measurements and updated comparison plots.
Do not interpret a validation score as a test score or an architectural cause.
The comparison differs in inputs, capacity, optimization and search budget.
One seed does not establish statistical significance.

**Checkpoint:** Which conclusions are measurements, and which are hypotheses?
What extra experiment would isolate calendar context from hidden width? What
would be needed to claim spatial attention caused an improvement? Which work
still remains before writing a final conclusion?

## A manageable reading schedule

| Session | Stages | Goal |
| --- | --- | --- |
| 1 | 1–3 | Explain the task, data, splits, scale and validity assumptions |
| 2 | 4–6 | Trace a sample into a batch and calculate errors correctly |
| 3 | 7–8 | Explain LSTM forward, loss, backward and updates |
| 4 | 9–10 | Explain selection, resume, GPU execution and diagnostics |
| 5 | 11–12 | Explain contextual inputs and the capacity experiment |
| 6 | 13–14 | Explain frozen evaluation, results and fair comparison limits |

Spend as long as needed on a session. The questions are more useful than rushing
through all files. After each session, write a few sentences without copying the
documentation; return to the relevant page when an explanation is unclear.

## Final self-check: can you explain the project without the code?

You should be able to:

- Describe one sensor/time reading, one example, one batch and one epoch.
- Derive a window boundary and explain why it does not leak future targets.
- Trace normalization and original-unit scoring.
- Explain every feature the contextual LSTM receives.
- Draw the two-layer LSTM and its direct forecast head.
- Calculate parameter counts and distinguish parameters from optimizer state.
- Describe one optimizer update and one frozen validation pass.
- Explain best versus last checkpoints and what pause/resume preserves.
- Compare matching splits and distinguish model quality from diagnostic checks.
- State the completed findings, remaining work and limitations honestly.

If one item is unclear, revisit its stage rather than starting a new experiment.
