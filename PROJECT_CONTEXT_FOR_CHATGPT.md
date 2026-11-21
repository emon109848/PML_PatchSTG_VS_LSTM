# Context for my traffic forecasting project

I’m studying my course project, `PML_PatchSTG_VS_LSTM`. It compares a contextual LSTM with PatchSTG for traffic forecasting. I want to understand the data pipeline, model architectures, training, and results well enough to explain them myself. Teach me step by step using the implementation described here. If an answer depends on code you haven’t seen, ask me to paste the relevant function.

## 1. Forecasting task and data

The raw traffic array is shaped `(35040, 716, 1)`: 35,040 chronological time steps, 716 sensors, and one measured feature, traffic flow. The project assumes an uninterrupted 15-minute cadence starting January 1, 2019, giving 96 time slots per day. Each example uses 12 past steps to predict the next 12, up to three hours ahead.

| Window | Input rows | Target rows |
|---|---|---|
| First | 0–11 | 12–23 |
| Second | 1–12 | 13–24 |
| Third | 2–13 | 14–25 |

The stride is 1. `TrafficDataset` returns normalized inputs `x`, actual targets `y`, a `target_mask`, and the first target row `target_start`. `DataLoader` stacks examples. A raw batch of four is shaped `(4, 12, 716, 1)`, where the first axis counts windows. Shuffling changes the order of windows, while time inside each window stays in order.

| Split | Target rows | Role |
|---|---|---|
| Training | 0–21,023 | Learn weights |
| Validation | 21,024–28,031 | Compare candidates and select checkpoints |
| Test | 28,032–35,039 | Score the selected models |

All 12 targets must be in the window's split. Earlier inputs can come from the preceding split because those readings are available at forecast time. For example, the first validation window uses rows 21,012–21,023 and predicts 21,024–21,035.

I already clarified that row 12 is a target for the first window and an input for the second. This overlap is safe within a batch because each forecast uses its own window. PatchSTG exchanges information among sensors within a window. A random split of overlapping windows could put the same target observations in training and test, creating leakage.

The project computes one global mean and standard deviation from training readings, including zeros. Every split uses those fixed statistics: `normalized = (raw - training_mean) / training_std`. Scoring uses original traffic-flow units. Only nonzero actual targets contribute to errors. Historical zeros remain in the inputs. Masking zeros is a benchmark convention because zeros may represent real zero flow or missing data.

## 2. Baselines

- **Persistence** repeats the last observed reading at all 12 future steps.
- **Seasonal average** looks up the mean of nonzero training readings for the same sensor, weekday, and 15-minute slot. Monday 8:15 and Monday 8:30 can have different predictions. The future clock time is already known when the forecast is made.

## 3. Models

### Original small LSTM

One layer, 32 hidden units, and traffic alone as input. It rearranges `(B, 12, 716, 1)` into `(B × 716, 12, 1)`. Each sequence is one sensor’s 12 readings from one window. The same learned weights process every sequence, with fresh recurrent states.

It reads the 12 steps in order and keeps the last 32-value hidden state. A linear layer converts that into 12 future values at once. The model restores the axes to `(B, 12, 716, 1)` and converts predictions back to the original flow units.

### Contextual LSTM

The selected model has two layers, 256 hidden units per layer, and **848,644 trainable parameters**.

| Input for one sensor at one historical step | Values |
|---|---:|
| Normalized traffic flow | 1 |
| Learned time-of-day embedding | 16 |
| Learned weekday embedding | 8 |
| Learned sensor ID embedding | 16 |
| **Total (`F`)** | **41** |

An embedding is a row of learned numbers looked up using a category. The time-of-day table has 96 rows, the weekday table has 7, and the sensor table has 716. Calendar indices are derived from the historical rows. The same sensor ID vector is reused at all 12 steps of that sensor’s sequence.

The model rearranges `(B, 12, 716, 41)` to `(B × 716, 12, 41)`. For four windows, that is `(2864, 12, 41)`. The first layer produces a hidden state at each step. The second layer reads those states in order. The forecast head uses the second layer’s final hidden state to produce all 12 predictions. Sensor identity gives sensor-specific context, and the model processes each sensor’s observed traffic independently.

### PatchSTG

PatchSTG has **2,284,844 trainable parameters**. It uses the same core information as the contextual LSTM: historical traffic, time of day, weekday, and sensor identity. It encodes them differently:

- It projects 12 historical steps with three channels: traffic, numeric time of day, and numeric weekday.
- It adds learned calendar embeddings for the last historical step and a learned sensor embedding.
- Sensor coordinates form spatial patches. Similarity derived from training traffic helps select sensors for extra patch positions. The model gathers sensor representations into the patch layout and restores sensor order at the end.
- Five attention blocks combine information within and across patches. Sensor B’s observed traffic can influence the forecast for sensor A in the same window.

## 4. Training and selection

For each batch, the model forecasts, computes masked absolute error, backpropagates gradients, and updates shared weights. One epoch is a pass through the training batches. Validation scores the weights at the end of the epoch with no optimizer update.

Lowest validation MAE determines the best checkpoint. A latest checkpoint can also save optimizer and random state for resuming. The contextual LSTM used Adam. PatchSTG used AdamW with a different learning rate schedule, weight decay, and gradient clipping.

| Candidate | Best validation epoch | Validation MAE |
|---|---:|---:|
| Original small LSTM | 49 | 41.1316 |
| Seasonal average | — | 19.7442 |
| Contextual LSTM, width 128 | 39 | 15.3385 |
| Contextual LSTM, width 256 | 37 | 15.0656 |
| PatchSTG, saved run | 50 | 14.9936 |

The 128- and 256-unit runs completed 50 epochs each. The unfinished 384-unit run was deferred. The project selected the 256-unit epoch-37 checkpoint by validation MAE. It reuses a completed 50-epoch PatchSTG run with seed 2023 and batch size 4.

Frozen test evaluation loads the selected checkpoint, makes predictions with fixed weights, and calculates test scores. The contextual LSTM evaluator also verified that the checkpoint file stayed unchanged.

## 5. Final test results

| Method | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Persistence | 60.7825 | 94.3350 | 41.7801% |
| Seasonal average | 28.0045 | 53.3906 | 19.9964% |
| Contextual LSTM, width 256 | 18.1696 | 31.9485 | 12.0646% |
| PatchSTG | 17.1943 | 29.1370 | 11.1706% |

The methods were scored on matching 6,997 test windows and 58,158,396 included forecast–target pairs.

- **MAE** is the average absolute error in traffic-flow units. PatchSTG’s MAE is about **0.98 flow units lower** than the contextual LSTM’s.
- **RMSE** squares the errors, averages them, and takes the square root. Larger errors have more influence.
- **MAPE** averages absolute errors as percentages of the actual readings. Zero actual readings are excluded.
- **Pooled MAE** weights each included prediction equally. Averaging separate sensor MAEs would instead weight each scored sensor equally.

### Report figures

- **Learning and capacity** shows training and validation MAE by epoch for the 128- and 256-unit contextual LSTMs. The validation dots mark their best epochs. Adding the full PatchSTG curves requires its original `runs/patchstg_sd_seed2023_50ep/report.json`. The checked-in summary has the best validation point.
- **Forecast distance** shows test MAE for each of the 12 leads, from 15 to 180 minutes. PatchSTG has the lowest MAE at every lead. Its MAEs at 15 and 180 minutes are 11.81 and 20.84, versus 12.62 and 22.11 for the contextual LSTM.
- **Sensor variation and data coverage** shows a histogram of contextual LSTM per-sensor test MAE. The x-axis shows error ranges and the y-axis counts sensors in each range. Of 716 sensors, 696 have eligible test targets and 20 have no MAE.

### What the comparison supports

PatchSTG performed better in these recorded runs. Both learned models beat both baselines. Spatial information, capacity, optimization, and tuning history differed between the models. This single-seed comparison cannot isolate the contribution of spatial attention. The test period had already been observed before the follow-up contextual capacity experiment, which limits its independence as an evaluation of the revised project.

## 6. My progress and how I want to learn

I’ve covered Lessons 1–11: the project overview, data shapes, chronological splits, leakage, normalization, masks, windows and batching, baselines, metrics, the small LSTM, gradients and Adam, checkpoints, GPU checks, and the contextual LSTM with embeddings.

The remaining lessons are:

12. Comparing hidden widths and resuming training.
13. Frozen test evaluation in the code.
14. PatchSTG architecture and what the comparison can establish.

Use plain language and trace one reading, sensor, window, or batch at a time. Explain what each axis counts. When pointing me to a `forward` method, name the relevant line and say what I should look for. If I’m confused, use a small concrete example or check which prerequisite I’m missing. Dense diagrams have been hard for me to follow. Continue one lesson at a time when I say “next,” and answer specific questions directly.

**Start with Lesson 12. Explain what changes when we go from 128 to 256 hidden units and how validation tells us which version to choose.**

## Files I can share for code-specific questions

- `guide.md` and `lessons/` for the learning sequence.
- `MY_DOUBTS.md` for saved questions and calculations.
- `reports/final_comparison/PROJECT_REPORT.md` for results and figures.
- `src/traffic_prediction/data_pipeline.py` for splits, windows, and masks.
- `src/traffic_prediction/lstm_model.py` and `src/traffic_prediction/context_lstm.py` for the two LSTM models.
- `src/traffic_prediction/train_context_lstm.py` and `src/traffic_prediction/finalize_context_lstm.py` for training, selection, and test scoring.
- `src/traffic_benchmark/models/patchstg.py` and `src/traffic_benchmark/data/spatial.py` for PatchSTG.
