# Lessons 1–10

## 1. What the project predicts

Both models use 12 observed time steps to forecast the next 12 across 716 traffic sensors. Each step represents an assumed 15 minutes, so the last prediction is three hours ahead. The contextual LSTM forecasts each sensor independently. PatchSTG can use other sensors' readings from the same input window. Both models use traffic, calendar, and learned sensor identity. [Project overview](../../project_overview.md)

## 2. How traffic readings are stored

The raw array has shape `(35040, 716, 1)` for time, sensor, and traffic-flow feature. `traffic[:12, 0, 0]` selects the first 12 readings of sensor 0. `traffic[:12, :, :]` selects those time steps for every sensor.

A batch of four windows has shape `(4, 12, 716, 1)`. In the contextual LSTM, `F` counts the values describing **one sensor at one time step**. It is 41 because the model combines one traffic value, 16 time-of-day values, 8 weekday values, and 16 sensor ID values. The LSTM receives `(4 × 716, 12, 41)`: one sequence per sensor per window. [Batch and feature count](../questions.md#4-what-do-the-batch-size-and-f-mean-in-the-contextual-lstm)

## 3. Chronological splits, normalization, and masks

Training targets occupy rows 0–21,023, validation targets occupy rows 21,024–28,031, and test targets occupy rows 28,032–35,039. A window belongs to the split containing all its targets. Its input may include earlier observations from the preceding split.

The input transform is `(reading - training_mean) / training_std`. The same training statistics are applied to all splits. A negative normalized value indicates a reading below the training mean. Zero targets are excluded from error calculations. Historical zeros are still model inputs and count toward the training normalization statistics.

**Overlap and leakage.** Window 1 reads rows 0–11 and predicts 12–23. Window 2 reads 1–12 and predicts 13–24. Row 12 is available when making Window 2's forecast. Both models keep windows independent during the forward pass, even in the same batch. PatchSTG's attention connects sensors within a window. A random split of overlapping windows could put the same targets in training and test. Chronological target boundaries prevent that. [Data pipeline](../../../src/traffic_prediction/data_pipeline.py)

## 4. From one example to a batch

`TrafficDataset` slices out one example when requested and returns normalized inputs `x`, actual future readings `y`, the `target_mask`, and the first target row `target_start`. The stride is one, so the next example shifts by one row. `DataLoader` stacks examples into a batch and keeps their inputs, targets, and masks aligned. Shuffling training examples changes the order of the windows. Time inside each window stays in order. [Dataset walkthrough](../../implementations/data_pipeline.md)

## 5. What the baselines tell us

- **Persistence** repeats the last observed reading at all 12 forecast times.
- **Seasonal average** looks up the same sensor, weekday, and 15-minute slot in an average table built from nonzero training readings. It can give different predictions for Monday 8:15 and Monday 8:30.

Test MAE was **60.78** for persistence, **28.00** for seasonal average, **18.17** for contextual LSTM, and **17.19** for PatchSTG. The seasonal baseline shows how much accuracy is already available from the weekly traffic pattern. [Test results](../../results_and_status.md)

## 6. How errors become scores

For actual values of 100 and 20 and predictions of 98 and 28, the absolute errors are 2 and 8. **MAE** averages those errors to 5. **RMSE** takes the square root of the mean squared error, about 5.83, giving larger misses more influence. **MAPE** averages the percentage errors, 2% and 40%, to get 21%.

Overall MAE pools included errors across test windows, sensors, and leads. The **forecast distance** graph pools results separately at each lead, from 15 to 180 minutes. The **sensor variation** histogram groups sensors by their individual LSTM MAE. It includes 696 sensors. The 20 sensors with no eligible targets have no MAE. PatchSTG's overall MAE is about **0.98 traffic-flow units lower** than the contextual LSTM's. [Metrics and aggregation](../../concepts/09_metrics_and_aggregation.md) · [Final report](../../../reports/final_comparison/PROJECT_REPORT.md)

## 7. How the small LSTM forecasts

For one window, the model turns 716 sensor columns into 716 separate histories. It reads each history's 12 normalized traffic values in order using the same LSTM weights and fresh states for each history. The cell state carries memory across steps. Gates control what to keep, add, and expose.

After the last observed step, the hidden state has 32 numbers. A linear layer turns that summary into all 12 future predictions together. The model restores the time and sensor axes and converts the predictions back to the original units. [Small LSTM](../../../src/traffic_prediction/lstm_model.py)

## 8. How the LSTM learns

For each batch, the forward pass predicts future readings, the masked loss grades the forecasts, backpropagation calculates gradients, and Adam updates the weights. A gradient describes how changing a weight would affect the loss. All windows in the batch use the same weights for their forecasts, followed by one optimizer update. During validation the code measures error with weights fixed. [Training, gradients, and Adam](../../concepts/12_training_gradients_and_adam.md)

## 9. Epochs and checkpoints

An epoch is one pass through the training batches. The best checkpoint is chosen by lowest validation MAE. A latest checkpoint can also keep optimizer state for continuing training. The original small LSTM's best validation MAE was **41.13 at epoch 49**. The seasonal average scored **19.74**, and the later 256-unit contextual LSTM scored **15.07 at epoch 37**. [Checkpoints and resume](../../concepts/13_checkpoints_and_resume.md) · [Validation scores](../../results_and_status.md)

## 10. GPU execution and checks

Move the model and tensors used in its calculations to compatible devices. The contextual LSTM probe checks that the recurrent input layout matches exactly when run alone and in a batch. It compares forecasts within a tolerance because floating-point calculations can differ slightly. The revised probe used explicit FP32 settings and passed all three widths.

For timing, the probe warms up the GPU and waits for GPU work to finish before reading the elapsed time. Its repeated update time uses a batch already on the GPU. Full epoch timing includes additional work such as loading batches and validating forecasts. [GPU precision and timing](../../concepts/15_gpu_precision_and_timing.md)
