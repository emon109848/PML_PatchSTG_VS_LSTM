# Traffic forecasting: what are we predicting?

The task is to predict future traffic flow at 716 sensors using past observations.
One example uses 12 historical readings and asks for 12 future readings at every
sensor. With the assumed 15-minute cadence, that is three hours of history and
three hours of forecasts.

```text
Observed history → forecasting method → predictions → compare with actual future
```

The target is the flow variable stored in the processed array. It is not speed.
The file does not itself establish exact physical aggregation semantics.

## Time step, example, batch, and epoch

- A time step contains all 716 sensors' readings at one array position.
- An example pairs a 12-step input window with a 12-step target window.
- A batch groups up to four examples for one model call.
- An epoch visits every eligible training example once.
- An optimizer update changes weights after one nonempty batch loss.

With 21,001 training windows and batch size four, an epoch has 5,251 batches.
The last has one example. Fifty epochs have 262,550 updates if no batch is skipped.

## Direct forecasts and rolling evaluation

Our LSTMs output all 12 future steps together; they do not feed predictions back
as inputs. At each later evaluation origin, actual observations available before
that origin supply history. This is rolling-origin evaluation, not a single
forecast of the entire test period.

See [windows](05_sliding_windows.md), [batching](07_dataset_and_dataloader.md),
and [the data implementation](../implementations/data_pipeline.md).
