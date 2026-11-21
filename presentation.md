# Traffic forecasting with contextual LSTM and PatchSTG

This is an intentionally long working draft. Each slide contains material to place on the slide and speaker notes to adapt in your own words. The complete draft will exceed 20 minutes. Select and combine slides after reviewing it.

The slide diagrams are descriptions to recreate in presentation software. Figure links point to images already in this repository. Replace presenter and course details with your own.

## Slide 1 — Title

**Insert on the slide**

- Traffic forecasting with contextual LSTM and PatchSTG
- Practical Machine Learning
- Presenter: [Your name]
- Date: [Presentation date]

**What to say**

“This project compares two approaches for forecasting traffic flow across a network of sensors. The first is a contextual LSTM that processes each sensor's history independently. The second is PatchSTG, which uses spatial attention to exchange information among sensors. I will explain the data, both models, the experiment, and what the results allow us to conclude.”

## Slide 2 — Motivation

**Insert on the slide**

- Traffic changes over time and across locations.
- Accurate short-term forecasts can support traffic management and planning.
- Recent traffic, recurring calendar patterns, and other locations may all contain useful information.

**Suggested visual:** A simple road map with several sensor markers and a small 3-hour forecast timeline.

**What to say**

“Traffic at one location varies by time of day and day of week. It may also be related to what is happening at other locations. This makes traffic forecasting a useful problem for comparing a model of individual sensor histories with a model that can exchange information across sensors.”

## Slide 3 — Research question

**Insert on the slide**

> How do a contextual LSTM and PatchSTG compare when forecasting the next three hours of traffic flow on the same sensor dataset?

- Shared input history and forecast horizon
- Matched test windows and scoring mask
- Persistence and seasonal-average baselines

**What to say**

“The goal is to compare the recorded forecasting performance of two models under a shared test protocol. Both observe 12 historical steps and predict the next 12. We also compare them with two simple rules to understand how much accuracy comes from recent traffic and regular weekly patterns.”

## Slide 4 — Dataset

**Insert on the slide**

| Property | Value |
|---|---:|
| Traffic array | `(35,040, 716, 1)` |
| Time steps | 35,040 |
| Sensors | 716 |
| Measured feature | Traffic flow |
| Assumed interval | 15 minutes |
| Assumed start | January 1, 2019 |

**Suggested visual:** A small grid with time as rows and sensors as columns.

**What to say**

“The processed SD array contains one traffic-flow value for each sensor and time step. The project assumes a continuous 15-minute timeline beginning on January 1, 2019. This gives 96 observations per day and 365 days in the array. Timestamp provenance and alignment of sensor columns with metadata remain documented assumptions.”

## Slide 5 — One forecasting example

**Insert on the slide**

```text
Input:  12 observed steps  →  Forecast: 12 future steps
         3 hours                         3 hours
```

| Example | Input indices | Target indices |
|---|---|---|
| Window 1 | 0–11 | 12–23 |
| Window 2 | 1–12 | 13–24 |

- Stride: one time step
- One window contains all 716 sensors

**What to say**

“Each example uses the last three hours to predict the following three hours. We advance the window by one time step, so consecutive examples overlap. At each forecast origin, the input uses observed readings available at that time. Each model predicts all 12 future steps together.”

## Slide 6 — Array and batch shapes

**Insert on the slide**

| Data | Shape | Meaning |
|---|---|---|
| One window | `(12, 716, 1)` | Time, sensors, flow |
| Four windows | `(4, 12, 716, 1)` | Batch, time, sensors, flow |
| Contextual LSTM input | `(4 × 716, 12, 41)` | Sensor sequences, time, features |

**What to say**

“A batch of four means four forecast windows are processed together. The LSTM rearranges the data so each sensor in each window becomes its own sequence. The contextual model has 41 input values per time step after adding calendar and sensor embeddings.”

## Slide 7 — Chronological split and leakage

**Insert on the slide**

| Split | Target indices | Purpose |
|---|---|---|
| Training | 0–21,023 | Fit model parameters |
| Validation | 21,024–28,031 | Select model and checkpoint |
| Test | 28,032–35,039 | Score selected model |

**Suggested visual:** A timeline divided into 60% training, 20% validation, and 20% test.

**What to say**

“We split targets chronologically to reflect forecasting into a later period. For example, the first validation forecast can use the final 12 training observations as history, but all its targets are in validation. Overlapping windows are processed separately. A random split of overlapping windows could place the same target readings into both training and test.”

## Slide 8 — Normalization and target mask

**Insert on the slide**

```text
Normalized input = (reading − training mean) / training standard deviation
```

- One global mean and standard deviation, fitted on training readings
- Training statistics include zero readings
- Zero targets are excluded from loss and metrics
- Predictions are evaluated in original traffic-flow units

**What to say**

“The models receive normalized traffic values using statistics fitted only on training data. We keep the same statistics for validation and test. The project excludes zero targets from scoring, while historical zeros remain in the inputs. A zero could reflect true zero flow or missing data, so this mask is a benchmark convention.”

## Slide 9 — Baselines

**Insert on the slide**

| Baseline | Prediction rule |
|---|---|
| Persistence | Repeat the last observed value for all 12 steps |
| Seasonal average | Use the training mean for the same sensor, weekday, and 15-minute slot |

**Suggested visual:** One short example showing a last observation of 100 and two seasonal lookup values.

**What to say**

“Persistence asks whether traffic will stay at the current level. Seasonal average asks what usually happens at this sensor at the forecast weekday and time. Seasonal statistics use nonzero training readings. These baselines help us judge the benefit of the neural models against simple alternatives.”

## Slide 10 — Metrics

**Insert on the slide**

- **MAE:** Average absolute prediction error in traffic-flow units
- **RMSE:** Square root of the average squared error
- **MAPE:** Average absolute error as a percentage of the actual value
- Lower values indicate smaller errors

**Suggested visual:** Actual 100, predicted 90 → absolute error 10 and percentage error 10%.

**What to say**

“MAE gives the average size of an error in the original units. RMSE gives larger errors more influence because it squares them before averaging. MAPE expresses error relative to the actual reading. Zero targets are excluded from all three calculations in this project. The overall metrics pool included forecast-target pairs.”

## Slide 11 — Why start with the small LSTM?

**Insert on the slide**

- One LSTM layer, 32 hidden units
- Traffic history only
- One shared model processes each sensor independently
- 12 past readings → 12 future predictions
- Best validation MAE: 41.13 at epoch 49

**What to say**

“The original small LSTM is a starting point for understanding the recurrent approach. It receives one sensor's traffic history at a time. The same weights are reused for all sensors, with a fresh state for each history. Its validation score was worse than seasonal average, motivating a stronger contextual model.”

## Slide 12 — How the small LSTM produces a forecast

**Insert on the slide**

```text
One sensor: 12 traffic readings
          ↓ LSTM, 32 hidden units
Final hidden state: 32 values
          ↓ linear forecast layer
12 future traffic values
```

**What to say**

“The LSTM reads the past values in time order, updating its hidden and cell states at each step. The model keeps the last hidden state as a representation of the history. A linear layer uses its 32 values to produce all 12 forecast values together.”

## Slide 13 — Contextual LSTM inputs

**Insert on the slide**

| Input for one sensor at one step | Values |
|---|---:|
| Normalized traffic | 1 |
| Time-of-day embedding | 16 |
| Weekday embedding | 8 |
| Sensor ID embedding | 16 |
| **Total** | **41** |

**What to say**

“The contextual model adds learned representations of time of day, weekday, and sensor identity. An embedding is a table of numbers learned during training. For example, a sensor ID selects a row of 16 learned values. We concatenate these with the traffic value, producing 41 inputs per historical step.”

## Slide 14 — Contextual LSTM architecture

**Insert on the slide**

```text
One sensor: (12 steps, 41 values)
          ↓ LSTM layer 1, 256 hidden units
          ↓ LSTM layer 2, 256 hidden units
Final hidden state: 256 values
          ↓ linear forecast layer
12 future values
```

- Shared parameters across sensors
- 848,644 trainable parameters

**What to say**

“The selected model has two stacked LSTM layers with width 256. The first layer produces a hidden state for each historical step, and the second processes that sequence. A linear head turns the final hidden state into 12 predictions. Sensor IDs allow sensor-specific behavior, while each sensor's observed history is processed independently.”

## Slide 15 — What a parameter count tells us

**Insert on the slide**

| Model | Trainable parameters |
|---|---:|
| Contextual LSTM, width 128 | 234,244 |
| Contextual LSTM, width 256 | 848,644 |
| PatchSTG | 2,284,844 |

- Parameters are learned weights and biases.
- Different capacities are one factor in the comparison.

**What to say**

“A parameter is a number learned during training. The selected LSTM has roughly 849 thousand parameters, while PatchSTG has about 2.28 million. More parameters give a model a different capacity and computational footprint, so the measured gap cannot be assigned to one architectural feature alone.”

## Slide 16 — PatchSTG overview

**Insert on the slide**

```text
Historical traffic + calendar
       ↓ temporal projection + embeddings
Sensor representations
       ↓ arrange into spatial patches
5 blocks: depth attention → breadth attention
       ↓ restore original sensor order
12-step forecast per sensor
```

**What to say**

“PatchSTG starts by representing each sensor's recent traffic and context. It arranges those representations into patches, then uses attention inside and across patches. After five blocks, it restores the original sensor order and predicts 12 future values for each sensor.”

## Slide 17 — PatchSTG input projection

**Insert on the slide**

| Part of one sensor's representation | Values |
|---|---:|
| Projection of 12 steps × 3 channels | 64 |
| Last historical time-slot embedding | 32 |
| Last historical weekday embedding | 32 |
| Sensor embedding | 64 |
| **Total per sensor** | **192** |

The three projected channels are normalized traffic, numeric time of day, and numeric weekday.

**What to say**

“For each sensor, the temporal projection combines all 12 historical steps and their three channels into 64 values. PatchSTG concatenates those with learned embeddings for the final historical time, weekday, and sensor identity. The result is a 192-value representation for each sensor.”

## Slide 18 — Spatial patches and padding

**Insert on the slide**

- Coordinates recursively partition the sensors.
- Training-derived traffic similarity selects extra positions.
- 716 original sensors → 1,024 positions
- 16 patches × 64 positions
- Repeated positions are restored to one output per original sensor.

**Suggested visual:** A toy grid of two patches with a repeated sensor highlighted as padding.

**What to say**

“The preparation code recursively splits sensors using latitude and longitude. Similarity calculated from training traffic helps choose sensors for padding. The final layout has 16 patches of 64 positions. Extra positions repeat sensors, and a saved mapping later retrieves one representation for each original sensor.”

## Slide 19 — What attention computes

**Insert on the slide**

```text
Sensor representation → Query, Key, Value

Scores  = Query × Keyᵀ / √192
Weights = softmax(scores)
Result  = Weights × Value
```

- One attention head
- Query, Key, and Value each have width 192

**What to say**

“A learned projection produces a query, key, and value for each position. A query is compared with the keys in its attention group. Scaling and softmax turn the scores into weights. The weights combine the value vectors, producing an updated vector for each querying position. The attention weights describe information exchange within this model.”

## Slide 20 — Depth attention within one patch

**Insert on the slide**

```text
64 positions × 192 values
Queries × Keys → 64 × 64 scores
Attention weights × Values → 64 × 192 results
```

**Suggested visual:** One patch with one highlighted query connected to the 64 positions.

**What to say**

“In depth attention, every position in one patch can attend to every position in that patch. One query gets 64 scores and uses the resulting weights to combine 64 value vectors into one 192-value result. All 64 queries do this, so the patch retains 64 updated representations.”

## Slide 21 — Breadth attention across patches

**Insert on the slide**

```text
Patch 0, position 5  ↔  Patch 1, position 5  ↔ … ↔ Patch 15, position 5
```

- Compare the same position across 16 patches.
- Depth attention happens first.
- Repeat for all 64 position numbers.

**What to say**

“Breadth attention groups the same position number across the 16 patches. Those positions are not guaranteed to represent especially similar sensors. Depth attention happens first, so a position can already contain information from other positions in its patch. Breadth can then carry that information between patches. This two-stage structure is repeated in five blocks.”

## Slide 22 — Restore and forecast

**Insert on the slide**

```text
After five blocks: (16, 64, 192)
         ↓ restore map
Original sensor order: (716, 192)
         ↓ shared forecast projection
12 predictions per sensor
```

**What to say**

“The restore map records the patch position to retrieve for each original sensor. After the attention blocks, the model selects those 716 representations in original sensor order. A shared forecast head converts each 192-value representation into 12 future predictions. Evaluation converts the normalized forecasts back to original traffic units.”

## Slide 23 — Model comparison

**Insert on the slide**

| Aspect | Contextual LSTM | PatchSTG |
|---|---|---|
| Historical traffic | 12 steps per sensor | 12 steps per sensor |
| Calendar and sensor identity | Learned embeddings | Numeric calendar inputs and learned embeddings |
| Sensor interaction | Independent histories | Depth and breadth spatial attention |
| Parameters | 848,644 | 2,284,844 |
| Optimizer | Adam | AdamW |

**What to say**

“Both models use traffic history, calendar context, and sensor identity. The LSTM shares its learned weights but processes sensor histories independently. PatchSTG can exchange information across sensor positions. They also differ in parameter count and optimization, which matters when interpreting their test scores.”

## Slide 24 — Training and validation

**Insert on the slide**

```text
Predict → masked MAE loss → backpropagation → optimizer update
                     ↓
          Validation after each epoch
                     ↓
          Select lowest validation MAE
```

| Setting | Contextual LSTM | PatchSTG |
|---|---|---|
| Completed epochs | 50 per candidate | 50 |
| Batch size | 4 | 4 |
| Optimizer | Adam, learning rate 0.001 | AdamW, initial learning rate 0.002 with schedule |
| Seed | 2023 | 2023 |

**What to say**

“Training updates weights using masked absolute error. Validation evaluates the weights without an optimizer update at the end of each epoch. The two models used the same batch size and epoch count, but their optimizers and schedules differed. Equal epochs do not imply equal computation or tuning effort.”

## Slide 25 — LSTM model selection

**Insert on the slide**

| Width | Parameters | Best epoch | Validation MAE |
|---|---:|---:|---:|
| 128 | 234,244 | 39 | 15.3385 |
| 256 | 848,644 | 37 | **15.0656** |

- Width 256, epoch 37 selected by validation MAE
- Width 384 was incomplete and deferred

**What to say**

“Both completed contextual candidates trained for 50 epochs. The width-256 candidate had the lower best validation MAE, at epoch 37, so those weights were chosen for the test evaluation. The original width-384 candidate was not completed and was deferred before the contextual test pass.”

## Slide 26 — Learning curves

**Insert on the slide**

- [LSTM learning curves](reports/final_comparison/learning_curves.png)
- Left panel: training MAE
- Right panel: validation MAE and selected epochs

**What to say**

“The curves show how the two LSTM candidates' errors changed across epochs. The validation panel uses each completed epoch's fixed weights. Training MAE pools predictions made while weights are changing, so interpret the two panels accordingly. Both final-epoch validation errors were above their respective best values, which supports choosing a validation checkpoint.”

**Preparation note:** This figure contains the LSTM candidates. The saved PatchSTG summary records its best validation score, but the full epoch history is absent from this checkout.

## Slide 27 — Frozen test evaluation

**Insert on the slide**

- Select the LSTM checkpoint using validation.
- Load selected weights and score test windows without weight updates.
- Check checkpoint identity and confirm it stays unchanged.
- Reuse the saved PatchSTG test evaluation.

**What to say**

“The contextual LSTM evaluator loads the chosen epoch-37 checkpoint, reproduces the validation score, and performs a frozen test pass. It checks model and checkpoint integrity. The PatchSTG result comes from the completed saved run and evaluation. It was not rerun for this comparison.”

## Slide 28 — Overall test results

**Insert on the slide**

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Persistence | 60.7825 | 94.3350 | 41.7801% |
| Seasonal average | 28.0045 | 53.3906 | 19.9964% |
| Contextual LSTM 256 | 18.1696 | 31.9485 | 12.0646% |
| PatchSTG | **17.1943** | **29.1370** | **11.1706%** |

6,997 matching test windows and 58,158,396 included forecast-target pairs.

**What to say**

“Both neural models outperform the two baselines in this recorded test. PatchSTG has the lowest value for all three metrics. Its MAE is about 0.98 traffic-flow units lower than the contextual LSTM's. The LSTM's MAE is approximately 5.67% higher when expressed relative to PatchSTG's MAE.”

## Slide 29 — Forecast distance

**Insert on the slide**

- [Test MAE by forecast lead](reports/final_comparison/test_horizons.png)
- Horizontal axis: 15 to 180 minutes ahead
- Vertical axis: test MAE

| Lead | Contextual LSTM MAE | PatchSTG MAE |
|---|---:|---:|
| 15 minutes | 12.62 | 11.81 |
| 180 minutes | 22.11 | 20.84 |

**What to say**

“This graph separates errors by how far ahead the model predicts. Both neural models have larger errors at the three-hour lead than at 15 minutes. PatchSTG is lower at all 12 leads in the saved results. Persistence becomes much less accurate with distance, while the seasonal-average curve is nearly flat.”

## Slide 30 — Sensor variation and coverage

**Insert on the slide**

- [LSTM sensor-error histogram](reports/final_comparison/sensor_errors.png)
- 696 sensors have at least one included test target.
- 20 sensors have no eligible targets and no reported MAE.

**What to say**

“The overall MAE combines many forecasts, but performance differs across sensors. This histogram shows the distribution of individual sensor MAEs for the selected LSTM. It does not establish why some sensors are harder to forecast, and the saved comparison lacks matching PatchSTG per-sensor forecasts.”

## Slide 31 — Reliability and limitations

**Insert on the slide**

- One recorded seed and one network
- Different capacities, optimizers, and model-selection effort
- Test period had been observed before the contextual redesign
- Timestamp, sensor alignment, and zero-mask assumptions
- No controlled timing comparison or spatial-attention ablation

**What to say**

“The result describes this particular recorded experiment. Model size, spatial inputs, optimization, and tuning differ, so we cannot attribute the gap specifically to attention. The test period had been observed before the later contextual redesign, which limits its independence. Metadata ordering, timestamps, and the interpretation of zeros also remain assumptions. We did not establish statistical significance or controlled speed differences.”

## Slide 32 — Conclusions and future work

**Insert on the slide**

- Both learned models beat persistence and seasonal average.
- PatchSTG achieved test MAE **17.19**; contextual LSTM achieved **18.17**.
- PatchSTG can combine information from sensor positions.
- Future work: multiple seeds, controlled ablations, and additional datasets.

**What to say**

“The contextual LSTM and PatchSTG both improved substantially over the simple baselines on this test. PatchSTG achieved the lower recorded error. Multiple seeds and controlled experiments that change one factor at a time would help assess reliability and the contribution of spatial information.”

## Slide 33 — Questions

**Insert on the slide**

> Thank you. Questions?

**What to say**

“Thank you for listening. I welcome questions about the data, model design, evaluation, or limitations.”

## Optional backup — Sources and implementation

**Insert on the slide**

- [Final results and limitations](reports/final_comparison/PROJECT_REPORT.md)
- [Contextual LSTM implementation](src/traffic_prediction/context_lstm.py)
- [PatchSTG implementation](src/traffic_benchmark/models/patchstg.py)
- [Spatial patch preparation](src/traffic_benchmark/data/spatial.py)
- [PatchSTG training configuration](configs/patchstg_sd_50epochs.json)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

**What to say**

“These files document the implementation, experiment configuration, saved results, and third-party attribution.”
