# Traffic forecasting presentation — corrected version

23 main slides for approximately 21.1 minutes, excluding questions.

The detailed deck appends 21 optional lesson/backup slides. Original decks are preserved under temp/presentation_before_review_2026-11-21.

All curves use saved measurements. PatchSTG was not retrained or reevaluated. Compact slide data under reports/presentation supports rebuilding on another PC.


## Slide 1 — Forecasting traffic across 716 sensors

**Section:** Opening · **Suggested minutes:** 0.45

**Slide content**

- Contextual LSTM versus PatchSTG

- SH Emon | Practical Machine Learning / November 2026

- 12 observed steps → 12 future steps | Processed LargeST-SD


**Speaker notes**

Introduce the practical forecasting question and the course-method LSTM. PatchSTG is integrated; its completed run is reused, not retrained. Present measured comparisons rather than promising a particular architecture should win.


## Slide 2 — Which system forecasts the next three hours better?

**Section:** Question · **Suggested minutes:** 0.9

**Slide content**

- Temporal context

- Recent flow, time of / day, weekday and / sensor identity

- Spatial interaction

- Can other sensors' / histories improve the / complete forecasting / system?

- Shared assessment

- Same history, horizon, / chronological targets / and zero mask

- A practical system comparison—not a controlled spatial-attention / ablation.


**Speaker notes**

Traffic can reflect recent trends, recurring calendar patterns and relationships between locations. Compare a contextual temporal LSTM with PatchSTG on matched forecast targets. We do not isolate the causal effect of spatial attention because capacity and optimization also differ.


## Slide 3 — Dataset: processed LargeST-SD, San Diego

**Section:** Data · **Suggested minutes:** 0.85

**Slide content**

- Property

- Value

- Time steps × sensors × feature

- 35,040 × 716 × 1

- History → horizon

- 12 → 12 steps; 3 hours → 3 hours

- Sampling / origin

- Assumed 15 min; 2019-01-01

- Forecast variable

- Traffic flow in original source units

- One example contains all sensors; each model uses that / information differently.


**Speaker notes**

The processed array is treated as SD 2019: 35,040 time positions, 716 sensors, one traffic-flow feature. Fifteen-minute cadence and January 1 midnight origin are assumptions: the NPZ has no timestamps or sensor IDs. Metadata ordering remains assumed. Cite Liu et al., LargeST, NeurIPS 2023.


## Slide 4 — Chronological targets prevent overlapping-window leakage

**Section:** Protocol · **Suggested minutes:** 1.15

**Slide content**

- Train: 60%

- 21,024 frames / Fit weights and scale

- Validate: 20%

- 7,008 frames / Choose model/epoch

- Test: 20%

- 7,008 frames / Freeze, then score

- Training-only normalization. Zero inputs remain; zero targets are / masked.

- Rolling forecasts may use earlier observed history across a split / boundary.


**Speaker notes**

Training targets are [0,21024), validation [21024,28032), test [28032,35040). All 12 targets stay inside their split; earlier observations can supply history. Fit one global scaler on training including zeros. Keep zero inputs, exclude zero targets as a benchmark convention, and score in original units.


## Slide 5 — Baselines make the learned-model results meaningful

**Section:** References · **Suggested minutes:** 0.95

**Slide content**

- Method

- Prediction rule

- Persistence

- Repeat the latest observation

- Seasonal average

- Training mean: sensor × weekday × slot

- MAE: absolute error | RMSE: larger-error penalty / MAPE: percentage error; zero actuals excluded

- All results below use pooled original-unit metrics. Lower is better.


**Speaker notes**

Persistence repeats the latest reading. Seasonal average uses nonzero training means for sensor, weekday and 15-minute slot, with sensor/global fallbacks. MAE is mean absolute error in original units; RMSE emphasizes large errors; MAPE is relative percentage error. All metrics exclude zero targets and pool errors/counts, not unweighted batch means.


## Slide 6 — Context makes each historical LSTM step 41-dimensional

**Section:** LSTM · **Suggested minutes:** 0.85

**Slide content**

- Historical input

- Values

- Normalized flow

- 1

- Time-slot embedding

- 16

- Weekday embedding

- 8

- Sensor-ID embedding

- 16

- 1 + 16 + 8 + 16 = 41 values per sensor per input step.


**Speaker notes**

A learned embedding is a vector selected by an integer ID, not a new traffic measurement. The same sensor embedding is reused over its 12 historical steps. Input calendar is reconstructed for past steps only. The shared LSTM still processes sensor histories independently.


## Slide 7 — One shared two-layer LSTM predicts all 12 leads

**Section:** LSTM · **Suggested minutes:** 1.15

**Slide content**

- Input sequence

- 12 historical steps / 41 values per step

- Two LSTM layers

- 256 hidden units each / Fresh recurrent states

- Forecast head

- Final hidden → 12 / Original flow units

- Shared parameters across sensors; no exchange of neighboring / traffic histories.


**Speaker notes**

Reorder (B,12,N,41) to (B,N,12,41), then combine batch and sensor axes. First LSTM layer produces a hidden sequence, the second reads it. Take the top layer's final hidden output, apply a linear 256-to-12 head, restore sensor order and inverse-normalize. Fresh states per independent window. All future outputs are direct, not autoregressive.


## Slide 8 — PatchSTG encodes history, then exchanges spatial information

**Section:** PatchSTG · **Suggested minutes:** 1.25

**Slide content**

- Encode

- 12-step history / 192 values per sensor

- Patch + attend

- 16 × 64 positions / Five dual-attention / blocks

- Restore + predict

- 716 original sensors / 12 forecasts each

- Coordinates form patches; training-only similarity chooses / padding positions.


**Speaker notes**

Fang et al., KDD 2025, provides the PatchSTG reference. The integrated shared-protocol adaptation projects 12 steps of traffic/numeric calendar, concatenates learned context to width 192, gathers 16 patches of 64 padded positions, applies five dual-attention blocks, restores 716 originals, and forecasts 12 normalized values. Inverse scaling occurs at loss/evaluation. Coordinates and training-derived similarity prepare patches; duplicates are not extra targets.


## Slide 9 — Attention combines weights and Values as separate inputs

**Section:** PatchSTG · **Suggested minutes:** 1.35

**Slide content**

- Weights = softmax(QKᵀ / √192) / Depth shape: 64 × 64

- Values: separate learned / projection / Depth shape: 64 × 192

- Weights × Values → 64 × 192

- 1. Depth: 64 positions within a patch / 2. Breadth: 16 matching positions across patches, jointly


**Speaker notes**

Queries and Keys produce softmax-scaled scores. Values are a separate projection of representations, not derived from the scores. Multiply the weights matrix by Values. Depth attention uses 64 tokens in one patch. Breadth attention uses 16 tokens at matching positions across patches, attending jointly rather than passing sequentially from patch 0 to patch 15. Depth runs first within each block.


## Slide 10 — Shared task and context; different modeling budgets

**Section:** Comparison · **Suggested minutes:** 0.95

**Slide content**

- Aspect

- Contextual LSTM

- PatchSTG

- History / horizon

- 12 / 12

- 12 / 12

- Calendar + sensor ID

- Yes

- Yes

- Sensor interaction

- Independent histories

- Spatial attention

- Parameters

- 848,644

- 2,284,844

- Compare complete systems; disclose the remaining differences.


**Speaker notes**

Both have historical traffic, time/weekday and sensor identity. LSTM is per-sensor; PatchSTG mixes spatial representations and uses coordinates/similarity. PatchSTG has 2,284,844 parameters versus 848,644. Similar context and shared splits improve comparability but parameter count, optimization and search effort differ. No causal attention-only claim.


## Slide 11 — Training protocol: 50 epochs, batch 4, seed 2023

**Section:** Training · **Suggested minutes:** 1.05

**Slide content**

- Setting

- Contextual LSTM

- PatchSTG

- Optimizer

- Adam

- AdamW

- Learning rate

- 0.001 fixed

- 0.002 + schedule

- Decay / clipping

- None / none

- 0.0001 / norm 5

- Model selection

- Lowest validation MAE

- Lowest validation MAE

- Same data passes; different optimizer, schedule, capacity and / tuning effort.


**Speaker notes**

Show the actual settings rather than only the generic learning loop. LSTM Adam has fixed lr .001 without schedule/weight decay/clipping. PatchSTG AdamW starts .002, decays at epoch ends 1/35/40 by .5, weight decay .0001 and gradient clipping 5. Both use original-unit masked MAE and full validation. Same epoch count is not equal compute or search budget.


## Slide 12 — Validation selects width 256, epoch 37

**Section:** Selection · **Suggested minutes:** 0.8

**Slide content**

- Candidate

- Best epoch

- Validation MAE

- LSTM 128

- 39

- 15.3385

- LSTM 256 — selected

- 37

- 15.0656

- PatchSTG

- 50

- 14.9936

- Both LSTMs completed 50 epochs. Width 384 is deferred, not / silently counted.


**Speaker notes**

The two completed LSTM candidates each trained 50 epochs. Width 256 best validation is 15.0656 at epoch 37; width 128 is 15.3385 at epoch 39. PatchSTG selected epoch 50 at 14.9936. Width 384 was deferred after interim validation and before the contextual test pass, revising the search scope. Do not call it a completed three-width search.


## Slide 13 — Learning curves now include PatchSTG's complete run

**Section:** Analysis · **Suggested minutes:** 1.25

**Slide content**

- Figure: [learning_curves_all_models.png](reports/presentation/learning_curves_all_models.png)

- Training improves steadily; validation fluctuates. Select / checkpoints, not final epochs.


**Speaker notes**

All three curves use their real saved 50-epoch reports. Training MAE pools evolving weights, while validation uses completed-epoch frozen weights. The 256 LSTM lowers training error relative to 128, but validation gains are modest. PatchSTG starts higher, then ends with the best recorded validation score. Different optimization prevents attributing trajectories solely to architecture. Dots mark each model's selected epoch.


## Slide 14 — Frozen test: identical origins, targets and masks

**Section:** Evaluation · **Suggested minutes:** 0.75

**Slide content**

- Freeze

- LSTM epoch 37 / PatchSTG epoch 50

- Score

- 6,997 test windows / 58,158,396 valid pairs

- Verify

- Matching / source/masks / No weight updates

- PatchSTG scores are reused from its saved run; no retraining or / new inference.


**Speaker notes**

Select before scoring. The contextual evaluator rechecked full validation, scored test under eval/no-grad, and verified unchanged model/checkpoint state. Reuse PatchSTG's saved completed test pass with matching hashes, origins and lead-wise target counts; it was not rerun. No significance claim from millions of overlapping targets: these are correlated forecast-target pairs, not independent experiments.


## Slide 15 — PatchSTG records the lowest error on all three metrics

**Section:** Results · **Suggested minutes:** 1.15

**Slide content**

- Method

- MAE ↓

- RMSE ↓

- MAPE ↓

- Persistence

- 60.78

- 94.34

- 41.78%

- Seasonal average

- 28.00

- 53.39

- 20.00%

- Contextual LSTM

- 18.17

- 31.95

- 12.06%

- PatchSTG

- 17.19

- 29.14

- 11.17%

- PatchSTG: 0.98 flow units lower MAE; 5.37% reduction relative to / LSTM.


**Speaker notes**

Read the four-method table. LSTM test MAE 18.1696 versus PatchSTG 17.1943 is a difference of .9753 original flow units. PatchSTG reduces MAE by 5.37% relative to LSTM. Saying LSTM is 5.67% higher uses the other denominator; do not mix them. These are observed single-seed differences, not statistical significance.


## Slide 16 — Longer leads hurt persistence; learned forecasts stay stronger

**Section:** Analysis · **Suggested minutes:** 1.25

**Slide content**

- Figure: [forecast_horizon_comparison.png](reports/presentation/forecast_horizon_comparison.png)

- Neural-model zoom makes the gap visible without hiding the / baseline context.


**Speaker notes**

The left plot includes all methods. The right deliberately zooms to the two neural curves with a clearly labeled different y range, so their modest gap is visible. At 15 min: LSTM12.62, Patch11.81; at180:22.11 and20.84. PatchSTG is lower at every recorded lead. Absolute gaps grow .81 to1.27, but relative gains do not monotonically establish increasing spatial benefit. Seasonal averages are almost flat; persistence grows sharply.


## Slide 17 — Competitive LSTM accuracy with 37.1% of the parameters

**Section:** Analysis · **Suggested minutes:** 0.85

**Slide content**

- LSTM 256

- 848,644 parameters / Test MAE 18.17

- PatchSTG

- 2,284,844 parameters / Test MAE 17.19

- Trade-off

- 37.1% of parameters / 5.67% higher MAE

- Fewer parameters ≠ demonstrated faster inference. Runtime was / not controlled.


**Speaker notes**

Parameter count is a capacity/storage measure, not an inference-speed benchmark. Selected LSTM848644 is37.1% of Patch2284844, while its error is5.67% higher relative to PatchSTG. This is an informative system trade-off, not proof of efficiency superiority or that parameters alone explain accuracy. We did not control timings.


## Slide 18 — Sensor averages hide a long tail of hard cases

**Section:** Analysis · **Suggested minutes:** 1.0

**Slide content**

- Figure: [sensor_errors_lstm.png](reports/presentation/sensor_errors_lstm.png)

- LSTM-only diagnostic: 696 scored sensors; 20 have no eligible test / targets.


**Speaker notes**

The histogram is LSTM-only; saved PatchSTG reports lack paired per-sensor predictions, so no comparative histogram is invented. 696 sensors have at least one eligible target,20none. A few large errors create a right tail. This motivates investigation but does not prove broken sensors or spatial causes. Overall pooled MAE weights forecast pairs, unlike equal weighting of these sensor means.


## Slide 19 — A matched forecast connects the numbers to the task

**Section:** Demonstration · **Suggested minutes:** 1.1

**Slide content**

- Figure: [matched_forecast_example.png](reports/presentation/matched_forecast_example.png)

- First replay origin + first sensor, fixed by position—not selected by / forecast quality.


**Speaker notes**

This example is sensor 1114091 at the first chronological replay origin, not selected for favorable errors. Show past observations, both forecasts and actual future. The gold future was not an input. Values are cached display-rounded outputs. If time allows, open runs/lstm_patchstg_replay_v1/index.html and change sensor/lead once; otherwise this figure is the offline demo. Do not tune using these test examples.


## Slide 20 — What the experiment does—and does not—establish

**Section:** Limitations · **Suggested minutes:** 1.0

**Slide content**

- • One recorded seed and one network; no significance claim.

- • Capacity, spatial inputs, optimizer and search budgets differ.

- • Test period was observed before the contextual redesign.

- • Calendar, sensor pairing and zero semantics remain assumptions.

- Next: multiple seeds, controlled context/spatial ablations and new / evaluation periods.


**Speaker notes**

Be explicit: one seed and one network; capacity, spatial information, optimization and tuning history differ. The same test period had already been observed before the contextual redesign. Timestamp/metadata pairing/zero meanings remain assumptions. Do not attribute gains solely to attention or claim significance/scalability. New seeds and controlled ablations are future work, not completed experiments.


## Slide 21 — Contextual LSTM is competitive; PatchSTG is more accurate here

**Section:** Conclusion · **Suggested minutes:** 0.7

**Slide content**

- • Both learned models outperform persistence and seasonal / average.

- • PatchSTG MAE 17.19 versus contextual LSTM 18.17.

- • LSTM uses 37.1% of PatchSTG's parameters.

- An observed system-level result—not proof that spatial attention / alone caused the gain.


**Speaker notes**

Summarize three findings in your own words. Both learned models beat simple baselines. PatchSTG is the lowest recorded error. Contextual LSTM provides competitive accuracy at lower parameter count; we cannot claim speed superiority. Calendar/capacity were changed together from the small model, so no isolated causal attribution. Invite questions about data leakage, state/shape and attention.


## Slide 22 — Sources, implementation and reproducible evidence

**Section:** References · **Suggested minutes:** 0.35

**Slide content**

- Fang et al. | KDD 2025 / Efficient Large-Scale Traffic Forecasting with Transformers: / A Spatial Data Management Perspective / arxiv.org/abs/2412.09972v2

- Liu et al. | NeurIPS 2023 / LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting / arxiv.org/abs/2306.08259

- Project: github.com/emon109848/PML_PatchSTG_VS_LSTM / Saved evidence: reports/final_comparison/PROJECT_REPORT.md


**Speaker notes**

PatchSTG: Yuchen Fang et al., Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective, accepted SIGKDD2025, arXiv2412.09972v2. LargeST: Xu Liu et al., LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting, NeurIPS2023 datasets/benchmarks. Project uses adapted processed SD and a code-aligned PatchSTG implementation; see attribution, pinned source commit and method deviations. URLs are written on the slide for the exported PDF too.


## Slide 23 — Questions

**Section:** Discussion · **Suggested minutes:** 0

**Slide content**

- Thank you

- Traffic forecasting with contextual LSTM and PatchSTG


**Speaker notes**

Thank the audience. Use the detailed deck's appendix to answer tensor-shape, masking, gate, map, optimizer or numerical-check questions. Do not present backup material unless needed.


## Backup 24 — How overlapping windows move

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- WINDOW 1

- Input 0–11 / Targets 12–23

- WINDOW 2

- Input 1–12 / Targets 13–24

- WINDOW 3

- Input 2–13 / Targets 14–25

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

Each new forecast uses actual readings available at its forecast time. Earlier target readings can appear as inputs later. Windows in a batch remain independent. Randomly splitting overlapping targets across train and test could leak answers.


## Backup 25 — Dataset, DataLoader and a batch

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- DATASET

- Builds one window / x, y, mask, / target_start

- DATALOADER

- Stacks four examples / shape (4, 12, 716, 1)

- TIME ORDER

- Training windows / shuffle / steps inside each stay / ordered

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The dataset uses target start indices to slice the original array on demand. The DataLoader groups examples, keeping inputs, targets and masks aligned. Four is the number of forecast windows in a batch.


## Backup 26 — Normalization: a numerical example

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- RAW VALUE

- 140 traffic-flow units

- TRAINING STATS

- Mean = 100 / Standard deviation = / 20

- MODEL INPUT

- (140 − 100) / 20 = 2

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

These numbers are illustrative. The normalized value of 2 means the reading is two training standard deviations above the training mean. A raw value of 80 normalizes to minus one. All splits use the same training statistics.


## Backup 27 — Masking and sensor coverage

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- ACTUAL = 30

- Prediction = 25 / Included error = 5

- ACTUAL = 0

- Prediction = 25 / Target excluded

- COVERAGE

- 696 scored sensors / 20 have no eligible / test targets

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The zero mask is applied to actual future targets, not to model predictions. A model prediction of zero is still scored when the actual target is nonzero. Zeros may represent true zero flow or missing values.


## Backup 28 — MAE, RMSE and MAPE by example

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- ACTUAL / PREDICTED

- 100 / 98 and 20 / 28 / Absolute errors 2 and / 8

- MAE / RMSE

- MAE = 5 / RMSE = 5.83

- MAPE

- 2% and 40% / Average = 21%

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

MAE averages absolute errors in the flow units. RMSE squares errors before averaging, so larger errors have greater influence. MAPE divides the absolute error by each nonzero actual target before averaging.


## Backup 29 — The LSTM state across time

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- STEP 1

- Read traffic at t₁ / Update hidden and / cell states

- STEP 2

- Read traffic at t₂ / Use states from step 1

- STEP 12

- Read last observation / Keep final hidden / state

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

For one sensor, the LSTM reads 12 values in order. The cell state carries information along the sequence. Gates regulate what is retained, added and exposed. Each window and sensor sequence begins with fresh states.


## Backup 30 — One shared LSTM for 716 sensors

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- INPUT

- One window / (12, 716, 1)

- REARRANGE

- 716 independent / sequences / each (12, 1)

- SHARED MODEL

- Same LSTM weights / fresh states per / sequence

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The project does not train 716 separate LSTMs. Each sensor history passes through the same learned model. Within this model, one sensor’s current readings do not enter another sensor’s sequence.


## Backup 31 — How a linear layer predicts 12 values

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- INPUT

- Final hidden state / 32 values for small / LSTM

- ONE FORECAST

- 32 weights + one bias / combine all 32 values

- OUTPUT

- 12 sets of weights / produce 12 forecast / steps

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

For each forecast time, a linear layer multiplies each of the 32 hidden values by its learned weight and sums them with a bias. The 12 outputs are generated together. The contextual LSTM uses 256 hidden values at its selected width.


## Backup 32 — Contextual LSTM: how 41 is formed

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- TRAFFIC

- 1 normalized reading

- CALENDAR

- 16 time-of-day values / 8 weekday values

- SENSOR ID

- 16 learned values / reused across 12 steps

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

At one sensor and one historical time step, the model concatenates 1 plus 16 plus 8 plus 16 values. For 12 past steps, one sensor has shape 12 by 41. Sensor identity is learned context and supplies no other sensor’s current traffic readings.


## Backup 33 — LSTM tensor shapes through the model

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- BATCH

- (4, 12, 716, 41)

- SENSOR SEQUENCES

- (4 × 716, 12, 41) / = (2,864, 12, 41)

- FORECAST

- (4, 12, 716, 1) / 12 future steps

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The axes are permuted so each sensor in each window is a separate sequence. The batch and sensor axes are combined, the two LSTM layers process each sequence, and the head produces 12 future values. The model restores time and sensor axes.


## Backup 34 — How we count trainable parameters

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- LSTM 256

- Embeddings 13,048 / LSTM layers 832,512 / Head 3,084

- PATCHSTG

- Projection 2,368 / Embeddings 49,120 / 5 blocks 2,231,040 / Head 2,316

- TOTALS

- LSTM 848,644 / PatchSTG 2,284,844

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

A trainable parameter is one learned weight or bias. Count the elements in all trainable tensors and add them. The 256-unit LSTM’s two recurrent layers have 306,176 and 526,336 parameters. Fixed gather and restore indices are buffers, not trainable parameters.


## Backup 35 — PatchSTG: project one sensor history

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- INPUT

- 12 time steps × 3 / values / traffic, slot and / weekday

- PROJECTION

- 36 inputs → 64 / outputs / learned weights and / biases

- ADD CONTEXT

- 64 + 32 + 32 + 64 / = 192 values

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The temporal convolution uses all three channels and 12 historical positions. It creates 64 values per sensor. The model concatenates time-slot and weekday embeddings for the last historical step and a sensor embedding to obtain a 192-value vector.


## Backup 36 — From one sensor to Query, Key and Value

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- SENSOR 5

- Representation / (1, 192)

- LINEAR LAYER

- 192 inputs → 576 / outputs / learned weights and / biases

- SPLIT

- Query (1, 192) / Key (1, 192) / Value (1, 192)

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

A single learned linear layer creates all three vectors together. Each of its 576 output numbers is a weighted sum of the 192 input values plus a bias. Splitting the output gives one query, key and value vector of width 192.


## Backup 37 — One query scores 64 keys

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- SENSOR 5 QUERY

- Shape (1, 192)

- 64 KEYS

- Shape (64, 192) / transpose to (192, 64)

- SCORES

- (1, 192) × (192, 64) / = (1, 64)

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The query at one position is compared with every key in its patch, including its own. Each score is a dot product. With all 64 queries, the scores form a 64 by 64 matrix. Each row is one query’s scores.


## Backup 38 — Scaling and softmax: numerical example

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- DOT PRODUCTS

- Query [1, 2] / Keys [1, 0] and [0, 1] / Scores = 1 and 2

- SCALE

- Divide by √2 / ≈ 0.71 and 1.41

- SOFTMAX

- Weights ≈ 0.33 and / 0.67 / The two weights sum / to 1

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

This is a two-dimensional teaching example. Actual queries and keys have width 192 and are divided by the square root of 192. Softmax is applied across the keys for each query, turning its scores into attention weights.


## Backup 39 — Where the Value vectors are added

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- WEIGHT VALUES

- 0.33 × [3, 0] = [0.99, / 0] / 0.67 × [0, 6] = [0, / 4.02]

- SUM COLUMNS

- [0.99, 0] + [0, 4.02] / = [0.99, 4.02]

- PROJECT SHAPES

- One query: (1, 64) × / (64, 192) / All queries: (64, 64) × / (64, 192)

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

Each query has its own set of weights. For one query, multiply every Value vector by its corresponding weight and add down each feature column. One query gives one 192-value result. The 64 queries therefore give 64 such results. Values are independently projected from input representations, not produced by attention scores.


## Backup 40 — Inside one PatchSTG block

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- DEPTH

- Attention over 64 / positions / within each patch

- RESIDUAL + MLP

- Normalize, attend, add / normalize, feed / forward, add

- BREADTH

- Attention over 16 / patches / at the same position

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

Each attention stage uses normalization, attention with a residual addition, then a feed-forward network with another residual addition. Depth occurs before breadth. A position can therefore carry information gathered from its patch into the breadth stage. The block is repeated five times with separate learned parameters.


## Backup 41 — Restore original sensor order

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- PATCH LAYOUT

- 16 × 64 positions / includes repeated / sensors

- RESTORE MAP

- One saved position / for each original / sensor

- FORECAST

- 716 representations × / 192 / → 12 predictions per / sensor

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The map is constructed before attention and records the designated original position for each sensor. After the fifth block, the model gathers those 716 positions in original sensor order. It does not average all repeated positions. The shared forecast head converts each 192-value representation to 12 normalized predictions.


## Backup 42 — Best checkpoint and latest checkpoint

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- BEST CHECKPOINT

- Lowest validation MAE / used for evaluation

- LATEST CHECKPOINT

- Latest weights + / Adam state / used to resume / training

- PROJECT SELECTION

- 256 units, epoch 37 / validation MAE / 15.0656

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

Best and latest checkpoints answer different needs. If training stops after epoch 40 with the best validation score at epoch 37, resume from the latest state but evaluate the validation-selected weights from epoch 37.


## Backup 43 — Interpreting the error graphs

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- FORECAST DISTANCE

- Each point pools a / single lead / 15 to 180 minutes / ahead

- SENSOR VARIATION

- Histogram of LSTM / sensor MAEs / 696 sensors with valid / targets

- OVERALL MAE

- Pools all eligible / forecast–target / pairs across sensors / and leads

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The forecast-distance plot reveals how accuracy changes with the prediction horizon. The sensor histogram shows variation for the selected LSTM only. Overall pooled MAE weights each included forecast-target pair equally, so it is not generally the simple average of per-sensor MAEs.


## Backup 44 — GPU checks and numerical precision

**Section:** Backup · **Suggested minutes:** 0

**Slide content**

- DEVICE

- Move model and batch / tensors / to compatible devices

- PARITY

- Compare isolated and / batched / inputs and outputs

- TIMING

- Warm up and / synchronize / measure the defined / scope

- Backup explanation · use only when this detail is relevant to a / question.


**Speaker notes**

The contextual probe checks the recurrent inputs exactly and compares outputs within tolerance. Floating point calculations can differ slightly across execution layouts. An early 256-width probe failed and the revised explicit FP32 check passed. A warmed repeated-batch timing omits other full-epoch work.
