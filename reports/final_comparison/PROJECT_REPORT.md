# LSTM versus PatchSTG: project results

## Task and experimental decision

Predict the next 12 traffic-flow readings from 12 historical readings on processed
SD (716 sensors; assumed 15-minute cadence). LSTM applies the course method;
PatchSTG is the project's integrated spatial-attention model. BigST and the
unfinished width-384 LSTM are deferred by user decision to complete this scope.

The original search planned widths 128/256/384. The user narrowed it to the two
completed 50-epoch candidates before the contextual test pass. Width 256 won by
best validation MAE, selecting epoch 37. Width 384 is
not a completed or rejected experiment; its existing artifacts are preserved.
This is a revised two-candidate selection, not the original completed suite.

## Shared evaluation protocol

Chronological train/validation/test boundaries are 21024/28032/35040. Training-only
global normalization includes zeros. Targets remain in original units; zero
targets are excluded, without imputation or prediction clipping. Score all 6,997
test windows and 58,158,396 included forecast-target
pairs. Reference reports match data hashes, window protocol, and lead-wise counts.

## Test accuracy

| Model | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Persistence | 60.7825 | 94.3350 | 41.7801 |
| Seasonal average | 28.0045 | 53.3906 | 19.9964 |
| PatchSTG | 17.1943 | 29.1370 | 11.1706 |
| Contextual LSTM 256 | 18.1696 | 31.9485 | 12.0646 |

Baseline/PatchSTG values reuse the integrated completed runs and saved reports;
PatchSTG was not retrained or reevaluated for this comparison.
The contextual result is a new frozen evaluation on cuda. The
LSTM MAE differs by +5.67% relative to PatchSTG; negative means lower.
This is a single-seed observed difference, not statistical significance.

## Learning and capacity

![Learning curves](learning_curves.png)

The 128-unit candidate selected epoch 39 (validation MAE 15.3385); width 256
selected epoch 37 (15.0656). Their final-epoch errors were higher than these
best values, supporting validation checkpoint selection rather than automatically
using epoch 50. Training scores pool changing weights, while validation scores
use each completed epoch's frozen model. The plots should be read accordingly.

Selected contextual LSTM has 848,644 parameters
versus PatchSTG's 2,284,844. Relative to
the small LSTM, context, layers and width changed together. Improved accuracy
cannot identify the separate contribution of each change without ablations.

## Forecast distance

![Test MAE by forecast lead](test_horizons.png)

| Assumed minutes ahead | LSTM MAE | LSTM RMSE | LSTM MAPE (%) |
| --- | ---: | ---: | ---: |
| 15 | 12.6201 | 20.3849 | 8.3125 |
| 30 | 14.2680 | 23.8801 | 9.1331 |
| 45 | 15.5069 | 26.3843 | 10.0173 |
| 60 | 16.5213 | 28.2402 | 10.8277 |
| 75 | 17.4568 | 30.0105 | 11.4375 |
| 90 | 18.2672 | 31.4652 | 12.0772 |
| 105 | 18.9844 | 32.8310 | 12.4731 |
| 120 | 19.6602 | 34.1987 | 13.0413 |
| 135 | 20.2702 | 35.4819 | 13.5678 |
| 150 | 20.8472 | 36.7519 | 13.9392 |
| 165 | 21.5214 | 38.1973 | 14.6868 |
| 180 | 22.1115 | 39.5253 | 15.2619 |

Read the plotted curves to distinguish near-term behavior from long-lead errors.
A pooled score can conceal such differences. Longer-lead error is consistent
with less immediate observational guidance, but this experiment alone does not
establish its cause or prove spatial attention is responsible for a difference.

## Sensor variation and data coverage

![Selected LSTM sensor errors](sensor_errors.png)

696 sensors have included test targets; 20 have none and
are explicitly assigned null metrics in report.json. Error variation among
scored sensors motivates sensor-level interpretation, but the distribution is
for LSTM only: reference reports do not provide matching per-sensor forecasts.
Do not claim a spatial explanation from this histogram alone. Zeros may indicate
missingness or true zero flow; masking is a benchmark convention.

## Fairness, interpretation, and limitations

Both contextual LSTM and PatchSTG use traffic, calendar and learned sensor identity.
The LSTM processes sensors independently. PatchSTG additionally uses coordinates,
training-derived similarity, and spatial attention. PatchSTG uses AdamW with
different learning rate, schedule, decay and clipping; LSTM uses fixed-lr Adam.
Matching 50 epochs aligns dataset passes, not computation or tuning budget.
The LSTM search used two completed widths; report that extra selection effort.

The integrated PatchSTG is an adapted implementation, not an official
full-paper reproduction. Only one seed and one network were studied. There is
no significance, scalability, or isolated spatial-attention causal claim.
PatchSTG and small-LSTM test results were already observed before this
follow-up design; the test set is reused rather than newly unseen. No test score
selected this contextual checkpoint and no further tuning follows this test pass.

Calendar timestamps and metadata/sensor pairing remain assumptions. Timing is
recorded in run reports but is not a controlled comparison of inference kernels.
PatchSTG's completed training report, selected checkpoint, spatial maps and
source code are integrated and integrity-checked. Comparison reuses its saved
test metrics without executing PatchSTG inference again.

## Reproduction and artifacts

- selection.json: frozen validation-only choice and explicit scope revision.
- report.json: frozen LSTM test metrics, horizons, sensors, identity and checks.
- comparison.json: reference report paths/hashes and overall metrics.
- Figures above: generated from stored learning histories and measured scores.

Run from the project root with a new output directory:

```bash
python scripts/finalize_context_lstm.py --device cuda --output-dir runs/final_comparison_v1
```

The script refuses existing outputs, creates no optimizer, uses no-grad scoring,
and checks model tensors and checkpoint bytes remain unchanged. Existing training
code, reports and checkpoints are not modified. Final test results are fixed.
