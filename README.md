# Traffic forecasting: contextual LSTM versus PatchSTG

Course project using processed SD traffic observations. The selected contextual
LSTM has two layers, 256 hidden units, calendar/sensor embeddings, and 848,644
parameters. It was selected at epoch 37 using validation MAE 15.0656 from the
two completed 50-epoch candidates. The unfinished 384-unit run is deferred.

## Visual comparison dashboard

Open [LSTM versus PatchSTG](runs/lstm_patchstg_replay_v1/index.html) in your browser.
Compare both forecasts with actual traffic, select a sensor/lead, play through
96 matching test origins, and switch the sensor map between models. Everything
is embedded locally. See [dashboard instructions](docs/comparison_dashboard.md).

## Project layout

```text
README.md                 Start here
guide.md                  Step-by-step reading path
src/traffic_prediction/   Data, models, metrics, training and evaluation code
src/traffic_benchmark/    Integrated PatchSTG, spatial maps, trainer and metrics
scripts/                  Runnable commands
configs/                  PatchSTG experiment configurations
tests/patchstg/            Integrated PatchSTG checks
demos/                    Saved replay's HTML template
requirements.txt          Direct Python dependencies
docs/                     Concept guides and implementation walkthroughs
reports/                  Audit, experiment decisions and recorded checks
runs/                     Current candidate checkpoints and generated final results
data/raw/largest_sd/      Local traffic array and metadata
reproducibility/          Original source snapshots and historical reference evidence
temp/                    Archived notes, scripts, logs and superseded experiments
```

Nothing was deleted during cleanup. `temp/` is not part of the main submission
narrative; it keeps earlier educational work recoverable. Keep `reproducibility/`:
existing checkpoints reference original source hashes, and the selected evaluator
checks them against those exact archived bytes. Current execution source hashes
are separately recorded in the new evaluation report.

## Run the selected evaluation and comparison

Both main models have completed test results:

| Model | Test MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Contextual LSTM (256, epoch 37) | 18.1696 | 31.9485 | 12.0646 |
| PatchSTG (epoch 50, saved run) | 17.1943 | 29.1370 | 11.1706 |

PatchSTG's implementation and full saved experiment are integrated. Read
[the PatchSTG guide](docs/patchstg/README.md). No PatchSTG retraining or
reevaluation is needed. To rebuild only report text and figures from saved scores:

```bash
python scripts/build_project_report.py
```

The command below is retained for deliberate reproduction of LSTM evaluation;
the existing completed scores should normally be reused.

Use the existing environment from the project root:

```bash
conda activate pml
python -u scripts/finalize_context_lstm.py --device cuda --output-dir runs/final_comparison_v1
```

The output directory must be new. This checks the selected checkpoint, reproduces
validation, scores frozen test predictions, and writes:

- `PROJECT_REPORT.md`: comparison, interpretation, and limitations.
- `report.json`, `selection.json`, `comparison.json`: machine-readable evidence.
- Learning curves, forecast-lead curves, and a sensor-error distribution.

Final contextual test scoring is complete: MAE **18.1696**, RMSE **31.9485**,
MAPE **12.0646%**. Read [the final report](reports/final_comparison/PROJECT_REPORT.md).
Do not rerun into the existing output directory. PatchSTG is integrated with its source, tests, configurations and saved experiment.
The comparison reuses its existing measured run without retraining or reevaluation.

For another machine, requirements.txt lists direct dependencies; recorded run
reports supply the actual experiment versions. A compatible CUDA-enabled PyTorch
environment is needed for the GPU command. No dependencies were installed or
changed during this cleanup.

## Checks and learning

```bash
python scripts/check_context_training.py
python scripts/inspect_data.py
```

The first is a synthetic training/validation/resume check. The second runs the
data walkthrough and baseline validation; neither starts a long training suite.

For GPU feasibility on a new report path:

```bash
python scripts/check_context_lstm.py --device cuda --output reports/gpu_probe_repeat.json
```

Start learning with [guide.md](guide.md), then use [the documentation index](docs/README.md).
See [cleanup notes](docs/project_structure.md) for moved files and reproduction details.
The original [PatchSTG-only replay](runs/patchstg_replay_v1/index.html) is also preserved.
Attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Submission notes

For GitHub, [the version-controlled final report](reports/final_comparison/PROJECT_REPORT.md)
and its figures are included. Raw data, weights, temp/, and generated runs stay
local. Read [the repository asset guide](docs/github.md) for what is included
and what is needed to reproduce the experiments.

Keep source, scripts, documentation, reports and reproducibility records together.
Raw data and model artifacts are local assets; include them only if required by
the course submission rules. Reproducing evaluation needs the selected candidate
checkpoints/reports and the original data files at the documented paths. Do not
claim a source-only submission can execute inference without those assets.

Exclude `temp/`, Python caches and live logs from the submission archive unless
the instructor requests development history. No archive has been published or
submitted automatically.
