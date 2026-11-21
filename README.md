# Traffic forecasting: contextual LSTM versus PatchSTG

Predict 12 future traffic-flow readings from 12 historical readings across 716
sensors in processed LargeST-SD. The completed project compares persistence,
seasonal average, contextual LSTM and PatchSTG using saved experiments.

## Start here

| What you need | Open |
| --- | --- |
| Final results, plots and interpretation | [Project report](reports/final_comparison/PROJECT_REPORT.md) |
| Main talk: 23 slides, about 21 minutes | [PowerPoint](presentations/presentation.pptx) · [PDF preview](presentations/presentation.pdf) |
| Explanation slides and speaker notes | [Detailed deck](presentations/presentation_detailed.pptx) · [Notes](presentations/presentation.md) |
| Understand the project step by step | [Reading guide](guide.md) |
| Find a concept or implementation | [Documentation index](docs/README.md) |
| Run a supported command | [Command reference](docs/commands.md) |
| Know which files belong where | [Project structure](docs/project_structure.md) |

## Recorded test results

| Method | MAE ↓ | RMSE ↓ | MAPE (%) ↓ |
| --- | ---: | ---: | ---: |
| Persistence | 60.7825 | 94.3350 | 41.7801 |
| Seasonal average | 28.0045 | 53.3906 | 19.9964 |
| Contextual LSTM 256 | 18.1696 | 31.9485 | 12.0646 |
| PatchSTG | 17.1943 | 29.1370 | 11.1706 |

PatchSTG has 5.37% lower MAE relative to LSTM. LSTM uses 37.1% of its parameter
count. This is a single-seed system comparison with different capacity and
optimization, rather than an isolated spatial-attention experiment. LSTM width
256 was selected at epoch 37 among two completed 50-epoch candidates; width 384
and BigST are deferred. Read the report for protocol and limitations.

## Refresh report and slides

Use the existing project environment from this folder:

```bash
conda activate pml
python scripts/build_project_report.py
python scripts/build_presentation.py
```

Both commands reuse saved measurements. Neither trains, loads checkpoints or
runs model inference. All three 50-epoch histories, including PatchSTG, are
retained in tracked compact plot data. The report and slides share the same
figure builder, including the neural-model horizon zoom and cached forecast.

Model execution is separate; see [commands](docs/commands.md). The dataset stays local and is excluded from GitHub; see [data instructions](data/README.md).
Trained weights remain local assets excluded from Git. Both are needed for model execution,
but not for reading or rebuilding the report and slides. Direct dependencies
are listed in [requirements.txt](requirements.txt); experiment reports record
actual versions. No environment changes are performed by document refreshes.

## Project layout

```text
README.md, guide.md       Entry point and reading route
presentations/            Main/detailed decks, PDF and speaker notes
docs/                     Concepts, implementation guides and learning notes
reports/final_comparison/  Final report, figures and frozen evaluation evidence
reports/presentation/     Shared plot assets and compact saved histories
reports/                  Other saved experiments and data-audit evidence
src/                      LSTM and PatchSTG implementation packages
scripts/                  Runnable tools and document builders
configs/, tests/, demos/  Configurations, checks and replay template
reproducibility/           Exact source snapshots and integrity records
data/                     Local dataset; only instructions/manifest tracked
runs/                     Local checkpoints and generated replay
temp/                     Local archives, previous slides and session exports
```

Keep `reproducibility/`: checkpoint verification relies on its original source
bytes. Saved result files retain their original paths and hashes as evidence.
Historical duplicate records are preserved for that reason.

## Visual dashboard

With local artifacts available, open [the saved comparison replay](runs/lstm_patchstg_replay_v1/index.html).
It compares cached forecasts for 96 matching test origins; it does not perform
live inference. See [dashboard instructions](docs/comparison_dashboard.md).
A fixed matched example is embedded in the tracked report and slides for PCs
without the ignored replay folder.

See [GitHub asset guidance](docs/github.md) for transferring data/weights separately,
and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution.
