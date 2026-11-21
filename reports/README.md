# Reports and evidence

## Current submission

Start with [the final project report](final_comparison/PROJECT_REPORT.md). It
includes all four methods, the three completed learning curves, validation
selection, both neural horizon curves, the parameter trade-off, a matched
forecast example and limitations.

| Folder / file | Purpose |
| --- | --- |
| [final_comparison/](final_comparison/PROJECT_REPORT.md) | Canonical written report, five plots and frozen result JSON |
| [presentation/](presentation/README.md) | Shared figures, compact histories and presentation build metadata |
| [data_audit_v1/](data_audit_v1/README.md) | Data inspection findings and figures |
| [patchstg/](patchstg/patchstg_sd_seed2023_test_v1.md) | Detailed PatchSTG checks and saved experiment records |
| [results summary](../docs/results_and_status.md) | Short validation/test status |

## Saved evidence

Root-level baseline/PatchSTG JSON files are canonical inputs used by comparison
and integrity tools. The `patchstg/` folder and `reproducibility/` also retain
historical copies. They are deliberately preserved to avoid changing experiment
identity and source hashes. The small-LSTM report and diagnostic probes describe
earlier work; they are not the final selected model.

To refresh the narrative and figures without training or inference:

```bash
python scripts/build_project_report.py
```

The default destination is `reports/final_comparison/`, using tracked saved
measurements. It leaves `report.json`, `selection.json`, `comparison.json` and
reference metric JSON unchanged. The report and slides share a plot builder;
rebuild the slides separately if their embedded images also need refreshing.
