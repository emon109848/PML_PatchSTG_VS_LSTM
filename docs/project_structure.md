# Project structure

The [root README](../README.md) is the entry point. Final written results live in
`reports/final_comparison/`; slides and notes live in `presentations/`. Concepts
and code walkthroughs are under `docs/`, and personal lessons/questions are under
`docs/learning/`. Report assets and compact histories are under
`reports/presentation/`, so report/slide refreshes work without ignored run files.

## Latest cleanup

| Previous location | Current location |
| --- | --- |
| Root `presentation*.pptx`, PDF and Markdown | `presentations/` |
| Root `MY_DOUBTS.md` | `docs/learning/questions.md` |
| Root `PROJECT_CONTEXT_FOR_CHATGPT.md` | `docs/learning/project_context.md` |
| Root `lessons/` | `docs/learning/lessons/` |
| Exported Codex session | `temp/session_exports/` |

Links and presentation output paths were updated. `temp/` contains local archives
and is ignored by Git. No datasets, checkpoints, measured results or exact source
snapshots were deleted or moved. Evidence JSON retains original bytes and paths.

## Code and evidence layout

Cleanup groups code and separates current work from development history. It
does not change the LSTM architecture, learned weights, data splits, masks,
normalization or existing scores. Existing checkpoint/report JSON files are
preserved byte-for-byte; their embedded original paths remain historical records.

## Where files moved

| Previous location/type | Current location |
| --- | --- |
| Reusable root Python modules | `src/traffic_prediction/` |
| Main runnable tools | `scripts/` entry points |
| Old root learning/decision notes | `temp/notes/` |
| Original small-LSTM-only diagnostic/evaluation scripts | `temp/legacy_scripts/` |
| Earlier small-model/interrupted run folders | `temp/experiments/` |
| Old terminal logs | `temp/logs/` |
| Superseded CPU feasibility reports | `temp/diagnostic_reports/` |
| Exact original root sources | `reproducibility/original_sources/` |
| Integrated saved baseline/PatchSTG evidence | `reproducibility/reference_project/` |

The current contextual suite stays under runs/context_lstm_suite_v1. Its deferred
384 artifacts are retained in place, so cleanup does not discard or relabel the
incomplete experiment. No source data/checkpoint was deleted.

## Commands and imports

Use `python scripts/NAME.py`, not the former root command. Each small entry point
adds src/ to the Python path and calls its package module's main function.
The package uses relative imports. A shared paths module resolves the project
root from the package location; data/report paths do not accidentally point inside src/.

Do not run the archived scripts directly as current entry points. They are exact
historical copies, not adapted executable tools. Some small-model training modules
remain in the package because other modules reuse their serialization, validation
or hash helpers; they are support code, not the recommended final experiment.

## Source hashes after a layout change

Original training records contain root-level source names and hashes. Relocation
requires import/path updates, so current file bytes differ. The original snapshots
and original_sources.json preserve the measured experiment's code identity.
recorded_source_path resolves old names to those snapshots; new runs record src/
paths. The frozen evaluator verifies the originals and records hashes of current
runtime modules separately. This does not pretend migrated bytes are unchanged.

path_migration.json lists old/new locations. These records support provenance and
recovery; they do not rewrite previously saved experimental evidence.

For an explicitly authorized return to the original three-candidate suite, the
migrated trainer supports existing suite metadata while checking original source
identities and unchanged settings. The current scope still defers 384; cleanup
does not restart it. CPU regression checks and frozen-selection checks are used
to verify migration without launching long training or test evaluation.

## Integrated PatchSTG

PatchSTG code lives under src/traffic_benchmark, with command tooling in scripts/,
configs in configs/, checks in tests/patchstg/, and guides in docs/patchstg/.
Its full completed training/test/replay artifacts are local under runs/.
The saved scores are reused without retraining or reevaluation. Model/preprocessing/
trainer source hashes and saved maps/checkpoints validate against the completed
experiment. A small dispatcher provides imports without changing the original
training-script bytes recorded in the checkpoint.

The comparison now reads canonical reports/ and runs/ artifacts directly.
reproducibility/reference_project retains the earlier bundled subset as an
integrity archive, not as a dependency on an external project.

See [README](../README.md) for submission/runtime asset requirements.
