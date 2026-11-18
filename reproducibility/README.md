# Reproducibility records

- original_sources/: exact pre-cleanup Python source snapshots.
- original_sources.json: original SHA-256 identities.
- path_migration.json: file relocation map.
- reference_project/: local subset of historical baseline/PatchSTG reports and
  the PatchSTG checkpoint referenced by its measured evaluation report.

Preserve these alongside the package for historical checkpoint verification.
Current runtime sources are in src/traffic_prediction and use updated imports/
project-root resolution. A source relocation is not represented as an unchanged
code hash: final evaluation records the current runtime hashes separately.

The reference project subset does not contain a newly trained PatchSTG or the
entire old project. It provides measured report evidence and checkpoint identity,
not a complete fresh PatchSTG reproduction environment.

PatchSTG is now fully integrated in the main project: source under
src/traffic_benchmark, complete saved run under runs/patchstg_sd_seed2023_50ep,
tests/configs/tooling and documentation alongside it. The reference_project
subset remains an integrity archive; current comparison tooling reads the main
project's canonical reports/ and runs/ paths. See patchstg_import_manifest.json,
patchstg_sources.json and reports/patchstg_integration.json for import/integrity
evidence. Its completed scores are reused without rerunning the model.
