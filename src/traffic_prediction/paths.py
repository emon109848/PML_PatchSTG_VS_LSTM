"""Resolve the project and preserve pre-cleanup source identities."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def recorded_source_path(root, recorded_name):
    """Locate exact code bytes referenced by historical checkpoint metadata.

    Old checkpoints name root-level scripts. Their immutable snapshots preserve
    those hashes after relocation/import changes. New checkpoints use explicit
    src/ paths. This verifies historical identity, not equality of migrated code;
    migration records and evaluation reports identify current execution sources.
    """
    current = root / recorded_name
    if current.is_file():
        return current
    historical = root / "reproducibility" / "original_sources" / recorded_name
    if historical.is_file():
        return historical
    raise FileNotFoundError(f"Recorded source not found: {recorded_name}")
