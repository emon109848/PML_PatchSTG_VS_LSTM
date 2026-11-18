# Data quality, metadata, and provenance

Good model scores require knowing what the input actually contains. Our files
were copied from the previous project without alteration; hashes matched.

`flowsd.npz` stores traffic; `sd_meta.csv` stores IDs, coordinates, and road fields.
An SHA-256 hash fingerprints file bytes. A matching hash establishes identical
bytes, not that the dataset's labels or provenance are correct.

## Checks and interpretation

The audit found no nonfinite or negative readings, a range of 0–999, and
563,841 zeros (2.2474%). Four consecutive frames, 6536–6539, are zero across
all sensors. No sensor is zero for the entire array, but some are zero throughout
individual splits. Four are zero throughout training.

Metadata has 716 rows, unique ID/ID2 fields, valid coordinates, no duplicate
coordinate pairs, and District 11 entries. These are structural checks.

## What remains assumed

The NPZ has neither timestamps nor sensor IDs. We assume uninterrupted
15-minute samples from January 1, 2019, and metadata-row/traffic-column pairing.
Calendar plausibility does not independently prove timestamps or timezone.
ID2 is a source-network index, not a direct index into the 716-column array.

Zero readings may represent no traffic or missing data; we cannot recover the
original validity flags. We preserve data and disclose the mask convention.
Do not infer a cause for the all-zero interval from an assumed calendar label.

See [audit implementation](../implementations/audit_data.md) and
[observed audit report](../../reports/data_audit_v1/README.md).
