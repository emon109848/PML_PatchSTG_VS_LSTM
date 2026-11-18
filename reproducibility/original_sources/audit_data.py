"""Audit local source files and plot traffic patterns without changing the data."""

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import load_traffic, split_boundaries


def audit_values(traffic):
    """Print source quality indicators; zeros are not automatically missing."""
    if traffic.shape != (35040, 716, 1):
        raise ValueError(f"Unexpected traffic shape: {traffic.shape}")
    if not np.issubdtype(traffic.dtype, np.number) or np.iscomplexobj(traffic):
        raise ValueError("Traffic must contain real numeric readings.")
    nonfinite = np.count_nonzero(~np.isfinite(traffic))
    negative = np.count_nonzero(traffic < 0)
    print(f"Shape: {traffic.shape}; dtype: {traffic.dtype}")
    print(f"Nonfinite values: {nonfinite}; negative values: {negative}")
    if nonfinite or negative:
        raise ValueError("Source has invalid readings; inspect before preprocessing.")
    zeros = traffic == 0
    print(f"Range: {traffic.min()} to {traffic.max()}")
    print(f"Zeros: {zeros.sum()} / {traffic.size} ({100 * zeros.mean():.4f}%)")
    zero_frames = np.flatnonzero(zeros.all(axis=(1, 2)))
    zero_sensors = np.flatnonzero(zeros.all(axis=(0, 2)))
    print("Entirely zero frame indices:", zero_frames.tolist())
    print("Entirely zero sensor indices:", zero_sensors.tolist())
    train_end, test_start = split_boundaries(len(traffic))
    for name, start, end in (("Training", 0, train_end),
                             ("Validation", train_end, test_start),
                             ("Test", test_start, len(traffic))):
        subset = zeros[start:end]
        print(f"{name}: {100 * subset.mean():.4f}% zero; "
              f"all-zero sensors: {np.flatnonzero(subset.all(axis=(0, 2))).tolist()}")
    # Coverage means nonzero fraction under our benchmark convention, not a
    # verified rate of source-valid measurements.
    coverage = 1 - zeros.mean(axis=(0, 2))
    print("Per-sensor nonzero coverage min/median/max (%):",
          np.round(100 * np.quantile(coverage, [0, 0.5, 1]), 4).tolist())
    return zero_frames


def audit_metadata(path, sensor_count):
    """Check metadata structure while preserving its original row order."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ID", "ID2", "Lat", "Lng", "District"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Metadata lacks required fields.")
        rows = list(reader)
    if len(rows) != sensor_count:
        raise ValueError("Metadata row count does not match sensor count.")
    if any(any(value is None or not value.strip() for value in row.values()) for row in rows):
        raise ValueError("Metadata contains empty or malformed fields.")
    for field in ("ID", "ID2"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {field} values.")
    coordinates = np.array([(float(row["Lat"]), float(row["Lng"])) for row in rows])
    if (not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:, 0]) > 90).any()
            or (np.abs(coordinates[:, 1]) > 180).any()):
        raise ValueError("Invalid coordinates.")
    if any(int(row["District"]) != 11 for row in rows):
        raise ValueError("Expected District 11 metadata.")
    duplicate_coordinates = len(rows) - len(np.unique(coordinates, axis=0))
    print(f"Metadata rows: {len(rows)}; IDs and ID2 unique; no empty fields; District 11")
    print("Latitude range:", coordinates[:, 0].min(), coordinates[:, 0].max())
    print("Longitude range:", coordinates[:, 1].min(), coordinates[:, 1].max())
    print("Duplicate coordinate pairs:", duplicate_coordinates)
    print("Sensor-column alignment remains an assumed metadata row-order pairing.")
    return rows


def plot_patterns(traffic, metadata, zero_frames, output):
    """Use training-only plots and deterministic sensor choices, avoiding test selection."""
    output.mkdir(parents=True, exist_ok=True)
    filenames = ["sensor_week.png", "daily_pattern.png", "zero_frames.png"]
    if any((output / name).exists() for name in filenames):
        raise FileExistsError("Audit plots already exist; choose a new output directory.")
    train_end, _ = split_boundaries(len(traffic))
    train = traffic[:train_end, :, 0]
    # Fixed positions give examples rather than a claim of representative sampling.
    sensors = [0, traffic.shape[1] // 2, traffic.shape[1] - 1]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, sensor in zip(axes, sensors):
        ax.plot(np.arange(7 * 96) / 96, train[:7 * 96, sensor], linewidth=0.8)
        ax.set_ylabel("Traffic flow")
        ax.set_title(f"Array sensor {sensor}; assumed metadata ID {metadata[sensor]['ID']}")
    axes[-1].set_xlabel("Days since assumed 2019-01-01 00:00 (first training week)")
    fig.tight_layout()
    fig.savefig(output / filenames[0], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for sensor in sensors:
        series = train[:, sensor]
        slot_means = []
        for slot in range(96):
            readings = series[slot::96]
            valid = readings[readings != 0]
            slot_means.append(valid.mean() if len(valid) else np.nan)
        ax.plot(np.arange(96) / 4, slot_means, label=f"Sensor {sensor}")
    ax.set(xlabel="Assumed hour of day", ylabel="Mean nonzero traffic flow",
           title="Training-only daily profiles; all weekdays pooled")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / filenames[1], dpi=140)
    plt.close(fig)

    # Show the first all-zero TRAINING frame and neighboring observations.
    candidates = zero_frames[zero_frames < train_end]
    if len(candidates):
        center = int(candidates[0])
        start, end = max(0, center - 48), min(train_end, center + 49)
        indices = np.arange(start, end)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(indices, (traffic[start:end] != 0).mean(axis=(1, 2)) * 100)
        ax.set(xlabel="Original time-step index", ylabel="Sensors with nonzero readings (%)",
               title="Around the first all-zero training frame; cause unknown")
        ax.set_ylim(-5, 105)
        fig.tight_layout()
        fig.savefig(output / filenames[2], dpi=140)
        plt.close(fig)
        assumed = datetime(2019, 1, 1) + timedelta(minutes=center * 15)
        print(f"First all-zero training frame: {center}; assumed calendar time: {assumed}")
    print("Plots saved to:", output)


def main():
    import argparse

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "data_audit_v1")
    args = parser.parse_args()
    directory = root / "data" / "raw" / "largest_sd"
    for name in ("flowsd.npz", "sd_meta.csv"):
        with (directory / name).open("rb") as stream:
            print(name, "SHA-256:", hashlib.file_digest(stream, "sha256").hexdigest())
    traffic = load_traffic(directory / "flowsd.npz")
    zero_frames = audit_values(traffic)
    metadata = audit_metadata(directory / "sd_meta.csv", traffic.shape[1])
    plot_patterns(traffic, metadata, zero_frames, args.output_dir)


if __name__ == "__main__":
    main()
