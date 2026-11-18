"""Verify integrated saved PatchSTG artifacts without constructing/running a model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from traffic_benchmark.data import GlobalStandardScaler
from traffic_benchmark.data.spatial import load_spatial_maps
from traffic_benchmark.evaluation.checkpoint import file_sha256, validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/patchstg_integration.json")
    args = parser.parse_args()
    run = ROOT / "runs/patchstg_sd_seed2023_50ep"
    report = json.loads((run / "report.json").read_text())
    test = json.loads((ROOT / "reports/patchstg_sd_seed2023_test_v1.json").read_text())
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    hashes = {name: file_sha256(ROOT / "data/raw/largest_sd" / name)
              for name in ("flowsd.npz", "sd_meta.csv")}
    # validate_checkpoint checks recorded source paths relative to project root.
    import os
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        identity = validate_checkpoint(checkpoint, report, hashes, report["identity"]["splits"],
                                       GlobalStandardScaler(**report["identity"]["scaler"]))
        maps = load_spatial_maps(run / "spatial_maps.npz", identity["spatial"])
    finally:
        os.chdir(previous_cwd)
    if [e["epoch"] for e in report["history"]] != list(range(1, 51)):
        raise ValueError("Incomplete training history.")
    if file_sha256(run / "best.pt") != test["checkpoint"]["sha256"]:
        raise ValueError("Selected checkpoint differs from saved test evaluation.")
    if file_sha256(run / "report.json") != test["training_report_sha256"]:
        raise ValueError("Full training report differs from saved test evaluation.")
    if file_sha256(run / "spatial_maps.npz") != test["spatial_maps_sha256"]:
        raise ValueError("Spatial maps differ from saved test evaluation.")
    if test["source_hashes"] != hashes or test["checkpoint"]["epoch"] != checkpoint["epoch"]:
        raise ValueError("Saved test data/epoch identity differs.")
    for name, expected in (("gather_indices", maps.gather), ("restore_indices", maps.restore)):
        if checkpoint["model"][name].tolist() != expected.tolist():
            raise ValueError("Saved model spatial buffers differ.")
    if any(not torch.isfinite(v).all() for v in checkpoint["model"].values()):
        raise ValueError("Saved model state contains nonfinite values.")
    result = {"status": "verified", "selected_epoch": checkpoint["epoch"], "completed_epochs": 50,
              "spatial_patches": maps.patch_count, "patch_size": maps.patch_size,
              "checkpoint_sha256": file_sha256(run / "best.pt"),
              "training_report_sha256": file_sha256(run / "report.json"),
              "data_sha256": hashes, "test": test["results"]["test"]["overall_pooled"],
              "execution": "hash/metadata/tensor/map checks only; no model construction, forward, training or evaluation",
              "test_metrics_source": "integrated completed saved run; unchanged"}
    if args.output.exists():
        raise FileExistsError("Integration check output exists; choose a new --output path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
