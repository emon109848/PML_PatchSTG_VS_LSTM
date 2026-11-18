"""Export matched LSTM/PatchSTG replay; reuse PatchSTG's cached forecasts."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .context_lstm import ContextLSTM, input_calendar
from .data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from .paths import PROJECT_ROOT, recorded_source_path
from .train_context_lstm import setup
from .train_lstm import sha256
from traffic_benchmark.evaluation.replay import render_replay


def load_cached_payload(path, manifest):
    if sha256(path) != manifest["artifact"]["sha256"]:
        raise ValueError("Cached PatchSTG HTML differs from its manifest.")
    match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',
                      path.read_text(encoding="utf-8"), re.S)
    if match is None:
        raise ValueError("Cached PatchSTG replay has no embedded payload.")
    payload = json.loads(match.group(1))
    if payload["history"] != 12 or payload["horizon"] != 12 or len(payload["sensors"]) != 716:
        raise ValueError("Unexpected cached forecast protocol.")
    if payload["provenance"]["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("Cached checkpoint identities disagree.")
    return payload


def canonical_groups(indices, length, batch_size):
    if not indices or indices != sorted(set(indices)) or min(indices) < 0 or max(indices) >= length:
        raise ValueError("Replay indices must be unique chronological test windows.")
    starts = sorted({index // batch_size * batch_size for index in indices})
    return [list(range(start, min(start + batch_size, length))) for start in starts]


def predict_origins(model, dataset, indices, batch_size, device):
    groups = canonical_groups(indices, len(dataset), batch_size)
    selected = set(indices)
    outputs = {}
    model.eval()
    with torch.no_grad():
        for group in groups:
            loader = DataLoader(Subset(dataset, group), batch_size=len(group), shuffle=False, num_workers=0)
            batch = next(iter(loader))
            calendar = input_calendar(batch["target_start"].to(device), model.history)
            forecasts = model(batch["x"].to(device), calendar).cpu().numpy()
            if forecasts.shape != (len(group), 12, 716, 1) or not np.isfinite(forecasts).all():
                raise ValueError("Invalid LSTM replay forecasts.")
            for position, index in enumerate(group):
                if index in selected:
                    outputs[index] = forecasts[position]
    return np.stack([outputs[i] for i in indices]), groups


def comparison_payload(cached, lstm_forecasts, provenance):
    expected = (len(cached["frames"]), 12, len(cached["sensors"]), 1)
    if lstm_forecasts.shape != expected or not np.isfinite(lstm_forecasts).all():
        raise ValueError("LSTM predictions do not match cached origins/sensors/leads.")
    payload = copy.deepcopy(cached)
    patchstg = payload.pop("predictions")
    payload.update(schema=3, title="Contextual LSTM and PatchSTG forecast comparison", provenance=provenance)
    payload["models"] = {
        "patchstg": {"label": "PatchSTG", "predictions": patchstg,
                     "full_test_metrics": provenance["models"]["patchstg"]["full_test_metrics"]},
        "lstm": {"label": "Contextual LSTM · 256 units", "predictions":
                 np.round(lstm_forecasts[..., 0].transpose(0, 2, 1).astype(np.float64), 3).tolist(),
                 "full_test_metrics": provenance["models"]["lstm"]["full_test_metrics"]},
    }
    return payload


def main():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/lstm_patchstg_replay_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Output directory exists; choose a new export path.")
    setup(args.device)
    cache_page = root / "runs/patchstg_replay_v1/index.html"
    cache_manifest = root / "runs/patchstg_replay_v1/manifest.json"
    manifest = json.loads(cache_manifest.read_text())
    cached = load_cached_payload(cache_page, manifest)
    measured_path = root / "runs/final_comparison_v1/report.json"
    measured = json.loads(measured_path.read_text())
    selection = json.loads((root / "runs/final_comparison_v1/selection.json").read_text())["selected"]
    # Use current project paths, rather than requiring the machine's old absolute path.
    checkpoint_path = root / f"runs/context_lstm_suite_v1/hidden_{selection['hidden_size']}/best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (measured["status"] != "complete" or measured["hidden_size"] != 256
            or measured["selected_epoch"] != checkpoint["epoch"]
            or checkpoint["epoch"] != selection["epoch"]
            or measured["checkpoint_sha256"] != checkpoint_hash
            or selection["checkpoint_sha256"] != checkpoint_hash):
        raise ValueError("LSTM checkpoint differs from the completed frozen evaluation.")
    metadata = checkpoint["metadata"]
    for name, expected in metadata["source_sha256"].items():
        if sha256(recorded_source_path(root, name)) != expected:
            raise ValueError(f"Recorded LSTM training source differs: {name}")
    patch_eval_path = root / "runs/patchstg_sd_seed2023_test_v1/report.json"
    patch_eval = json.loads(patch_eval_path.read_text())
    patch_checkpoint = root / "runs/patchstg_sd_seed2023_50ep/best.pt"
    if (sha256(patch_checkpoint) != manifest["checkpoint_sha256"]
            or sha256(patch_eval_path) != manifest["evaluation_report_sha256"]
            or sha256(root / "runs/patchstg_sd_seed2023_50ep/report.json") != manifest["training_report_sha256"]):
        raise ValueError("Cached PatchSTG replay differs from the integrated completed run.")
    if patch_eval["results"]["test"]["overall_pooled"] != manifest["full_test_metrics"]:
        raise ValueError("PatchSTG cache and recorded benchmark metrics differ.")
    data_path = root / "data/raw/largest_sd/flowsd.npz"
    metadata_path = root / "data/raw/largest_sd/sd_meta.csv"
    hashes = {p.name: sha256(p) for p in (data_path, metadata_path)}
    if hashes != manifest["source_sha256"] or hashes["flowsd.npz"] != measured["data_sha256"]:
        raise ValueError("Dashboard data differs from recorded experiments.")
    values = load_traffic(data_path)
    train_end, test_start = split_boundaries(len(values))
    mean, std = fit_normalization(values[:train_end])
    if {"mean": float(mean), "std": float(std)} != metadata["scaler"]:
        raise ValueError("Training scale differs from checkpoint.")
    if manifest["training_identity"]["scaler"] != metadata["scaler"]:
        raise ValueError("LSTM and PatchSTG scales differ.")
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        sensor_rows = list(csv.DictReader(stream))
    if [row["ID"] for row in sensor_rows] != [row["id"] for row in cached["sensors"]]:
        raise ValueError("Cached sensor identity/order differs from metadata.")
    dataset = TrafficDataset(values, (test_start, len(values)), mean, std)
    indices = [frame["window_index"] for frame in cached["frames"]]
    for frame in cached["frames"]:
        if dataset.target_starts[frame["window_index"]] != frame["target_start"]:
            raise ValueError("Forecast origin mismatch.")
    base = cached["frames"][0]["target_start"] - cached["frames"][0]["observation_offset"]
    np.testing.assert_array_equal(np.asarray(cached["observations"]), values[base:base + len(cached["observations"]), :, 0])
    template = root / "demos/lstm_patchstg_replay.html"
    protected = {p: sha256(p) for p in (checkpoint_path, measured_path, cache_page, cache_manifest,
                                       patch_checkpoint, patch_eval_path)}
    model = ContextLSTM(mean, std, hidden_size=256).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print(f"Reusing PatchSTG cache; predicting {len(indices)} matched LSTM origins on {args.device}.", flush=True)
    forecasts, groups = predict_origins(model, dataset, indices, measured["batch_size"], args.device)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected parameter gradients during export.")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "kind": "Matched cached historical forecast replay; no training or new full-test benchmark",
                  "models": {
                      "patchstg": {"epoch": 50, "checkpoint_sha256": manifest["checkpoint_sha256"],
                                   "prediction_source": "Existing PatchSTG replay cache reused exactly; no PatchSTG inference",
                                   "full_test_metrics": manifest["full_test_metrics"]},
                      "lstm": {"epoch": checkpoint["epoch"], "hidden_size": 256,
                               "checkpoint_sha256": checkpoint_hash,
                               "prediction_source": "Frozen contextual LSTM on matching recorded test origins",
                               "full_test_metrics": measured["overall"], "device": args.device,
                               "canonical_test_batches": groups}},
                  "source_sha256": hashes, "lstm_test_report_sha256": sha256(measured_path),
                  "patchstg_cache_sha256": sha256(cache_page),
                  "selection": {"origins": indices, "policy": "Same chronological origins as saved PatchSTG export; no error filtering"},
                  "scaler": metadata["scaler"], "lstm_precision": metadata["precision"],
                  "note": "Forecasts displayed to 3 decimals; local errors use displayed values. Full-test metrics are separately recorded.",
                  "export_source_sha256": sha256(Path(__file__).resolve())}
    payload = comparison_payload(cached, forecasts, provenance)
    if payload["models"]["patchstg"]["predictions"] != cached["predictions"]:
        raise RuntimeError("Cached PatchSTG forecasts changed.")
    page = render_replay(template.read_text(), payload)
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Protected artifact changed: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    export = {**provenance, "frames": len(indices), "sensors": len(sensor_rows),
              "artifact_sha256": sha256(args.output_dir / "index.html"),
              "checkpoints_unchanged": True, "lstm_state_unchanged": True,
              "patchstg_forecasts_reused_exactly": True}
    (args.output_dir / "manifest.json").write_text(json.dumps(export, indent=2) + "\n")
    print("Open:", args.output_dir / "index.html", flush=True)


if __name__ == "__main__":
    main()
