"""Train one fresh LSTM epoch, validate, and save best/last checkpoints."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from evaluation import masked_mae
from lstm_model import TrafficLSTM


def validate(model, loader):
    """Evaluate frozen weights, pooling errors in float64 across all targets."""
    model.eval()
    absolute_sum = squared_sum = relative_sum = 0.0
    valid_count = windows = 0
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["x"])
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite validation prediction.")
            mask = batch["target_mask"]
            target = batch["y"][mask].to(torch.float64)
            errors = (prediction[mask].to(torch.float64) - target).abs()
            absolute_sum += errors.sum().item()
            squared_sum += errors.square().sum().item()
            relative_sum += (errors / target.abs()).sum().item()
            valid_count += target.numel()
            windows += len(batch["x"])
    if valid_count == 0:
        raise ValueError("Validation has no included targets.")
    return {"mae": absolute_sum / valid_count,
            "rmse": (squared_sum / valid_count) ** 0.5,
            "mape_percent": 100 * relative_sum / valid_count,
            "valid_targets": valid_count, "windows": windows}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "runs/lstm_epoch1_v1")
    args = parser.parse_args()
    # Refuse to overwrite an earlier experiment, including an interrupted one.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {"seed": 2023, "batch_size": 4, "learning_rate": 0.001,
              "epochs": 1, "hidden_size": 32, "history": 12, "horizon": 12,
              "optimizer": "Adam", "device": "cpu", "cpu_threads": 2,
              "loss": "original-unit masked MAE", "shuffle_training": True}
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    data_file = root / "data/raw/largest_sd/flowsd.npz"
    traffic = load_traffic(data_file)
    train_end, test_start = split_boundaries(len(traffic))
    mean, std = fit_normalization(traffic[:train_end])
    train_dataset = TrafficDataset(traffic, (0, train_end), mean, std)
    val_dataset = TrafficDataset(traffic, (train_end, test_start), mean, std)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    model = TrafficLSTM(mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    source_files = ["train_lstm.py", "lstm_model.py", "data_pipeline.py", "evaluation.py"]
    provenance = {
        "config": config, "scaler": {"mean": float(mean), "std": float(std)},
        "splits": {"train": [0, train_end], "validation": [train_end, test_start]},
        "data_sha256": sha256(data_file),
        "source_sha256": {name: sha256(root / name) for name in source_files},
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Fresh one-epoch CPU run:", args.output_dir, flush=True)
    print(f"Training windows: {len(train_dataset)}; validation windows: {len(val_dataset)}", flush=True)
    model.train()
    start = time.perf_counter()
    absolute_sum = 0.0
    target_count = windows = updates = skipped = 0
    for batch in train_loader:
        windows += len(batch["x"])
        mask = batch["target_mask"]
        if not mask.any():
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["x"])
        loss = masked_mae(prediction, batch["y"], mask)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Invalid gradient: {name}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Invalid updated weights.")
        absolute_sum += (prediction[mask].detach().double() - batch["y"][mask].double()).abs().sum().item()
        target_count += mask.sum().item()
        updates += 1
        if updates % 500 == 0:
            print(f"Update {updates}/{len(train_loader)}; pooled training MAE={absolute_sum / target_count:.4f}", flush=True)
    train_seconds = time.perf_counter() - start
    if windows != len(train_dataset) or target_count == 0:
        raise RuntimeError("Incomplete training epoch or no included targets.")
    validation_start = time.perf_counter()
    metrics = validate(model, val_loader)
    validation_seconds = time.perf_counter() - validation_start
    if metrics["windows"] != len(val_dataset) or metrics["valid_targets"] != 58367674:
        raise RuntimeError("Validation coverage differs from the established SD protocol.")

    # With one epoch, this is both the best validation checkpoint and the last.
    checkpoint = {"epoch": 1, "best_epoch": 1, "model_state": model.state_dict(),
                  "optimizer_state": optimizer.state_dict(), "validation": metrics,
                  "provenance": provenance, "torch_rng_state": torch.get_rng_state(),
                  "loader_rng_state": generator.get_state()}
    torch.save(checkpoint, args.output_dir / "best.pt")
    torch.save(checkpoint, args.output_dir / "last.pt")

    # Verify serialized weights reproduce the complete validation score exactly.
    loaded = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    restored = TrafficLSTM(loaded["provenance"]["scaler"]["mean"],
                           loaded["provenance"]["scaler"]["std"])
    restored.load_state_dict(loaded["model_state"], strict=True)
    reloaded_metrics = validate(restored, val_loader)
    if reloaded_metrics != metrics:
        raise RuntimeError("Checkpoint reload did not reproduce validation metrics.")
    report = {"status": "complete", "epoch": 1, "updates": updates,
              "skipped_batches": skipped, "training_windows": windows,
              "training_valid_targets": target_count, "training_mae": absolute_sum / target_count,
              "validation": metrics, "training_seconds": train_seconds,
              "validation_seconds": validation_seconds, "reload_validation_exact": True,
              "provenance": provenance}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "provenance"}, indent=2), flush=True)
    print("Saved best.pt and last.pt. Test set was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
