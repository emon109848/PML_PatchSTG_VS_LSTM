"""Continue saved LSTM/Adam training on CPU or CUDA."""

from .paths import PROJECT_ROOT, recorded_source_path

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from .evaluation import masked_mae
from .lstm_model import TrafficLSTM
from .train_lstm import sha256, validate


def restore_training(checkpoint, device="cpu"):
    """Restore parameters, Adam moments, and both random-generator states."""
    config = checkpoint["provenance"]["config"]
    scaler = checkpoint["provenance"]["scaler"]
    model = TrafficLSTM(scaler["mean"], scaler["std"], config["hidden_size"],
                        config["history"], config["horizon"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    generator = torch.Generator()
    generator.set_state(checkpoint["loader_rng_state"])
    # Restore AFTER model construction, which initializes random parameters.
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if str(device).startswith("cuda") and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    return model, optimizer, generator


def device_batches(loader, device):
    """Move only model inputs, targets and masks; calendar indices stay on CPU."""
    for batch in loader:
        yield {key: value.to(device) if key in ("x", "y", "target_mask") else value
               for key, value in batch.items()}


def validate_device(model, loader):
    device = next(model.parameters()).device
    return validate(model, device_batches(loader, device))


def metrics_match(actual, expected):
    """Allow small CPU/CUDA rounding differences; coverage must match exactly."""
    return (actual["windows"] == expected["windows"]
            and actual["valid_targets"] == expected["valid_targets"]
            and all(math.isclose(actual[k], expected[k], rel_tol=1e-5, abs_tol=1e-4)
                    for k in ("mae", "rmse", "mape_percent")))


def train_epoch(model, optimizer, loader):
    model.train()
    absolute_sum = 0.0
    count = windows = updates = skipped = 0
    for batch in device_batches(loader, next(model.parameters()).device):
        windows += len(batch["x"])
        mask = batch["target_mask"]
        if not mask.any():
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["x"])
        loss = masked_mae(prediction, batch["y"], mask)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Invalid gradient: {name}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Invalid updated weights.")
        absolute_sum += (prediction[mask].detach().double() - batch["y"][mask].double()).abs().sum().item()
        count += mask.sum().item()
        updates += 1
        if updates % 1000 == 0:
            print(f"  Update {updates}/{len(loader)}; pooled train MAE={absolute_sum / count:.4f}", flush=True)
    if windows != len(loader.dataset) or count == 0:
        raise RuntimeError("Incomplete epoch or no included targets.")
    return {"mae": absolute_sum / count, "windows": windows, "valid_targets": count,
            "updates": updates, "skipped_batches": skipped}


def save_checkpoint(checkpoint, path):
    """Replace only after the new checkpoint has been written successfully."""
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_report(report, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=root / "runs/lstm_epoch1_v1/last.pt")
    parser.add_argument("--total-epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=root / "runs/lstm_50epochs_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify validation and one disposable Adam update; write no outputs.")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this terminal; no CPU fallback was started.")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    if args.total_epochs <= checkpoint["epoch"]:
        raise ValueError("Total epochs must exceed the checkpoint epoch.")
    provenance = checkpoint["provenance"]
    config = provenance["config"]
    if config["optimizer"] != "Adam":
        raise ValueError("This continuation supports Adam checkpoints only.")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    for name, expected in provenance["source_sha256"].items():
        if sha256(recorded_source_path(root, name)) != expected:
            raise ValueError(f"Original source changed: {name}")
    data_path = root / "data/raw/largest_sd/flowsd.npz"
    if sha256(data_path) != provenance["data_sha256"]:
        raise ValueError("Dataset differs from the checkpoint.")
    traffic = load_traffic(data_path)
    train_end, val_end = split_boundaries(len(traffic))
    if provenance["splits"] != {"train": [0, train_end], "validation": [train_end, val_end]}:
        raise ValueError("Split boundaries differ from the checkpoint.")
    mean, std = fit_normalization(traffic[:train_end])
    if {"mean": float(mean), "std": float(std)} != provenance["scaler"]:
        raise ValueError("Training normalization differs from the checkpoint.")
    model, optimizer, generator = restore_training(checkpoint, args.device)
    datasets = [TrafficDataset(traffic, interval, mean, std, config["history"], config["horizon"])
                for interval in ((0, train_end), (train_end, val_end))]
    loader = DataLoader(datasets[0], batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    val_loader = DataLoader(datasets[1], batch_size=config["batch_size"], shuffle=False, num_workers=0)
    # Preserve the previous best even when it predates the last checkpoint.
    best = torch.load(args.resume.parent / "best.pt", map_location="cpu", weights_only=True)
    if (best["epoch"] != checkpoint["best_epoch"] or best["provenance"] != provenance
            or best["validation"]["mae"] > checkpoint["validation"]["mae"]):
        raise ValueError("Previous best checkpoint is inconsistent with the resume checkpoint.")
    # Full frozen validation verifies the checkpoint before any new weight update.
    rng = torch.get_rng_state()
    if not metrics_match(validate_device(model, val_loader), checkpoint["validation"]):
        raise ValueError("Resume checkpoint does not reproduce saved validation metrics.")
    torch.set_rng_state(rng)  # Verification must not advance the training RNG.
    if args.check_only:
        print("Resume checks passed: source/data identity, normalization, Adam state, and full validation.")
        # Exercise one disposable update on the requested device. No artifact
        # writes occur, and restored weights in the source checkpoint are intact.
        probe_loader = DataLoader(datasets[0], batch_size=config["batch_size"], shuffle=False)
        batch = next(device_batches(probe_loader, args.device))
        before = [p.detach().clone() for p in model.parameters()]
        # Validation selected eval mode. cuDNN needs a training-mode forward
        # to retain the RNN intermediates required by backward.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_mae(model(batch["x"]), batch["y"], batch["target_mask"])
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Device probe has invalid gradients.")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Device probe has invalid updated weights.")
        if not any(not torch.equal(a, b) for a, b in zip(before, model.parameters())):
            raise RuntimeError("Device probe did not update weights.")
        print(f"Disposable Adam forward/backward/update passed on {args.device}.")
        print(f"Ready for epochs {checkpoint['epoch'] + 1}–{args.total_epochs}; no epoch run or output writes.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_checkpoint(best, args.output_dir / "best.pt")
    save_checkpoint(checkpoint, args.output_dir / "last.pt")
    run_metadata = {"resume": str(args.resume.resolve()), "resume_sha256": sha256(args.resume),
                    "total_epochs": args.total_epochs,
                    "execution_device": args.device,
                    "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                    "resume_validation_tolerance": {"rtol": 1e-5, "atol": 1e-4},
                    "continuation_source_sha256": sha256(Path(__file__).resolve()),
                    "torch": str(torch.__version__), "provenance": provenance}
    save_report(run_metadata, args.output_dir / "config.json")
    report = {"status": "running", "starting_epoch": checkpoint["epoch"],
              "target_total_epochs": args.total_epochs, "epochs": [],
              "best_epoch": best["epoch"], "best_validation": best["validation"]}
    save_report(report, args.output_dir / "report.json")
    print(f"Resuming epoch {checkpoint['epoch']}; training through epoch {args.total_epochs} on {args.device}.", flush=True)
    for epoch in range(checkpoint["epoch"] + 1, args.total_epochs + 1):
        print(f"Epoch {epoch}/{args.total_epochs}", flush=True)
        start = time.perf_counter()
        training = train_epoch(model, optimizer, loader)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        metrics = validate_device(model, val_loader)
        validation_seconds = time.perf_counter() - start
        if metrics["windows"] != len(datasets[1]) or metrics["valid_targets"] != 58367674:
            raise RuntimeError("Unexpected validation coverage.")
        improved = metrics["mae"] < best["validation"]["mae"]
        current = {"epoch": epoch, "best_epoch": epoch if improved else best["epoch"],
                   "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "validation": metrics, "provenance": provenance,
                   "torch_rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state()}
        current["execution_device"] = args.device
        if args.device == "cuda":
            current["cuda_rng_state"] = torch.cuda.get_rng_state()
        if improved:
            best = copy.deepcopy(current)
            save_checkpoint(best, args.output_dir / "best.pt")
        save_checkpoint(current, args.output_dir / "last.pt")
        report["epochs"].append({"epoch": epoch, "training": training, "validation": metrics,
                                  "training_seconds": training_seconds,
                                  "validation_seconds": validation_seconds})
        report.update(best_epoch=best["epoch"], best_validation=best["validation"])
        save_report(report, args.output_dir / "report.json")
        print(f"  Validation MAE={metrics['mae']:.4f}; best epoch={best['epoch']}", flush=True)
    # Check the saved best model against its recorded full validation score.
    saved_best = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved_best["model_state"])
    reloaded_metrics = validate_device(model, val_loader)
    if not metrics_match(reloaded_metrics, saved_best["validation"]):
        raise RuntimeError("Saved best checkpoint failed validation reproduction.")
    report.update(status="complete", best_reload_validation_verified=True,
                  best_reload_validation_exact=reloaded_metrics == saved_best["validation"])
    save_report(report, args.output_dir / "report.json")
    print(f"Completed {args.total_epochs} total epochs. Best epoch: {best['epoch']}. Test set untouched.", flush=True)


if __name__ == "__main__":
    main()
