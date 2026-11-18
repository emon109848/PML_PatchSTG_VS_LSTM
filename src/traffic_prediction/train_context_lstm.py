"""Train contextual LSTM widths 128/256/384 sequentially; validation-only selection."""

from .paths import PROJECT_ROOT, recorded_source_path

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .context_lstm import ContextLSTM, input_calendar
from .continue_lstm import save_checkpoint, save_report
from .data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from .evaluation import masked_mae
from .train_lstm import sha256


def setup(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing a silent CPU fallback.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"


def run_epoch(model, loader, optimizer=None):
    """Training when optimizer is supplied; otherwise frozen validation."""
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    totals = np.zeros(3, dtype=np.float64)
    count = windows = updates = skipped = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            windows += len(batch["x"])
            mask = batch["target_mask"].to(device)
            if not mask.any():
                skipped += 1
                continue
            x, target = batch["x"].to(device), batch["y"].to(device)
            calendar = input_calendar(batch["target_start"].to(device), model.history)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x, calendar)
            loss = masked_mae(prediction, target, mask)
            if training:
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("Invalid model gradients.")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise RuntimeError("Invalid updated weights.")
                updates += 1
            included = target[mask].detach().double()
            errors = (prediction[mask].detach().double() - included).abs()
            totals += [errors.sum().item(), errors.square().sum().item(),
                       (errors / included.abs()).sum().item()]
            count += included.numel()
            if training and updates % 1000 == 0:
                print(f"    Update {updates}/{len(loader)}; train MAE={totals[0] / count:.4f}", flush=True)
    if not count or windows != len(loader.dataset):
        raise RuntimeError("Incomplete epoch or no included targets.")
    return {"mae": float(totals[0] / count), "rmse": float(np.sqrt(totals[1] / count)),
            "mape_percent": float(100 * totals[2] / count), "valid_targets": count,
            "windows": windows, "updates": updates, "skipped_batches": skipped}


def restore(checkpoint, model, optimizer, generator, device):
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    generator.set_state(checkpoint["loader_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if device == "cuda":
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])


def train_candidate(width, datasets, scaler, output, metadata, resume):
    """Each last checkpoint owns its history and best state for robust resume."""
    device = metadata["device"]
    torch.manual_seed(2023)
    generator = torch.Generator().manual_seed(2023)
    model = ContextLSTM(scaler["mean"], scaler["std"], hidden_size=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(datasets[0], batch_size=4, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(datasets[1], batch_size=4, shuffle=False, num_workers=0)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    starting_epoch = 0
    last_path = output / "last.pt"
    if last_path.exists():
        if not resume:
            raise FileExistsError("Use --resume-suite to continue existing checkpoints.")
        previous = torch.load(last_path, map_location=device, weights_only=True)
        # RNG state tensors must be CPU tensors even when weights live on CUDA.
        for key in ("loader_rng_state", "torch_rng_state", "cuda_rng_state"):
            if key in previous:
                previous[key] = previous[key].cpu()
        if previous["hidden_size"] != width or previous["metadata"] != metadata:
            raise ValueError("Checkpoint configuration differs from this suite.")
        restore(previous, model, optimizer, generator, device)
        history, best, starting_epoch = previous["history"], previous["best"], previous["epoch"]
        del previous
    if best is not None:
        save_checkpoint(best, output / "best.pt")  # Repair derived file after an interrupted save.
    print(f"\nWidth {width}: {sum(p.numel() for p in model.parameters()):,} parameters; "
          f"starting after epoch {starting_epoch}", flush=True)
    for epoch in range(starting_epoch + 1, 51):
        print(f"Width {width}, epoch {epoch}/50", flush=True)
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_metrics = run_epoch(model, val_loader)
        val_seconds = time.perf_counter() - start
        if (train_metrics["windows"] != 21001 or val_metrics["windows"] != 6997
                or train_metrics["valid_targets"] != 177394754
                or val_metrics["valid_targets"] != 58367674):
            raise RuntimeError("Coverage differs from the established SD protocol.")
        history.append({"epoch": epoch, "training": train_metrics, "validation": val_metrics,
                        "training_seconds": train_seconds, "validation_seconds": val_seconds})
        if best is None or val_metrics["mae"] < best["validation"]["mae"]:
            best = {"epoch": epoch, "hidden_size": width,
                    "model_state": copy.deepcopy(model.state_dict()),
                    "validation": val_metrics, "metadata": metadata}
        current = {"epoch": epoch, "hidden_size": width, "model_state": model.state_dict(),
                   "optimizer_state": optimizer.state_dict(), "history": history, "best": best,
                   "metadata": metadata, "torch_rng_state": torch.get_rng_state(),
                   "loader_rng_state": generator.get_state()}
        if device == "cuda":
            current["cuda_rng_state"] = torch.cuda.get_rng_state()
        # Commit authoritative last state first; best/report can be reconstructed.
        save_checkpoint(current, last_path)
        save_checkpoint(best, output / "best.pt")
        save_report({"status": "running", "hidden_size": width, "epochs": history,
                     "best_epoch": best["epoch"], "best_validation": best["validation"]}, output / "report.json")
        print(f"  Train MAE={train_metrics['mae']:.4f}; val MAE={val_metrics['mae']:.4f}; "
              f"best epoch={best['epoch']}; seconds={train_seconds + val_seconds:.1f}", flush=True)
    saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"], strict=True)
    verified = run_epoch(model, val_loader)
    if verified != best["validation"]:
        raise RuntimeError("Saved best failed exact full-validation reproduction.")
    result = {"status": "complete", "hidden_size": width, "epochs": history,
              "best_epoch": best["epoch"], "best_validation": best["validation"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "best_checkpoint": str((output / "best.pt").resolve()),
              "reload_validation_exact": True}
    save_report(result, output / "report.json")
    return {key: value for key, value in result.items() if key != "epochs"}


def main():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/context_lstm_suite_v1")
    parser.add_argument("--resume-suite", action="store_true")
    args = parser.parse_args()
    setup(args.device)
    if args.output_dir.exists() and not args.resume_suite:
        raise FileExistsError("Output exists; use --resume-suite or a new directory.")
    if args.resume_suite and not (args.output_dir / "config.json").exists():
        raise FileNotFoundError("No suite configuration to resume.")
    data_path = root / "data/raw/largest_sd/flowsd.npz"
    traffic = load_traffic(data_path)
    train_end, val_end = split_boundaries(len(traffic))
    mean, std = fit_normalization(traffic[:train_end])
    metadata = {"device": args.device, "seed": 2023, "total_epochs_per_candidate": 50,
                "widths": [128, 256, 384], "layers": 2, "batch_size": 4,
                "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0,
                "dropout": 0, "scheduler": None, "gradient_clip": None,
                "history": 12, "horizon": 12, "shuffle": True,
                "selection": "lowest full validation MAE; earliest epoch, then smallest width on exact ties",
                "precision": "IEEE FP32 matmul and cuDNN RNN, no AMP",
                "scaler": {"mean": float(mean), "std": float(std)},
                "splits": {"train": [0, train_end], "validation": [train_end, val_end]},
                "data_sha256": sha256(data_path),
                "source_sha256": {f"src/traffic_prediction/{name}": sha256(Path(__file__).parent / name) for name in
                                  ("context_lstm.py", "data_pipeline.py", "evaluation.py",
                                   "train_context_lstm.py", "continue_lstm.py", "train_lstm.py", "paths.py")},
                "python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None}
    if args.resume_suite:
        saved_metadata = json.loads((args.output_dir / "config.json").read_text())
        for name, expected in saved_metadata["source_sha256"].items():
            if sha256(recorded_source_path(root, name)) != expected:
                raise ValueError(f"Saved training source identity differs: {name}")
        # Root-level sources were archived intact during submission cleanup.
        # Keep original experiment metadata when resuming those checkpoints.
        expected_settings = {k: v for k, v in saved_metadata.items() if k != "source_sha256"}
        actual_settings = {k: v for k, v in metadata.items() if k != "source_sha256"}
        if expected_settings != actual_settings:
            raise ValueError("Suite source/data/settings/environment changed; resume rejected.")
        metadata = saved_metadata
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        save_report(metadata, args.output_dir / "config.json")
    datasets = [TrafficDataset(traffic, interval, mean, std)
                for interval in ((0, train_end), (train_end, val_end))]
    results = []
    for width in (128, 256, 384):
        result = train_candidate(width, datasets, metadata["scaler"],
                                 args.output_dir / f"hidden_{width}", metadata, args.resume_suite)
        results.append(result)
        save_report({"status": "running", "completed_candidates": results}, args.output_dir / "summary.json")
        if args.device == "cuda":
            torch.cuda.empty_cache()
    winner = min(results, key=lambda item: (item["best_validation"]["mae"], item["hidden_size"]))
    summary = {"status": "complete", "candidates": results,
               "selected_hidden_size": winner["hidden_size"], "selected_epoch": winner["best_epoch"],
               "selected_checkpoint": winner["best_checkpoint"], "test_evaluated": False}
    save_report(summary, args.output_dir / "summary.json")
    print("\nAll three 50-epoch runs complete. Validation-only selection:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
