"""Evaluate a validation-selected frozen LSTM once, including each forecast lead."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from continue_lstm import metrics_match, validate_device
from data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from lstm_model import TrafficLSTM
from train_lstm import sha256


def score_test(model, loader, horizon):
    """Pool errors by horizon; do not average batch scores or per-lead RMSEs."""
    sums = np.zeros((3, horizon), dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    windows = 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["x"].to(device)).cpu().double().numpy()
            targets = batch["y"].double().numpy()
            masks = batch["target_mask"].numpy()
            if prediction.shape != targets.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid test predictions.")
            for lead in range(horizon):
                mask = masks[:, lead]
                target = targets[:, lead][mask]
                errors = np.abs(prediction[:, lead][mask] - target)
                sums[:, lead] += [errors.sum(), np.square(errors).sum(),
                                  (errors / np.abs(target)).sum()]
                counts[lead] += len(target)
            windows += len(targets)

    def metrics(total, count):
        if count == 0:
            raise ValueError("No included test targets.")
        return {"mae": float(total[0] / count), "rmse": float(np.sqrt(total[1] / count)),
                "mape_percent": float(100 * total[2] / count), "valid_targets": int(count)}

    per_horizon = [dict(lead=i + 1, assumed_minutes=(i + 1) * 15,
                        **metrics(sums[:, i], counts[i])) for i in range(horizon)]
    return {"overall": metrics(sums.sum(axis=1), counts.sum()),
            "per_horizon": per_horizon, "windows": windows}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/lstm_50epochs_gpu_v1/best.pt")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/lstm_test_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing evaluations are preserved.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Confirm this is the best checkpoint of the completed training run.
    training_report = json.loads((args.checkpoint.parent / "report.json").read_text())
    if (training_report["status"] != "complete"
            or checkpoint["epoch"] != training_report["best_epoch"]
            or checkpoint["validation"] != training_report["best_validation"]):
        raise ValueError("Checkpoint is not the completed run's validation-selected best.")
    provenance = checkpoint["provenance"]
    for name, expected in provenance["source_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Training source changed: {name}")
    data_path = root / "data/raw/largest_sd/flowsd.npz"
    if sha256(data_path) != provenance["data_sha256"]:
        raise ValueError("Dataset differs from training.")
    traffic = load_traffic(data_path)
    train_end, test_start = split_boundaries(len(traffic))
    if provenance["splits"] != {"train": [0, train_end], "validation": [train_end, test_start]}:
        raise ValueError("Split boundaries differ from training.")
    mean, std = fit_normalization(traffic[:train_end])
    if provenance["scaler"] != {"mean": float(mean), "std": float(std)}:
        raise ValueError("Normalization differs from training.")
    config = provenance["config"]
    model = TrafficLSTM(mean, std, config["hidden_size"], config["history"], config["horizon"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(args.device)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    def loader(interval):
        dataset = TrafficDataset(traffic, interval, mean, std, config["history"], config["horizon"])
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    validation = validate_device(model, loader((train_end, test_start)))
    if not metrics_match(validation, checkpoint["validation"]):
        raise ValueError("Evaluation-device validation differs from checkpoint beyond tolerance.")
    print(f"Epoch {checkpoint['epoch']} frozen; validation reproduced on {args.device}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = score_test(model, loader((test_start, len(traffic))), config["horizon"])
    seconds = time.perf_counter() - start
    if result["windows"] != 6997:
        raise ValueError("Incomplete test coverage.")
    # Independently count masks directly from source slices at each lead.
    starts = range(test_start, len(traffic) - config["horizon"] + 1)
    for lead, item in enumerate(result["per_horizon"]):
        expected = np.count_nonzero(traffic[starts.start + lead:starts.stop + lead])
        if expected != item["valid_targets"]:
            raise ValueError("Test target coverage mismatch.")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)
    if any(p.grad is not None for p in model.parameters()) or sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Frozen-state verification failed.")
    result.update(status="complete", selected_epoch=checkpoint["epoch"], device=args.device,
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                  checkpoint_unchanged=True, model_state_unchanged=True,
                  validation_recheck=validation, scoring_seconds=seconds,
                  test_target_interval=[test_start, len(traffic)], batch_size=config["batch_size"],
                  python=platform.python_version(), torch=str(torch.__version__),
                  evaluation_source_sha256=sha256(Path(__file__).resolve()),
                  device_helpers_sha256=sha256(root / "continue_lstm.py"), provenance=provenance)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"epoch": result["selected_epoch"], "windows": result["windows"],
                      "test": result["overall"], "scoring_seconds": seconds}, indent=2), flush=True)
    print("No optimizer, gradients or weight updates. Report:", args.output_dir / "report.json")


if __name__ == "__main__":
    main()
