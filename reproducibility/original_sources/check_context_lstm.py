"""Probe all three contextual LSTMs; updates are disposable, not full training."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from context_lstm import ContextLSTM, input_calendar
from data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from evaluation import masked_mae


def probe(width, batch, mean, std, device, steps):
    torch.manual_seed(2023)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = ContextLSTM(mean, std, hidden_size=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x, y, mask = [batch[k].to(device) for k in ("x", "y", "target_mask")]
    calendar = input_calendar(batch["target_start"].to(device))
    model.train()
    times = []
    for step in range(steps + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, calendar)
        if prediction.shape != y.shape:
            raise RuntimeError("Prediction shape mismatch.")
        loss = masked_mae(prediction, y, mask)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Missing or nonfinite parameter gradient.")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite updated weights.")
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step:  # Exclude one warm-up update, including first optimizer allocation.
            times.append(elapsed)
    model.eval()
    with torch.no_grad():
        # Inspect the LSTM input directly: this checks axis ordering exactly,
        # independently of batch-dependent numerical kernels.
        captured = []
        handle = model.lstm.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        full = model(x, calendar)
        handle.remove()
        sequences = captured.pop()
        for example, sensor in ((0, 0), (1, 358), (3, 715)):
            sequence = sequences[example * x.shape[2] + sensor]
            torch.testing.assert_close(sequence[:, :1], x[example, :, sensor], rtol=0, atol=0)
            torch.testing.assert_close(sequence[:, 1:17],
                                       model.slot_embedding(calendar[example, :, 0]), rtol=0, atol=0)
            torch.testing.assert_close(sequence[:, 17:25],
                                       model.weekday_embedding(calendar[example, :, 1]), rtol=0, atol=0)
            torch.testing.assert_close(sequence[:, 25:],
                                       model.sensor_embedding.weight[sensor].expand(12, -1), rtol=0, atol=0)
        del sequences
        isolated_max_difference = 0.0
        # Confirm reshaping preserves sensor identity and its chronological input.
        for sensor in (0, 358, 715):
            isolated = model(x[:1, :, sensor:sensor + 1], calendar[:1],
                             torch.tensor([sensor], device=device))
            isolated_max_difference = max(isolated_max_difference,
                (isolated - full[:1, :, sensor:sensor + 1]).abs().max().item())
            torch.testing.assert_close(isolated, full[:1, :, sensor:sensor + 1], rtol=1e-4, atol=1e-3)
        changed = x.clone()
        changed[:, :, 0] += 1
        torch.testing.assert_close(model(changed, calendar)[:, :, 1:], full[:, :, 1:])
    return {"hidden_size": width, "layers": 2,
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch_size": len(x), "prediction_shape": list(full.shape),
            "warmup_updates": 1, "timed_updates": steps,
            "mean_step_seconds": sum(times) / len(times),
            "isolated_max_difference_original_units": isolated_max_difference,
            "exact_lstm_input_layout": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None,
            "finite_gradients_weights_and_sensor_layout": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional new JSON report path.")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("At least one timed step is required.")
    if args.output and args.output.exists():
        raise FileExistsError("Existing reports are not overwritten.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Explicit IEEE FP32 for matmul AND cuDNN RNNs; deterministic algorithms
    # alone do not control TF32 precision or require batched/sliced parity.
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    root = Path(__file__).resolve().parent
    traffic = load_traffic(root / "data/raw/largest_sd/flowsd.npz")
    train_end, _ = split_boundaries(len(traffic))
    mean, std = fit_normalization(traffic[:train_end])
    dataset = TrafficDataset(traffic, (0, train_end), mean, std)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    results = []
    for width in (128, 256, 384):
        result = probe(width, batch, mean, std, args.device, args.steps)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    report = {"device": args.device, "torch": str(torch.__version__), "seed": 2023,
              "precision": "IEEE FP32 matmul and cuDNN RNN; no AMP",
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "results": results, "scope": "Repeated training-batch diagnostic only; weights discarded"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("No checkpoints, full training, validation or test evaluation. Probe weights discarded.")


if __name__ == "__main__":
    main()
