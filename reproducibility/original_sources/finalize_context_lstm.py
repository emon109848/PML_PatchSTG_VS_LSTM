"""Freeze the completed 128/256 search, evaluate the winner, and build the report."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from context_lstm import ContextLSTM, input_calendar
from data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from train_context_lstm import run_epoch, setup
from train_lstm import sha256


def select_completed(suite, root):
    """Use validation only; 384 is deliberately outside the revised search."""
    candidates = []
    suite_metadata = json.loads((suite / "config.json").read_text())
    for width in (128, 256):
        directory = suite / f"hidden_{width}"
        report = json.loads((directory / "report.json").read_text())
        if report["status"] != "complete" or [e["epoch"] for e in report["epochs"]] != list(range(1, 51)):
            raise ValueError(f"Width {width} is not a completed 50-epoch run.")
        best_row = min(report["epochs"], key=lambda e: e["validation"]["mae"])
        if best_row["epoch"] != report["best_epoch"] or best_row["validation"] != report["best_validation"]:
            raise ValueError("Training report selection is inconsistent.")
        path = directory / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["metadata"] != suite_metadata:
            raise ValueError("Candidate metadata differs from the suite protocol.")
        if (checkpoint["hidden_size"] != width or checkpoint["epoch"] != report["best_epoch"]
                or checkpoint["validation"] != report["best_validation"]):
            raise ValueError("Best checkpoint disagrees with training report.")
        for epoch in report["epochs"]:
            if (epoch["training"]["windows"] != 21001
                    or epoch["validation"]["windows"] != 6997
                    or epoch["validation"]["valid_targets"] != 58367674):
                raise ValueError("Unexpected training/validation coverage.")
        for name, expected in checkpoint["metadata"]["source_sha256"].items():
            if sha256(root / name) != expected:
                raise ValueError(f"Original training source changed: {name}")
        candidates.append({"hidden_size": width, "epoch": checkpoint["epoch"],
                           "validation": checkpoint["validation"],
                           "parameters": report["parameters"],
                           "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256(path),
                           "training_report_sha256": sha256(directory / "report.json")})
    winner = min(candidates, key=lambda c: (c["validation"]["mae"], c["hidden_size"]))
    return {"selection_rule": "lowest best validation MAE among completed widths 128/256; smaller width on exact ties",
            "revised_scope": "User deferred width 384 for project completion before contextual test evaluation",
            "excluded_hidden_size": 384, "original_three_candidate_suite_complete": False,
            "candidates": candidates, "selected": winner,
            "frozen_utc": datetime.now(timezone.utc).isoformat()}


def metric_values(sums, count):
    if count == 0:
        return {"mae": None, "rmse": None, "mape_percent": None, "valid_targets": 0}
    return {"mae": float(sums[0] / count), "rmse": float(np.sqrt(sums[1] / count)),
            "mape_percent": float(100 * sums[2] / count), "valid_targets": int(count)}


def score_frozen(model, loader):
    """Score horizons and sensors without training or storing all predictions."""
    horizon = model.horizon
    sensor_count = model.sensors
    lead_sums = np.zeros((3, horizon), dtype=np.float64)
    sensor_sums = np.zeros((3, sensor_count), dtype=np.float64)
    lead_counts = np.zeros(horizon, dtype=np.int64)
    sensor_counts = np.zeros(sensor_count, dtype=np.int64)
    windows = 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            calendar = input_calendar(batch["target_start"].to(device), model.history)
            prediction = model(batch["x"].to(device), calendar).cpu().double().numpy()
            targets = batch["y"].double().numpy()
            mask = batch["target_mask"].numpy()
            if prediction.shape != targets.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid frozen prediction.")
            error = np.abs(prediction - targets)
            absolute = np.where(mask, error, 0)
            squared = np.square(absolute)
            relative = np.divide(absolute, np.abs(targets), out=np.zeros_like(absolute), where=mask)
            for index, values in enumerate((absolute, squared, relative)):
                lead_sums[index] += values.sum(axis=(0, 2, 3))
                sensor_sums[index] += values.sum(axis=(0, 1, 3))
            lead_counts += mask.sum(axis=(0, 2, 3))
            sensor_counts += mask.sum(axis=(0, 1, 3))
            windows += len(targets)
            if windows % 1000 == 0:
                print(f"  Test windows scored: {windows}/{len(loader.dataset)}", flush=True)
    if not lead_counts.sum() or windows != len(loader.dataset):
        raise ValueError("Incomplete or empty test evaluation.")
    return {"windows": windows, "overall": metric_values(lead_sums.sum(axis=1), lead_counts.sum()),
            "per_horizon": [dict(lead=i + 1, assumed_minutes=15 * (i + 1),
                                 **metric_values(lead_sums[:, i], lead_counts[i])) for i in range(horizon)],
            "per_sensor": [dict(sensor_index=i, **metric_values(sensor_sums[:, i], sensor_counts[i]))
                           for i in range(sensor_count)]}


def check_reference(reference, current, data_hash, scaler):
    """Verify the old reported benchmark is on matching data, origins and masks."""
    if reference["source_hashes"]["flowsd.npz"] != data_hash or reference["scaler"] != scaler:
        raise ValueError("PatchSTG data/normalization does not match.")
    cfg = reference["config"]
    if (cfg["history"], cfg["horizon"], cfg["stride"]) != (12, 12, 1):
        raise ValueError("PatchSTG window protocol differs.")
    if cfg["splits"]["test"] != [28032, 35040]:
        raise ValueError("PatchSTG test boundaries differ.")
    recorded = reference["results"]["test"]
    if recorded["windows"] != current["windows"]:
        raise ValueError("PatchSTG test-window count differs.")
    if recorded["overall_pooled"]["valid_targets"] != current["overall"]["valid_targets"]:
        raise ValueError("PatchSTG mask coverage differs.")
    if [h["valid_targets"] for h in recorded["per_horizon"]] != [h["valid_targets"] for h in current["per_horizon"]]:
        raise ValueError("PatchSTG per-horizon mask coverage differs.")


def build_comparison(output, selection, result, previous, root):
    """Use existing reference reports; never claim they were newly evaluated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_width = selection["selected"]["hidden_size"]

    files = {"Persistence": "persistence_sd_v1.json", "Seasonal average": "seasonal_average_sd_v1.json",
             "PatchSTG adaptation (previous run)": "patchstg_sd_seed2023_test_v1.json"}
    refs = {label: json.loads((previous / "reports" / name).read_text()) for label, name in files.items()}
    patch = refs["PatchSTG adaptation (previous run)"]
    check_reference(patch, result, result["data_sha256"], result["scaler"])
    patch_checkpoint = previous / patch["checkpoint"]["path"]
    if sha256(patch_checkpoint) != patch["checkpoint"]["sha256"]:
        raise ValueError("Previous PatchSTG checkpoint differs from its measured report.")
    comparison = []
    for label, reference in refs.items():
        if reference["source_hashes"]["flowsd.npz"] != result["data_sha256"]:
            raise ValueError("Baseline data differs.")
        cfg = reference["config"]
        if (cfg["history"], cfg["horizon"], cfg["stride"]) != (12, 12, 1) or cfg["splits"]["test"] != [28032, 35040]:
            raise ValueError("Baseline test protocol differs.")
        test = reference["results"]["test"]
        if (test["windows"] != result["windows"]
                or [h["valid_targets"] for h in test["per_horizon"]]
                != [h["valid_targets"] for h in result["per_horizon"]]):
            raise ValueError("Reference target coverage differs.")
        comparison.append({"model": label, "test": test["overall_pooled"],
                           "source": str((previous / "reports" / files[label]).resolve()),
                           "source_sha256": sha256(previous / "reports" / files[label]),
                           "evaluation_origin": "historical measured report; not rerun here"})
    comparison.append({"model": f"Contextual LSTM {selected_width}", "test": result["overall"],
                       "evaluation_origin": "new frozen evaluation in this project"})
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for candidate in selection["candidates"]:
        report = json.loads((Path(candidate["checkpoint"]).parent / "report.json").read_text())
        epochs = report["epochs"]
        width = candidate["hidden_size"]
        axes[0].plot([e["epoch"] for e in epochs], [e["training"]["mae"] for e in epochs], label=f"{width} train")
        axes[1].plot([e["epoch"] for e in epochs], [e["validation"]["mae"] for e in epochs], label=f"{width} validation")
        axes[1].scatter([candidate["epoch"]], [candidate["validation"]["mae"]])
    for axis, title in zip(axes, ("Training (weights evolve)", "Validation (fixed weights per epoch)")):
        axis.set(xlabel="Epoch", ylabel="MAE", title=title)
        axis.legend()
    fig.tight_layout(); fig.savefig(output / "learning_curves.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, reference in refs.items():
        h = reference["results"]["test"]["per_horizon"]
        ax.plot(range(15, 181, 15), [item["mae"] for item in h], label=label)
    ax.plot(range(15, 181, 15), [h["mae"] for h in result["per_horizon"]], label=f"Contextual LSTM {selected_width}")
    ax.set(xlabel="Forecast lead (assumed minutes)", ylabel="Test MAE", title="Frozen forecast errors by lead")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "test_horizons.png", dpi=150); plt.close(fig)
    usable = [s["mae"] for s in result["per_sensor"] if s["mae"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(usable, bins=30)
    ax.set(xlabel="Per-sensor test MAE", ylabel="Sensors", title="Selected LSTM: sensor error distribution")
    fig.tight_layout(); fig.savefig(output / "sensor_errors.png", dpi=150); plt.close(fig)
    rows = "\n".join(f"| {row['model']} | {row['test']['mae']:.4f} | {row['test']['rmse']:.4f} | {row['test']['mape_percent']:.4f} |" for row in comparison)
    lead_rows = "\n".join(f"| {h['assumed_minutes']} | {h['mae']:.4f} | {h['rmse']:.4f} | {h['mape_percent']:.4f} |" for h in result["per_horizon"])
    lstm_mae = result["overall"]["mae"]
    patch_mae = patch["results"]["test"]["overall_pooled"]["mae"]
    relative_difference = 100 * (lstm_mae / patch_mae - 1)
    text = f'''# LSTM versus PatchSTG: project results

## Task and experimental decision

Predict the next 12 traffic-flow readings from 12 historical readings on processed
SD (716 sensors; assumed 15-minute cadence). LSTM applies the course method;
PatchSTG is the previous project's spatial-attention adaptation. BigST and the
unfinished width-384 LSTM are deferred by user decision to complete this scope.

The original search planned widths 128/256/384. The user narrowed it to the two
completed 50-epoch candidates before the contextual test pass. Width 256 won by
best validation MAE, selecting epoch {selection['selected']['epoch']}. Width 384 is
not a completed or rejected experiment; its existing artifacts are preserved.
This is a revised two-candidate selection, not the original completed suite.

## Shared evaluation protocol

Chronological train/validation/test boundaries are 21024/28032/35040. Training-only
global normalization includes zeros. Targets remain in original units; zero
targets are excluded, without imputation or prediction clipping. Score all 6,997
test windows and {result['overall']['valid_targets']:,} included forecast-target
pairs. Reference reports match data hashes, window protocol, and lead-wise counts.

## Test accuracy

| Model | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
{rows}

Baseline/PatchSTG values are previous measured reports, not new executions.
The contextual result is a new frozen evaluation on {result['device']}. The
LSTM MAE differs by {relative_difference:+.2f}% relative to PatchSTG; negative means lower.
This is a single-seed observed difference, not statistical significance.

## Learning and capacity

![Learning curves](learning_curves.png)

The 128-unit candidate selected epoch 39 (validation MAE 15.3385); width 256
selected epoch 37 (15.0656). Their final-epoch errors were higher than these
best values, supporting validation checkpoint selection rather than automatically
using epoch 50. Training scores pool changing weights, while validation scores
use each completed epoch's frozen model. The plots should be read accordingly.

Selected contextual LSTM has {selection['selected']['parameters']:,} parameters
versus PatchSTG's 2,284,844. Relative to
the small LSTM, context, layers and width changed together. Improved accuracy
cannot identify the separate contribution of each change without ablations.

## Forecast distance

![Test MAE by forecast lead](test_horizons.png)

| Assumed minutes ahead | LSTM MAE | LSTM RMSE | LSTM MAPE (%) |
| --- | ---: | ---: | ---: |
{lead_rows}

Read the plotted curves to distinguish near-term behavior from long-lead errors.
A pooled score can conceal such differences. Longer-lead error is consistent
with less immediate observational guidance, but this experiment alone does not
establish its cause or prove spatial attention is responsible for a difference.

## Sensor variation and data coverage

![Selected LSTM sensor errors](sensor_errors.png)

{len(usable)} sensors have included test targets; {716-len(usable)} have none and
are explicitly assigned null metrics in report.json. Error variation among
scored sensors motivates sensor-level interpretation, but the distribution is
for LSTM only: reference reports do not provide matching per-sensor forecasts.
Do not claim a spatial explanation from this histogram alone. Zeros may indicate
missingness or true zero flow; masking is a benchmark convention.

## Fairness, interpretation, and limitations

Both contextual LSTM and PatchSTG use traffic, calendar and learned sensor identity.
The LSTM processes sensors independently. PatchSTG additionally uses coordinates,
training-derived similarity, and spatial attention. PatchSTG uses AdamW with
different learning rate, schedule, decay and clipping; LSTM uses fixed-lr Adam.
Matching 50 epochs aligns dataset passes, not computation or tuning budget.
The LSTM search used two completed widths; report that extra selection effort.

The previous PatchSTG result is an adapted implementation, not an official
full-paper reproduction. Only one seed and one network were studied. There is
no significance, scalability, or isolated spatial-attention causal claim.
Previous-project and small-LSTM test results were already observed before this
follow-up design; the test set is reused rather than newly unseen. No test score
selected this contextual checkpoint and no further tuning follows this test pass.

Calendar timestamps and metadata/sensor pairing remain assumptions. Timing is
recorded in run reports but is not a controlled comparison of inference kernels.
Historical PatchSTG reports were validated structurally; this run does not
reproduce its forward predictions or independently re-establish its source data.

## Reproduction and artifacts

- selection.json: frozen validation-only choice and explicit scope revision.
- report.json: frozen LSTM test metrics, horizons, sensors, identity and checks.
- comparison.json: reference report paths/hashes and overall metrics.
- Figures above: generated from stored learning histories and measured scores.

Run from the project root with a new output directory:

```bash
python finalize_context_lstm.py --device cuda --output-dir runs/final_comparison_v1
```

The script refuses existing outputs, creates no optimizer, uses no-grad scoring,
and checks model tensors and checkpoint bytes remain unchanged. Existing training
code, reports and checkpoints are not modified. Final test results are fixed.
'''
    (output / "PROJECT_REPORT.md").write_text(text)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "runs/context_lstm_suite_v1")
    parser.add_argument("--previous-project", type=Path, default=root.parent / "Practical ML Project")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/final_comparison_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze selection and check identity; no forward passes or test scoring.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new directory; previous results are preserved.")
    # Preparation can run on CPU without requiring a GPU in the agent environment.
    setup("cpu" if args.prepare_only else args.device)
    selection = select_completed(args.suite, root)
    checkpoint = torch.load(selection["selected"]["checkpoint"], map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    data_path = root / "data/raw/largest_sd/flowsd.npz"
    if sha256(data_path) != metadata["data_sha256"]:
        raise ValueError("Data hash differs from training.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Frozen selection: hidden {checkpoint['hidden_size']}, epoch {checkpoint['epoch']}; width 384 deferred.", flush=True)
    if args.prepare_only:
        print("Preparation complete; no validation or test predictions evaluated.")
        return
    traffic = load_traffic(data_path)
    train_end, test_start = split_boundaries(len(traffic))
    mean, std = fit_normalization(traffic[:train_end])
    if metadata["scaler"] != {"mean": float(mean), "std": float(std)}:
        raise ValueError("Training normalization differs.")
    if metadata["splits"] != {"train": [0, train_end], "validation": [train_end, test_start]}:
        raise ValueError("Split boundaries differ.")
    model = ContextLSTM(mean, std, hidden_size=checkpoint["hidden_size"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    val_loader = DataLoader(TrafficDataset(traffic, (train_end, test_start), mean, std), batch_size=4, shuffle=False)
    print("Checking complete frozen validation...", flush=True)
    validation = run_epoch(model, val_loader)
    for metric in ("mae", "rmse", "mape_percent"):
        if not math.isclose(validation[metric], checkpoint["validation"][metric], rel_tol=1e-5, abs_tol=1e-4):
            raise ValueError("Validation reproduction failed; test not evaluated.")
    if validation["valid_targets"] != 58367674 or validation["windows"] != 6997:
        raise ValueError("Validation coverage differs.")
    print("Validation reproduced. Evaluating frozen test set...", flush=True)
    start = time.perf_counter()
    test_loader = DataLoader(TrafficDataset(traffic, (test_start, len(traffic)), mean, std), batch_size=4, shuffle=False)
    result = score_frozen(model, test_loader)
    result["scoring_seconds"] = time.perf_counter() - start
    if result["windows"] != 6997 or result["overall"]["valid_targets"] != 58158396:
        raise ValueError("Incomplete test coverage.")
    starts = range(test_start, len(traffic) - model.horizon + 1)
    for lead, metrics in enumerate(result["per_horizon"]):
        if np.count_nonzero(traffic[starts.start + lead:starts.stop + lead]) != metrics["valid_targets"]:
            raise ValueError("Per-lead mask count mismatch.")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected parameter gradients.")
    path = Path(selection["selected"]["checkpoint"])
    if sha256(path) != selection["selected"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint bytes changed.")
    result.update(status="complete", hidden_size=checkpoint["hidden_size"], selected_epoch=checkpoint["epoch"],
                  device=args.device, scaler=metadata["scaler"], data_sha256=metadata["data_sha256"],
                  checkpoint_sha256=sha256(path), model_state_unchanged=True, checkpoint_unchanged=True,
                  evaluation_source_sha256=sha256(Path(__file__).resolve()), validation_recheck=validation,
                  torch=str(torch.__version__), numpy=np.__version__, test_target_interval=[test_start, len(traffic)],
                  batch_size=4, precision=metadata["precision"], training_metadata=metadata)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2), flush=True)
    build_comparison(args.output_dir, selection, result, args.previous_project, root)
    print("Complete. Read:", args.output_dir / "PROJECT_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
