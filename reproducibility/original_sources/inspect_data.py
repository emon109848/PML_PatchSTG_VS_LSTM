from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline import (TrafficDataset, make_window, load_traffic, split_boundaries,
                           fit_normalization, make_calendar_keys)
from baselines import persistence_forecast, fit_seasonal_average
from evaluation import masked_mae, evaluate_persistence, evaluate_seasonal


# Resolve the data path relative to this script, wherever Python is run from.
def main():
    """Run the data walkthrough and baseline validation checks."""
    data_path = Path(__file__).resolve().parent / "data" / "raw" / "largest_sd" / "flowsd.npz"

    traffic = load_traffic(data_path)

    print("Shape:", traffic.shape)
    print("Data type:", traffic.dtype)
    print("First sensor's first 12 readings:")
    print(traffic[:12, 0, 0])

    # Split along time, keeping every sensor and feature in each slice.
    # Match the previous project: validation receives any rounding remainder.
    total_steps = traffic.shape[0]
    train_end, test_start = split_boundaries(total_steps)

    # Slice endpoints are excluded. These slices share the original array's memory.
    train_data = traffic[:train_end]
    validation_data = traffic[train_end:test_start]
    test_data = traffic[test_start:]

    print("\nChronological splits (start included, end excluded):")
    print(f"Training:   [0, {train_end}) — shape {train_data.shape}")
    print(f"Validation: [{train_end}, {test_start}) — shape {validation_data.shape}")
    print(f"Test:       [{test_start}, {total_steps}) — shape {test_data.shape}")

    # Learn one shared scale from training observations only, including zeros.
    train_mean, train_std = fit_normalization(train_data)

    # Demonstrate on one sensor's history rather than copying the whole dataset.
    example_history = train_data[:12, 0, 0]
    normalized_history = ((example_history - train_mean) / train_std).astype(np.float32)
    restored_history = normalized_history * train_std + train_mean

    print("\nNormalization (statistics from training data only):")
    print("Training mean:", train_mean)
    print("Training standard deviation:", train_std)
    print("Original history:", example_history)
    print("Normalized history:", normalized_history)
    print("Restored history:", restored_history)
    print("Restored values match originals:", np.allclose(restored_history, example_history))


    history = 12
    horizon = 12
    split_intervals = {
        "Training": (0, train_end),
        "Validation": (train_end, test_start),
        "Test": (test_start, total_steps),
    }

    print("\nSliding windows (12 history steps, 12 future steps, stride 1):")
    for split_name, (split_start, split_end) in split_intervals.items():
        # A start index denotes the first target. All 12 targets stay in this split.
        # At validation/test boundaries, history can come from the preceding split.
        target_starts = range(max(split_start, history), split_end - horizon + 1)
        x, y, target_mask = make_window(traffic, target_starts[0], train_mean, train_std,
                                        history, horizon)
        print(f"{split_name}: {len(target_starts)} windows")
        print(f"  First input: [{target_starts[0] - history}, {target_starts[0]})")
        print(f"  First target: [{target_starts[0]}, {target_starts[0] + horizon})")
        print(f"  Shapes: x={x.shape}, y={y.shape}, mask={target_mask.shape}")

        # Check every eligible start against the split boundaries without allocating
        # all windows. Check first/last samples against the source observations.
        assert all(t >= history and split_start <= t and t + horizon <= split_end
                   for t in target_starts)
        for target_start in (target_starts[0], target_starts[-1]):
            sample_x, sample_y, sample_mask = make_window(
                traffic, target_start, train_mean, train_std, history, horizon)
            assert sample_x.shape == sample_y.shape == sample_mask.shape == (12, 716, 1)
            np.testing.assert_allclose(
                sample_x.astype(np.float64) * train_std + train_mean,
                traffic[target_start - history:target_start], atol=1e-4, rtol=1e-6)
            np.testing.assert_array_equal(sample_y, traffic[target_start:target_start + horizon])
            np.testing.assert_array_equal(sample_mask, sample_y != 0)
            assert not np.shares_memory(sample_x, traffic)
            assert not np.shares_memory(sample_y, traffic)

    first_x, first_y, first_mask = make_window(traffic, history, train_mean, train_std)
    print("\nFirst training window, first sensor:")
    print("Normalized input:", first_x[:, 0, 0])
    print("Target in original units:", first_y[:, 0, 0])
    print("Target mask (True means included in loss/metrics):", first_mask[:, 0, 0])
    print("Window boundary and source-value checks passed.")


    # All datasets share the raw array and the same training-only statistics.
    train_dataset = TrafficDataset(traffic, split_intervals["Training"], train_mean, train_std)
    validation_dataset = TrafficDataset(traffic, split_intervals["Validation"], train_mean, train_std)
    test_dataset = TrafficDataset(traffic, split_intervals["Test"], train_mean, train_std)

    # DataLoader converts NumPy samples to PyTorch tensors and stacks them.
    # Keep this learning example sequential. Training shuffle will be added later.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(train_loader))

    print("\nFirst PyTorch batch:")
    for name, tensor in batch.items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print("First target indices:", batch["target_start"].tolist())
    print("Training batches:", len(train_loader))

    # Verify batching preserves the first sample and includes the final partial batch.
    np.testing.assert_array_equal(batch["x"][0].numpy(), first_x)
    np.testing.assert_array_equal(batch["y"][0].numpy(), first_y)
    np.testing.assert_array_equal(batch["target_mask"][0].numpy(), first_mask)
    assert batch["target_start"].tolist() == [12, 13, 14, 15]
    assert len(train_loader) == (len(train_dataset) + 3) // 4
    last_batch_start = (len(train_loader) - 1) * 4
    last_samples = [train_dataset[i] for i in range(last_batch_start, len(train_dataset))]
    assert len(last_samples) == 1
    assert last_samples[-1]["target_start"] + horizon == train_end
    print("Batch checks passed; the final training batch contains 1 example.")


    prediction = persistence_forecast(batch["x"], train_mean, train_std, horizon)
    mae = masked_mae(prediction, batch["y"], batch["target_mask"])

    print("\nPersistence baseline — first training batch only:")
    print("Prediction shape:", tuple(prediction.shape))
    print("First example, first sensor prediction:", prediction[0, :, 0, 0].numpy())
    print("First example, first sensor target:", batch["y"][0, :, 0, 0].numpy())
    print("Included target count:", batch["target_mask"].sum().item())
    print("Masked MAE:", mae.item())

    # Independently compare predictions with the unnormalized source observations.
    raw_latest = np.stack([traffic[t - 1] for t in batch["target_start"].tolist()])
    expected_prediction = np.repeat(raw_latest[:, None, :, :], horizon, axis=1)
    np.testing.assert_allclose(prediction.numpy(), expected_prediction, atol=1e-4, rtol=1e-6)
    expected_errors = np.abs(expected_prediction - batch["y"].numpy())
    expected_mae = expected_errors[batch["target_mask"].numpy()].mean()
    np.testing.assert_allclose(mae.item(), expected_mae, atol=1e-4, rtol=1e-6)

    # A small hand-check: the zero target is excluded, so MAE = (2 + 4) / 2 = 3.
    toy_prediction = torch.tensor([12.0, 100.0, 24.0])
    toy_target = torch.tensor([10.0, 0.0, 20.0])
    assert masked_mae(toy_prediction, toy_target, toy_target != 0).item() == 3.0
    try:
        masked_mae(toy_prediction, toy_target, torch.zeros(3, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("An empty mask must not produce an accuracy score.")
    print("Persistence and masked-MAE checks passed.")


    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=0)
    validation_metrics, validation_valid_count, validation_windows = evaluate_persistence(
        validation_loader, train_mean, train_std, horizon)
    print("\nPersistence — full validation set:")
    print("Windows evaluated:", validation_windows)
    print("Included target count:", validation_valid_count)
    print("Masked MAE:", validation_metrics["mae"])
    print("Masked RMSE:", validation_metrics["rmse"])
    print("Masked MAPE (%):", validation_metrics["mape_percent"])

    # Independent check directly in original units, without Dataset normalization,
    # DataLoader batching or the forecast function. Process one window at a time.
    raw_error_sum = 0.0
    raw_squared_error_sum = 0.0
    raw_relative_error_sum = 0.0
    raw_valid_count = 0
    for target_start in validation_dataset.target_starts:
        raw_targets = traffic[target_start:target_start + horizon]
        raw_errors = np.abs(raw_targets - traffic[target_start - 1])
        raw_mask = raw_targets != 0
        raw_error_sum += raw_errors[raw_mask].sum()
        raw_squared_error_sum += np.square(raw_errors[raw_mask]).sum()
        raw_relative_error_sum += (raw_errors[raw_mask] / np.abs(raw_targets[raw_mask])).sum()
        raw_valid_count += np.count_nonzero(raw_mask)
    assert validation_windows == len(validation_dataset)
    assert validation_valid_count == raw_valid_count
    raw_metrics = {
        "mae": raw_error_sum / raw_valid_count,
        "rmse": (raw_squared_error_sum / raw_valid_count) ** 0.5,
        "mape_percent": 100 * raw_relative_error_sum / raw_valid_count,
    }
    for metric_name in raw_metrics:
        np.testing.assert_allclose(validation_metrics[metric_name], raw_metrics[metric_name],
                                   atol=1e-4, rtol=1e-6)
    print("Full validation matches the independent raw-data calculation.")


    # Assumed calendar: 2019-01-01 was Tuesday (Monday=0, Tuesday=1).
    # These are reconstructed positions, not timestamps stored in the data file.
    slots_per_day = 96
    calendar_keys = make_calendar_keys(total_steps, slots_per_day)
    seasonal_table = fit_seasonal_average(train_data, calendar_keys[:train_end])

    # Hand-check grouping and both fallback levels on a tiny two-sensor example.
    toy_values = np.array([[[10.0], [0.0]], [[20.0], [0.0]], [[30.0], [0.0]]])
    toy_table = fit_seasonal_average(toy_values, np.array([0, 0, 1]))
    np.testing.assert_array_equal(toy_table[0, :, 0], [15.0, 20.0])
    np.testing.assert_array_equal(toy_table[1, :, 0], [30.0, 20.0])
    np.testing.assert_array_equal(toy_table[2, :, 0], [20.0, 20.0])

    seasonal_metrics, seasonal_valid_count, seasonal_windows = evaluate_seasonal(
        validation_loader, seasonal_table, calendar_keys, horizon)
    assert seasonal_windows == len(validation_dataset)
    assert seasonal_valid_count == validation_valid_count
    assert np.isfinite(seasonal_table).all()
    print("\nSeasonal average — full validation set:")
    print("Learned table shape (weekday/slot groups, sensors, features):", seasonal_table.shape)
    print("First example, first sensor prediction:", seasonal_table[calendar_keys[train_end:train_end + horizon], 0, 0])
    print("Windows evaluated:", seasonal_windows)
    print("Included target count:", seasonal_valid_count)
    print("Masked MAE:", seasonal_metrics["mae"])
    print("Masked RMSE:", seasonal_metrics["rmse"])
    print("Masked MAPE (%):", seasonal_metrics["mape_percent"])
    print("Seasonal grouping, fallback and evaluation-count checks passed.")


if __name__ == "__main__":
    main()
