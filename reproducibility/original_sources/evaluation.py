"""Masked error metrics and baseline evaluation."""

import numpy as np
import torch

from baselines import persistence_forecast


def masked_mae(prediction, target, mask):
    """Average absolute error over included targets only."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask must have the same shape.")
    if mask.dtype != torch.bool:
        raise ValueError("The target mask must contain booleans.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Predictions and targets must be finite.")
    if not mask.any():
        raise ValueError("MAE is undefined when no targets are included.")
    absolute_errors = torch.abs(prediction - target)
    return absolute_errors[mask].mean()



def evaluate_persistence(loader, mean, std, horizon=12):
    """Pool errors across batches before calculating MAE, RMSE and MAPE."""
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    relative_error_sum = 0.0
    valid_target_count = 0
    window_count = 0
    with torch.no_grad():
        for validation_batch in loader:
            forecasts = persistence_forecast(validation_batch["x"], mean, std, horizon)
            targets = validation_batch["y"]
            mask = validation_batch["target_mask"]
            if not torch.isfinite(forecasts).all() or not torch.isfinite(targets).all():
                raise ValueError("Evaluation requires finite predictions and targets.")
            errors = torch.abs(forecasts - targets)
            included_errors = errors[mask]
            included_targets = targets[mask].to(torch.float64)
            absolute_error_sum += included_errors.sum().item()
            squared_error_sum += included_errors.square().sum().item()
            # Select nonzero targets BEFORE division to avoid division by zero.
            relative_error_sum += (included_errors / included_targets.abs()).sum().item()
            valid_target_count += mask.sum().item()
            window_count += targets.shape[0]
    if valid_target_count == 0:
        raise ValueError("Evaluation has no included targets.")
    metrics = {
        "mae": absolute_error_sum / valid_target_count,
        "rmse": (squared_error_sum / valid_target_count) ** 0.5,
        "mape_percent": 100 * relative_error_sum / valid_target_count,
    }
    return metrics, valid_target_count, window_count



def evaluate_seasonal(loader, seasonal_table, calendar_keys, horizon=12):
    """Evaluate calendar-table forecasts in original traffic units."""
    seasonal_error_sum = 0.0
    seasonal_squared_error_sum = 0.0
    seasonal_relative_error_sum = 0.0
    seasonal_valid_count = 0
    seasonal_windows = 0
    with torch.no_grad():
        for validation_batch in loader:
            # Build the 12 future calendar positions for every example in this batch.
            target_indices = validation_batch["target_start"].numpy()[:, None] + np.arange(horizon)
            forecasts = seasonal_table[calendar_keys[target_indices]]
            targets = validation_batch["y"].numpy()
            mask = validation_batch["target_mask"].numpy()
            errors = np.abs(forecasts - targets)[mask]
            seasonal_error_sum += errors.sum()
            seasonal_squared_error_sum += np.square(errors).sum()
            seasonal_relative_error_sum += (errors / np.abs(targets[mask])).sum()
            seasonal_valid_count += np.count_nonzero(mask)
            seasonal_windows += len(target_indices)

    if seasonal_valid_count == 0:
        raise ValueError("Seasonal evaluation has no included targets.")
    return {
        "mae": seasonal_error_sum / seasonal_valid_count,
        "rmse": (seasonal_squared_error_sum / seasonal_valid_count) ** 0.5,
        "mape_percent": 100 * seasonal_relative_error_sum / seasonal_valid_count,
    }, seasonal_valid_count, seasonal_windows
