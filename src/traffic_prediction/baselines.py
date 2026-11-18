"""Simple traffic forecasting methods."""

import numpy as np
import torch


def persistence_forecast(x, mean, std, horizon=12):
    """Repeat each sensor's latest observation, returning original traffic units."""
    # Keep the time dimension: (batch, 1, sensors, features).
    latest = x[:, -1:, :, :].to(torch.float64) * std + mean
    return latest.repeat(1, horizon, 1, 1)



def fit_seasonal_average(training_values, training_keys, slots_per_day=96):
    """Learn nonzero means for each weekday/time slot and sensor."""
    valid = training_values != 0
    if not valid.any():
        raise ValueError("Seasonal averages need at least one nonzero training value.")
    global_mean = training_values[valid].mean()
    sensor_counts = valid.sum(axis=0)
    sensor_means = np.divide(
        training_values.sum(axis=0), sensor_counts,
        out=np.full(training_values.shape[1:], global_mean), where=sensor_counts > 0)

    # Start with each sensor's mean as fallback for any missing calendar group.
    table = np.broadcast_to(sensor_means, (7 * slots_per_day, *sensor_means.shape)).copy()
    for key in range(7 * slots_per_day):
        group = training_values[training_keys == key]
        counts = (group != 0).sum(axis=0)
        np.divide(group.sum(axis=0), counts, out=table[key], where=counts > 0)
    return table

