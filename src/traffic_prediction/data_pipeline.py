"""Traffic loading, chronological splits, normalization and windows."""

import numpy as np
from torch.utils.data import Dataset


def load_traffic(path):
    """Read the numerical traffic array without changing the source file."""
    with np.load(path, allow_pickle=False) as archive:
        return archive["data"]


def split_boundaries(total_steps):
    """Use 60% training, 20% test, and the remainder for validation."""
    return int(total_steps * 0.6), total_steps - int(total_steps * 0.2)


def fit_normalization(training_values):
    """Calculate one global mean/std, including training zeros."""
    mean = np.mean(training_values, dtype=np.float64)
    std = np.std(training_values, dtype=np.float64)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Normalization requires a finite mean and positive standard deviation.")
    return mean, std


def make_calendar_keys(total_steps, slots_per_day=96):
    """Assume uninterrupted steps from Tuesday, January 1, 2019 at midnight."""
    time_indices = np.arange(total_steps)
    slots = time_indices % slots_per_day
    weekdays = (1 + time_indices // slots_per_day) % 7
    return weekdays * slots_per_day + slots


def make_window(values, target_start, mean, std, history=12, horizon=12):
    """Build one example; target_start is the first future time-step index."""
    if history < 1 or horizon < 1:
        raise ValueError("History and horizon must be positive.")
    if target_start < history or target_start + horizon > len(values):
        raise ValueError("The window must fit inside the available observations.")

    past = values[target_start - history:target_start]
    future = values[target_start:target_start + horizon]
    x = ((past - mean) / std).astype(np.float32)
    y = future.astype(np.float32, copy=True)
    target_mask = y != 0
    return x, y, target_mask



class TrafficDataset(Dataset):
    """Describe available windows and build a single window when requested."""

    def __init__(self, values, target_interval, mean, std, history=12, horizon=12):
        self.values = values
        self.mean = mean
        self.std = std
        self.history = history
        self.horizon = horizon
        split_start, split_end = target_interval
        if not 0 <= split_start < split_end <= len(values):
            raise ValueError("The target interval must lie inside the data.")
        if history < 1 or horizon < 1:
            raise ValueError("History and horizon must be positive.")
        self.target_starts = range(max(split_start, history), split_end - horizon + 1)

    def __len__(self):
        return len(self.target_starts)

    def __getitem__(self, index):
        target_start = self.target_starts[index]
        x, y, target_mask = make_window(
            self.values, target_start, self.mean, self.std, self.history, self.horizon)
        return {"x": x, "y": y, "target_mask": target_mask,
                "target_start": target_start}

