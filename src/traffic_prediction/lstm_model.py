"""A shared per-sensor LSTM with direct multi-step traffic forecasts."""

import math

import torch
from torch import nn


class TrafficLSTM(nn.Module):
    """Accept normalized histories; return forecasts in original traffic units."""

    def __init__(self, mean, std, hidden_size=32, history=12, horizon=12):
        super().__init__()
        for size in (hidden_size, history, horizon):
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError("Model dimensions must be positive integers.")
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("Normalization requires a finite mean and positive std.")
        self.history = history
        self.horizon = horizon
        # Buffers follow the model's device and are saved with its state, but
        # they are fixed training statistics, not optimizer parameters.
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(float(std), dtype=torch.float32))
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.history or x.shape[3] != 1:
            raise ValueError(f"Expected (batch, {self.history}, sensors, 1) input.")
        if x.shape[0] < 1 or x.shape[2] < 1 or not x.is_floating_point():
            raise ValueError("Input must be a nonempty floating-point batch.")
        batch_size, _, sensors, _ = x.shape

        # Group each sensor's history before combining batch and sensor axes.
        sequences = x.permute(0, 2, 1, 3).reshape(batch_size * sensors, self.history, 1)
        # Omitting initial states makes PyTorch use zeros on every call.
        sequence_output, _ = self.lstm(sequences)
        final_hidden = sequence_output[:, -1, :]
        normalized_forecast = self.output(final_hidden)

        # Restore (batch, horizon, sensors, feature) to match Dataset targets.
        normalized_forecast = normalized_forecast.reshape(batch_size, sensors, self.horizon)
        normalized_forecast = normalized_forecast.permute(0, 2, 1).unsqueeze(-1)
        return normalized_forecast * self.std + self.mean
