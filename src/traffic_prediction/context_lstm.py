"""Shared per-sensor LSTM with learned calendar and sensor context."""

import math

import torch
from torch import nn


class ContextLSTM(nn.Module):
    """Normalized traffic + historical calendar in; original-unit forecasts out."""

    def __init__(self, mean, std, hidden_size=128, sensors=716, history=12, horizon=12):
        super().__init__()
        if hidden_size not in (128, 256, 384):
            raise ValueError("Predefined hidden widths are 128, 256 and 384.")
        if any(not isinstance(n, int) or n < 1 for n in (sensors, history, horizon)):
            raise ValueError("Dimensions must be positive integers.")
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("Invalid training normalization.")
        self.history, self.horizon, self.sensors = history, horizon, sensors
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))
        self.slot_embedding = nn.Embedding(96, 16)
        self.weekday_embedding = nn.Embedding(7, 8)
        self.sensor_embedding = nn.Embedding(sensors, 16)
        self.lstm = nn.LSTM(41, hidden_size, num_layers=2, batch_first=True, dropout=0)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, x, calendar, sensor_ids=None):
        """calendar: (B,history,2), columns (15-minute slot, Monday-zero weekday)."""
        if x.ndim != 4 or x.shape[1] != self.history or x.shape[-1] != 1:
            raise ValueError("Expected traffic (batch, history, sensors, 1).")
        batch, steps, sensors, _ = x.shape
        if calendar.shape != (batch, steps, 2) or calendar.dtype != torch.long:
            raise ValueError("Expected int64 calendar (batch, history, 2).")
        if calendar.device != x.device:
            raise ValueError("Calendar and traffic must share a device.")
        if sensor_ids is None:
            if sensors != self.sensors:
                raise ValueError("Full sensor count required unless sensor_ids are supplied.")
            sensor_ids = torch.arange(sensors, device=x.device)
        if sensor_ids.shape != (sensors,) or sensor_ids.dtype != torch.long:
            raise ValueError("sensor_ids must be an int64 vector matching the sensor axis.")
        slot = self.slot_embedding(calendar[..., 0])[:, :, None, :].expand(-1, -1, sensors, -1)
        weekday = self.weekday_embedding(calendar[..., 1])[:, :, None, :].expand(-1, -1, sensors, -1)
        identity = self.sensor_embedding(sensor_ids)[None, None].expand(batch, steps, -1, -1)
        features = torch.cat((x, slot, weekday, identity), dim=-1)
        sequences = features.permute(0, 2, 1, 3).reshape(batch * sensors, steps, 41)
        # Fresh zero hidden/cell states for every independent window.
        sequence_output, _ = self.lstm(sequences)
        forecast = self.output(sequence_output[:, -1])
        forecast = forecast.reshape(batch, sensors, self.horizon).permute(0, 2, 1).unsqueeze(-1)
        return forecast * self.std + self.mean


def input_calendar(target_starts, history=12):
    """Reconstruct ONLY historical calendar from assumed 2019-01-01 midnight."""
    indices = target_starts[:, None] + torch.arange(-history, 0, device=target_starts.device)
    return torch.stack((indices % 96, (1 + indices // 96) % 7), dim=-1)
