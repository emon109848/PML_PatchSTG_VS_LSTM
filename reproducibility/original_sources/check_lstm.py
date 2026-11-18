"""Check the LSTM on training data without taking an optimizer step."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from evaluation import masked_mae
from lstm_model import TrafficLSTM


def main():
    torch.manual_seed(2023)
    # Keep this small CPU diagnostic predictable on machines with many cores.
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    traffic = load_traffic(root / "data/raw/largest_sd/flowsd.npz")
    train_end, _ = split_boundaries(len(traffic))
    mean, std = fit_normalization(traffic[:train_end])
    dataset = TrafficDataset(traffic, (0, train_end), mean, std)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    model = TrafficLSTM(mean, std)
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}

    prediction = model(batch["x"])
    assert prediction.shape == batch["y"].shape == (4, 12, 716, 1)
    assert torch.isfinite(prediction).all()
    loss = masked_mae(prediction, batch["y"], batch["target_mask"])
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        assert parameter.grad.abs().sum() > 0, f"Zero gradient: {name}"
        torch.testing.assert_close(parameter, initial_parameters[name], rtol=0, atol=0)
    assert model.mean.grad is None and model.std.grad is None

    model.eval()
    with torch.no_grad():
        # Independent one-sensor calls must agree with their batched positions.
        for example, sensor in ((0, 0), (1, 358), (3, 715)):
            isolated = model(batch["x"][example:example + 1, :, sensor:sensor + 1, :])
            torch.testing.assert_close(isolated, prediction[example:example + 1, :, sensor:sensor + 1, :],
                                       rtol=1e-5, atol=1e-4)
        # Changing one sensor must leave all other sensors' forecasts alone.
        changed = batch["x"].clone()
        changed[:, :, 0, :] += 1
        changed_prediction = model(changed)
        torch.testing.assert_close(changed_prediction[:, :, 1:], prediction[:, :, 1:])
        # A repeated call starts with fresh zero states, not retained memory.
        torch.testing.assert_close(model(batch["x"]), prediction)
        # Exercise the final partial-batch size as well.
        assert model(batch["x"][:1]).shape == (1, 12, 716, 1)

    print("Input shape:", tuple(batch["x"].shape))
    print("Prediction shape:", tuple(prediction.shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
    print("Untrained first-batch masked MAE:", loss.item())
    print("Finite nonzero gradients reached every trainable parameter tensor.")
    print("Sensor layout, independence, fresh states and partial-batch checks passed.")
    print("No optimizer step, training run, validation or test evaluation was performed.")


if __name__ == "__main__":
    main()
