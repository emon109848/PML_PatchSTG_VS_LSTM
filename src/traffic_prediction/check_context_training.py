"""Small synthetic CPU checks for contextual training, scoring and resume."""

import copy
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .context_lstm import ContextLSTM, input_calendar
from .train_context_lstm import restore, run_epoch, setup
from .continue_lstm import save_checkpoint


def main():
    setup("cpu")
    torch.manual_seed(2023)
    samples = []
    for i in range(5):
        target = torch.full((12, 3, 1), 10.0 + i)
        target[0, 0] = 0
        samples.append({"x": torch.full((12, 3, 1), i / 10), "y": target,
                        "target_mask": target != 0, "target_start": 12 + i})
    generator = torch.Generator().manual_seed(2023)
    loader = DataLoader(samples, batch_size=2, shuffle=True, generator=generator)
    model = ContextLSTM(10., 2., hidden_size=128, sensors=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    first = run_epoch(model, loader, optimizer)
    assert first["updates"] == 3 and first["windows"] == 5 and first["valid_targets"] == 175
    state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
             "loader_rng_state": generator.get_state(), "torch_rng_state": torch.get_rng_state()}
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "last.pt"
        save_checkpoint(state, path)
        expected = run_epoch(model, loader, optimizer)
        expected_weights = copy.deepcopy(model.state_dict())
        expected_adam = copy.deepcopy(optimizer.state_dict())
        restored = ContextLSTM(10., 2., hidden_size=128, sensors=3)
        adam = torch.optim.Adam(restored.parameters(), lr=.001)
        gen = torch.Generator()
        restore(torch.load(path, weights_only=True), restored, adam, gen, "cpu")
        actual = run_epoch(restored, DataLoader(samples, batch_size=2, shuffle=True, generator=gen), adam)
        assert actual == expected
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
        for index, values in adam.state_dict()["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, expected_adam["state"][index][name], rtol=0, atol=0)
    before = copy.deepcopy(model.state_dict())
    metrics = run_epoch(model, DataLoader(samples, batch_size=2))
    assert metrics["updates"] == 0 and metrics["valid_targets"] == 175
    model.eval()
    with torch.no_grad():
        target = torch.stack([s["y"] for s in samples])
        mask = target != 0
        prediction = model(torch.stack([s["x"] for s in samples]), input_calendar(torch.arange(12, 17)))
        error = (prediction.double() - target.double()).abs()[mask]
        torch.testing.assert_close(torch.tensor(metrics["mae"], dtype=torch.float64),
                                   error.mean(), rtol=1e-5, atol=1e-5)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    zero_sample = dict(samples[0], target_mask=torch.zeros_like(samples[0]["target_mask"]))
    try:
        run_epoch(model, DataLoader([zero_sample]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Empty scoring set accepted.")
    print("Passed: updates, partial batch, masked scoring, frozen validation, empty-mask rejection,")
    print("and exact serialized resume of next-epoch metrics, parameters and Adam state.")


if __name__ == "__main__":
    main()
