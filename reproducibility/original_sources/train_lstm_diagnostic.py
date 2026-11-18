"""Run exactly 20 CPU Adam updates on training windows; save no checkpoint."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline import TrafficDataset, fit_normalization, load_traffic, split_boundaries
from evaluation import masked_mae
from lstm_model import TrafficLSTM


def main():
    seed = 2023
    update_limit = 20
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    traffic = load_traffic(root / "data/raw/largest_sd/flowsd.npz")
    train_end, _ = split_boundaries(len(traffic))
    mean, std = fit_normalization(traffic[:train_end])
    dataset = TrafficDataset(traffic, (0, train_end), mean, std)
    # Shuffle windows, preserving chronological order inside each window.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator,
                        num_workers=0)
    model = TrafficLSTM(mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}

    # A fixed TRAINING batch lets us compare before/after on identical examples.
    # This probe is diagnostic and is not a validation set.
    probe = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    model.eval()
    with torch.no_grad():
        before = masked_mae(model(probe["x"]), probe["y"], probe["target_mask"]).item()

    print(f"CPU diagnostic: seed={seed}, Adam lr=0.001, batch_size=4, updates={update_limit}")
    print("Fixed training-probe MAE before:", before)
    model.train()
    updates = 0
    skipped = 0
    error_sum = 0.0
    target_count = 0
    for batch in loader:
        if not batch["target_mask"].any():
            skipped += 1
            continue  # No included targets means there is no loss to optimize.

        # Gradients accumulate by default, so clear them for every update.
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["x"])
        loss = masked_mae(prediction, batch["y"], batch["target_mask"])
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Missing or nonfinite gradient: {name}")
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"Nonfinite updated parameter: {name}")

        updates += 1
        count = batch["target_mask"].sum().item()
        error_sum += loss.item() * count
        target_count += count
        print(f"Update {updates:02d}/{update_limit}: pre-update batch MAE={loss.item():.4f}")
        if updates == update_limit:
            break

    if updates != update_limit:
        raise RuntimeError(f"Expected {update_limit} updates, completed {updates}.")
    changed = sum(not torch.equal(p.detach(), initial_parameters[name])
                  for name, p in model.named_parameters())
    if changed == 0:
        raise RuntimeError("Optimizer did not change any parameter tensor.")
    model.eval()
    with torch.no_grad():
        after = masked_mae(model(probe["x"]), probe["y"], probe["target_mask"]).item()
    print("Updates completed:", updates, "| All-masked batches skipped:", skipped)
    print("Parameter tensors changed:", changed, "/", len(initial_parameters))
    print("Pooled MAE across diagnostic training batches:", error_sum / target_count)
    print("Fixed training-probe MAE after:", after)
    print("Probe improvement is a training diagnostic, not evidence of generalization.")
    print("No validation/test evaluation or checkpoint saved. Weights are discarded on exit.")


if __name__ == "__main__":
    main()
