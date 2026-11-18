# Implementation: inspect_data.py

Source: [inspect_data.py](../../src/traffic_prediction/inspect_data.py).

This is the executable learning walkthrough. `main()` calls imported preparation,
baseline and evaluation functions, prints examples, and runs inline checks.
The main guard prevents loading or evaluation when another module imports it.

## What happens when run

1. Resolve local data relative to the script and load traffic.
2. Print shape, dtype, first sensor readings, split boundaries and slice shapes.
3. Fit training normalization and demonstrate inverse recovery.
4. Construct/inspect split windows and verify eligibility/source correspondence.
5. Build Datasets and a sequential first training batch for inspection.
6. Demonstrate persistence and masked MAE on that batch.
7. Evaluate persistence on full validation and independently recompute raw scores.
8. Fit seasonal averages and exercise synthetic fallback examples.
9. Evaluate seasonal average on the same validation windows and check counts.

```bash
python scripts/inspect_data.py
```

No neural training, checkpoint saving, or test prediction evaluation occurs.
Despite its name, it now includes preprocessing and baseline demonstrations,
not just file inspection. The separate audit handles source quality/plots.

## Checks to understand

Source and inverse-scale checks guard wrong slices/units. Batch checks guard
collation/layout errors. Raw persistence recomputation bypasses model batching
and normalization. Synthetic examples check zero-mask MAE and seasonal fallback.
Assertions are educational checks; running Python with -O disables assertions.
There is no large formal pytest suite in this new implementation.

See [data pipeline](data_pipeline.md), [baselines](baselines.md), and
[evaluation](evaluation.md).
