# PatchSTG GPU feasibility evidence

The included completed feasibility report records full SD batches 1, 2 and 4 on
an RTX 5060 Laptop GPU. It checked finite forward outputs/gradients and one
disposable AdamW update per batch. Those probes established feasibility, not
forecast quality or throughput.

| Batch | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: |
| 1 | 168.93 | 198 |
| 2 | 268.40 | 282 |
| 4 | 455.00 | 514 |

[Measured report](../../reports/patchstg/patchstg_gpu_feasibility.json) and the
[original summary](../../reports/patchstg/patchstg_gpu_feasibility.md) are included.
The [feasibility helper](../../src/traffic_benchmark/evaluation/feasibility.py)
handles optimizer probes and batch sweeps. The CLI remains available through
`scripts/patchstg.py gpu-check` for reproducibility, but no new probe or PatchSTG
execution was performed during integration.

The selected accuracy comparison uses the completed full training/test run,
not the disposable feasibility weights. Timing/precision/settings differ from
the contextual LSTM; memory and accuracy tables answer different questions.
