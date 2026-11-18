# LSTM versus PatchSTG visual dashboard

Open [the comparison dashboard](../runs/lstm_patchstg_replay_v1/index.html) in a
browser. It is a standalone HTML file with embedded observations and predictions;
no server, internet connection, Python or GPU is needed for playback.

From a WSL terminal, open it with:

```bash
explorer.exe "$(wslpath -w runs/lstm_patchstg_replay_v1/index.html)"
```

Or use Windows Explorer to open `runs/lstm_patchstg_replay_v1/index.html`.

## What you can explore

- Choose any of 716 sensors using its ID/road or by clicking the coordinate map.
- Play/pause 96 chronological test origins or move the origin slider manually.
- Choose a forecast lead from 15 to 180 assumed minutes ahead.
- Overlay both forecasts or show either model alone.
- Reveal/hide actual future and inspect all 12 forecast values/errors in a table.
- Switch the map between models using a fixed shared color scale.
- Distinguish local sensor/window errors from recorded full-test benchmark cards.

Blue is observed history, green is PatchSTG, purple is contextual LSTM, and gold
is actual future. Metadata positions are coordinates, not road connectivity.
Flow does not itself establish congestion or speed. Calendar timestamps remain
reconstructed assumptions, without timezone verification.

## Models and data

The LSTM is selected width 256, epoch 37; PatchSTG is selected epoch 50.
All origins, sensor order, histories and actual future values are aligned.
Origins are the existing PatchSTG replay's test windows 0–95, with assumed issue
times from October 19, 2019 at 23:45 to October 20 at 23:30. Selection is
chronological, not chosen by low error or favorable model performance.

PatchSTG forecasts are reused exactly from its existing saved HTML cache.
PatchSTG was not instantiated, retrained or reevaluated. Only LSTM inference was
performed for these 96 matched windows, using the frozen selected checkpoint.
The exporter preserves original test-batch grouping and uses training-only
normalization with explicit contextual IEEE precision settings.

Both forecast arrays are displayed to three decimals. Local errors use those
displayed values and exclude zero targets. The full-test cards use unchanged
recorded metrics: LSTM MAE 18.1696 and PatchSTG MAE 17.1943. A single sensor/window
can favor either model without contradicting those aggregate scores.

This replay is an illustration, not a new full-test benchmark. It does not
change selection, checkpoints, recorded evaluation scores, or data. Do not tune
models using individual test examples shown in it.

## Files and reproduction

| File | Purpose |
| --- | --- |
| [index.html](../runs/lstm_patchstg_replay_v1/index.html) | Standalone dashboard |
| [manifest.json](../runs/lstm_patchstg_replay_v1/manifest.json) | Source/checkpoint identity, origins and integrity flags |
| [Template](../demos/lstm_patchstg_replay.html) | HTML/CSS/JavaScript interface source |
| [Exporter](../src/traffic_prediction/comparison_dashboard.py) | Cache validation, matched frozen LSTM predictions and safe payload embedding |
| [Command](../scripts/export_comparison_dashboard.py) | Runnable project entry point |
| [UI checks](../tests/test_comparison_dashboard.cjs) | Dependency-free DOM/canvas interaction harness |

The existing dashboard is already generated. For a deliberate new export use a
new directory:

```bash
python scripts/export_comparison_dashboard.py --device cpu --output-dir runs/lstm_patchstg_replay_v2
```

CUDA can also be requested explicitly. Existing outputs are never overwritten.
No dependencies were installed. Template/source, cached HTML identity, measured
report identity, data hashes, sensor order, origin bounds and scale are checked.
The exporter asserts LSTM state/checkpoints are unchanged and PatchSTG values
are preserved exactly.

## Verification

Synthetic and real-export Node checks passed for sensor/lead selection, forecast
and error values, zero masks, reveal/hide, model-line/map selection, horizon
table, timeline, playback boundaries and hidden-tab pause. Real export checks
also verified identical origins/observations/sensors and exact PatchSTG reuse.

```bash
node tests/test_comparison_dashboard.cjs --comparison
node tests/test_comparison_dashboard.cjs runs/lstm_patchstg_replay_v1/index.html
```

A desktop Edge screenshot attempt was blocked by the tool environment's WSL
socket error. Consequently no rendered-browser visual inspection is claimed;
the interaction checks execute the actual JavaScript in a DOM/canvas harness.
The layout retains responsive breakpoints from the integrated replay template.
