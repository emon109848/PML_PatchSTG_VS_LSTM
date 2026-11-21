# Presentation assets

Open `presentation.pptx` at the project root for the 23-slide main talk (about 21 minutes plus questions). `presentation_detailed.pptx` adds 21 optional explanation slides. `presentation.md` contains the slide outline and speaker notes. `presentation.pdf` is a portable source-layout preview.

The figures compare the existing completed experiments; no training, checkpoint loading or new inference was performed to build them.

- `learning_curves_all_models.png`: actual 50-epoch training and validation histories for LSTM 128, LSTM 256 and PatchSTG.
- `validation_zoom_all_models.png`: a labeled zoom of the same validation measurements.
- `forecast_horizon_comparison.png`: all four methods, plus a separate LSTM/PatchSTG zoom.
- `matched_forecast_example.png`: both saved forecasts at the first replay origin and first sensor, selected by position rather than error.
- `sensor_errors_lstm.png`: an explicitly LSTM-only diagnostic; paired PatchSTG per-sensor metrics were unavailable.

`slide_data.json` retains the compact histories and cached example, including original source paths and SHA-256 hashes. Keep this folder in Git so the lab PC does not need the ignored `runs/` directory to reproduce the presentation. The test metrics still come from the project's tracked reports. `appendix_content.json` retains the original detailed lessons for rebuilding without the local backup.

Rebuild from the project root with the project Python environment:

```bash
python scripts/build_presentation.py
```

The builder uses NumPy, Matplotlib, Pillow and standard-library OOXML. Its layout font is DejaVu Sans on Linux; the editable PowerPoint text uses Arial. The PDF and PNG previews are rendered from the same layout definitions, rather than by PowerPoint itself. Check the native deck in PowerPoint on the presentation PC for font substitution.

The original decks and notes are preserved locally under `temp/presentation_before_review_2026-11-21/`. This backup directory is ignored by Git. `build_manifest.json` records the current output hashes and sources.
