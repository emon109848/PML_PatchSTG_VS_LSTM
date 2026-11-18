# Implementation: check_context_training.py

Source: [check_context_training.py](../../src/traffic_prediction/check_context_training.py).

This small synthetic CPU diagnostic exercises actual contextual trainer functions
without touching real experiment checkpoints or starting the candidate suite.
It uses hidden width 128, three synthetic sensors, five samples and batch size two.

## Checks

1. One training epoch has three updates, a final partial batch, and 175 valid
   targets after masking five zero positions.
2. Save model, optimizer, shuffle generator and CPU RNG state using the real
   atomic checkpoint helper into a temporary directory.
3. Run a second epoch uninterrupted and keep resulting metrics/weights/Adam state.
4. Reconstruct/restore the saved first epoch and rerun the second with restored
   shuffle/RNG state. Metrics, weights and Adam state match exactly.
5. Frozen validation makes no updates and preserves model state; its MAE agrees
   with a direct calculation within numerical tolerance.
6. An all-empty scoring set raises an error.

```bash
python scripts/check_context_training.py
```

The checks passed locally. Temporary files are removed automatically. This
covers CPU training/resume semantics; GPU feasibility is supported separately
by the user's host probe. It is not a complete test suite for every filesystem,
GPU failure, interruption instant, or malformed checkpoint.

See [training implementation](train_context_lstm.md) and
[checkpoint concepts](../concepts/13_checkpoints_and_resume.md).
