# Frozen test evaluation

After all model/epoch choices are fixed using validation, score the selected
checkpoint on the held-out test period without changing parameters. This is
assessment, not a training stage.

Our small LSTM was selected at epoch 49. Its evaluator confirms training source
and data hashes, scaler and splits, completed-run selection metadata, and
validation reproduction before scoring test. Evaluation uses model.eval() and
no gradient tracking. No optimizer is constructed.

The evaluator records aggregate and 12 per-horizon metrics. It checks source
mask counts, unchanged model state and unchanged checkpoint hash. CPU evaluation
followed GPU training, with a validation tolerance guarding small device rounding
differences. The device and tolerance must be disclosed.

The small LSTM test MAE is 39.7079. Contextual models are different architectures
and have a different checkpoint schema: evaluate_lstm.py cannot be used directly
on their best.pt files. A separate explained evaluator is still needed after
contextual candidate selection.

## What to do after seeing test scores

Keep results fixed. Do not use a poor horizon or overall score to choose a new
hidden width, scheduler or checkpoint and call the same test period unseen.
If additional follow-ups are undertaken, disclose the chronology and limitation.

See [small-model evaluator](../implementations/evaluate_lstm.md) and
[results/status](../results_and_status.md).
