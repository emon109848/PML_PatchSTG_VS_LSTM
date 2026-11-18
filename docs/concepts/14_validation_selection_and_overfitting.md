# Validation selection and overfitting

Training data changes parameters; validation guides model choices. Test scores
must not choose hidden width, learning rate, epoch, or the next attempted variant.

Our contextual study predefines widths 128, 256, and 384. Each trains 50 epochs.
Within a candidate, select lowest full-validation MAE, keeping the earlier epoch
on exact ties. Across candidates, select lowest best MAE; exact ties favor the
smaller hidden width. Record all candidates, not only the winner.

## Reading learning curves

Training loss pools evolving-model predictions; validation scores frozen weights.
If training continues improving while validation worsens, this suggests
overfitting or an optimization path that generalizes less well. One higher
validation epoch is not proof; examine the curve and variation.

The completed contextual 128/256 candidates had their best checkpoints before
epoch 50. That is why best and last have distinct roles. A larger model is not
guaranteed better validation performance.

## Diagnostics versus evidence

A shape/gradient check says the computation works. A repeated-batch probe can
show updates help that batch. Neither is a validation result. One seed does not
show statistical significance; one network does not demonstrate scalability.

The contextual redesign followed observed small-LSTM test results. We disclose
this and select subsequent candidates on validation. A final contextual test
pass has not yet occurred at the documented snapshot.

See [results snapshot](../results_and_status.md) and
[capacity experiment rules](../../temp/notes/LSTM_CAPACITY_EXPERIMENT.md).
