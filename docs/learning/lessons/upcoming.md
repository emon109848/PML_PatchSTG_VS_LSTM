# Lessons 11–14

## 11. Contextual LSTM and embeddings

Follow one sensor at one time step from its raw traffic value to its 41-value input. Trace how `input_calendar` builds time-of-day and weekday indices from the historical rows, and how embeddings turn those indices and the sensor ID into learned vectors. Follow the two LSTM layers and the forecast head. Compare the small and contextual models to see which changes were introduced together.

**Check your understanding.** What information does sensor ID supply? Where does the 41-value input go? Which index is the first future target? [Contextual LSTM code](../../../src/traffic_prediction/context_lstm.py) · [Embeddings and context](../../concepts/11_embeddings_and_context.md)

## 12. Comparing hidden widths and resuming training

Follow one epoch through the contextual training runner. See how it saves the best checkpoint, writes the latest checkpoint, and restores the optimizer and random state when resuming. Read the selection record to see how the 128- and 256-unit candidates were compared using validation MAE. The unfinished 384-unit run was deferred.

**Check your understanding.** Which checkpoint would you use to resume training? Which would you use to evaluate the selected model? How was the final width chosen? [Training walkthrough](../../implementations/train_context_lstm.md) · [Selection record](../../../reports/final_comparison/selection.json)

## 13. Frozen test evaluation

Trace how the finalizer loads the selected contextual checkpoint, checks its identity, reproduces its validation score, and computes test errors with weights fixed. Follow how it pools overall errors and records errors for individual forecast leads and sensors. This builds on the earlier discussion of what “frozen” means.

**Check your understanding.** Why was epoch 37 selected before test scoring? What is the difference between holding the weights fixed and using a never-before-seen test period? [Frozen test evaluation](../../concepts/18_frozen_test_evaluation.md) · [Finalizer walkthrough](../../implementations/finalize_context_lstm.md)

## 14. PatchSTG and interpreting the comparison

Follow sensor information through spatial grouping, attention within and across patches, and restoration to sensor order. Then use the test table, forecast-distance graph, and recorded training settings to explain what the comparison found. Review what both models shared and where capacity, spatial inputs, and optimization differed.

**Check your understanding.** Where can sensor B influence sensor A's forecast? What does PatchSTG's lower test MAE establish for these runs? What further experiment could isolate the effect of spatial attention? [PatchSTG architecture](../../patchstg/architecture.md) · [Comparison and fairness](../../concepts/17_comparison_and_fairness.md) · [Final report](../../../reports/final_comparison/PROJECT_REPORT.md)
