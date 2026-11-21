# Traffic forecasting lessons

**Review pass:** Restarted at Lesson 1, one forecast window. Work through one concept at a time and pause for an understanding check before moving on.

## Learning progress as of November 20, 2026

- **Previously covered:** Lessons 1–11. I understand overlapping windows, the shared small LSTM, the contextual LSTM's 41 inputs and stacked layers, and how gradients update the weights and embeddings.
- **PatchSTG concepts understood:** The three initial input channels, the role of the learned projection and calendar/sensor embeddings, spatial patches, and the basic purpose of Query, Key, and Value.
- **PatchSTG implementation still to study:** Trace the exact dimensions and attention heads, work through the input projection numerically, follow real sensor indices through patch construction, calculate scaled attention and weighted values by hand, and see how the five blocks and forecast head produce 12 predictions per sensor.
- **Experiment interpretation:** PatchSTG had lower test MAE in the recorded comparison. The runs differed in capacity, optimization, and training configuration, so the result does not isolate the effect of spatial attention.

- [Lessons 1–10](completed.md) records what we covered, including questions about batch shapes, overlapping windows, and reading the report.
- [Lessons 11–14](upcoming.md) contains the contextual LSTM review and the plans for training, frozen evaluation, and PatchSTG.
- [My Doubts](../MY_DOUBTS.md) holds the questions and calculations you asked to keep separately.

This sequence follows the project's [reading guide](../guide.md). “Covered” means we discussed the lesson together.
