# Vocabulary used in the project

| Term | Meaning here |
| --- | --- |
| Feature/channel | A value measured at a sensor/time; one raw traffic-flow channel |
| Sensor | One traffic series/location; original array order is preserved |
| Cadence | Assumed time between consecutive observations: 15 minutes |
| Origin | Time boundary after available history, before future targets |
| History | Twelve observed steps used as input |
| Horizon/lead | Number of future steps / a particular future offset |
| Stride | How far an example start moves: one step |
| Window/example | One paired historical input and future target |
| Batch | Up to four examples processed together |
| Epoch | One full pass through eligible training examples |
| Update | One optimizer step after a nonempty batch loss |
| Tensor | PyTorch numerical array supporting device placement/autograd |
| Parameter | Learned weight or bias |
| Buffer | Fixed model-associated state such as scaler statistics |
| Embedding | Learned vector selected by an integer category/identity |
| Hidden state | Recurrent representation exposed by an LSTM step |
| Cell state | LSTM internal memory updated by gates |
| Loss | Training objective; original-unit masked MAE here |
| Gradient | Derivative used to adjust a parameter |
| Adam state | Running optimizer moments and step information |
| Mask | Boolean inclusion flags; nonzero targets included |
| Coverage | Number/fraction of included targets, not proven source validity |
| Validation | Later data used to select settings/checkpoints |
| Test | Frozen-system assessment after choices are fixed |
| Overfitting | Training improvements that fail to generalize adequately |
| Checkpoint | Saved model/experiment state |
| best.pt | Selected lowest-validation-MAE weights |
| last.pt | Latest completed epoch state used for resume |
| RNG state | Position of a random-number generator, beyond its initial seed |
| CUDA | GPU execution path supported in the user's pml terminal |
| TF32 / IEEE FP32 | Different numerical execution precision settings |
| Provenance | Recorded data/code/config/environment identity |
| Leakage | Use of information unavailable under the intended prediction protocol |
| Baseline | Simple reference predictor |
| Ablation | Controlled change removing/altering a component to investigate its effect |
| Shared per-sensor model | Same weights process each sensor independently |
| Spatial interaction | One sensor's prediction can use information from other sensors |

For explanations rather than definitions, start at the [index](README.md).
