# GPU execution, precision, utilization, and timing

A CUDA-enabled environment does not automatically put a model on the GPU.
Weights, input tensors, targets, calendar, and masks must be on compatible
devices. Restored Adam state must also match its parameters' device requirements.
Our first LSTM epoch was CPU; later continuation used CUDA in the user's terminal.

GPU utilization is sampled activity, not a direct measure of useful speed.
Small kernels, CPU preparation, transfers, and synchronization can limit it.
Compare epoch duration under documented settings rather than targeting 100%.

## Precision and the failed parity check

An early contextual GPU probe passed width 128 but failed isolated-versus-batched
agreement at width 256. Batched/sliced numerical kernels can differ. The agent's
build had cuDNN RNN TF32 enabled by default; this was a possible contributor,
not a proven diagnosis of the host failure.

The contextual probe/trainer now explicitly sets global, matmul, and cuDNN RNN
precision to IEEE FP32, with no AMP. Deterministic algorithms alone do not set
these precision controls. The corrected GPU probe passed all three widths.
It also checks actual recurrent inputs exactly, so layout correctness does not
rely only on output tolerance. See the
[recorded follow-up](../../temp/notes/LSTM_CAPACITY_EXPERIMENT.md).

## Timing scope

CUDA runs asynchronously. The probe synchronizes before/after timed updates and
excludes one warm-up update. It repeatedly uses an already transferred batch;
its estimate omits full-epoch loading/validation. Training reports record real
epoch times including checks and accumulation. These are not controlled
cross-architecture speed benchmarks.

The agent environment cannot currently access CUDA; the user's terminal can.
CPU checks and host GPU evidence are identified separately in reports.

See [GPU probe implementation](../implementations/check_context_lstm.md).
