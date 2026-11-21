# Supported commands

## Refresh submission documents — no model execution

Run from the project root:

```bash
python scripts/build_project_report.py
python scripts/build_presentation.py
```

The first refreshes `reports/final_comparison/PROJECT_REPORT.md` and its five plots.
The second writes decks, PDF and notes under `presentations/`. Both use saved
measurements and portable plot data; local checkpoints and raw data are not needed.
The following sections describe model/data commands and may require local assets.


## Current completion scope

Width 384 has been deferred. Do not resume the three-candidate trainer unless
explicitly returning to that work. Both main models have completed measured
results. Rebuild the integrated project report and figures without model execution:

```bash
python scripts/build_project_report.py
```

PatchSTG's saved run is reused; no retraining or reevaluation is needed.
Its [model/tooling guide](patchstg/README.md) describes the integrated sources,
tests, configurations, artifacts and saved replay.

For deliberate reproduction of the LSTM test pass into a new directory only:

```bash
python -u scripts/finalize_context_lstm.py --device cuda --output-dir runs/final_comparison_repeat
```

See [finalization status](../reports/finalization_status.md). The start/resume
examples below are retained for reference, not the currently recommended next step.

Run from the Traffic Prediction PML folder in the existing environment:

```bash
conda activate pml
```

These commands are references, not instructions to rerun every completed
experiment. Audit/evaluation outputs require new paths; training data/checkpoints
should be preserved. Do not launch duplicate training processes against one suite.

## Read data and verify implementations

```bash
python scripts/inspect_data.py
python scripts/check_context_training.py
```

The first runs baseline validation, not just a file listing. The second uses
synthetic data/temp files. The archived small-LSTM diagnostic is documented in
[its walkthrough](implementations/check_lstm.md); it is not a current root command.

For a deliberate new audit:

```bash
python scripts/audit_data.py --output-dir reports/data_audit_v2
```

For a deliberate repeated contextual GPU probe, choose an unused report name:

```bash
python scripts/check_context_lstm.py --device cuda --steps 5 --output reports/context_lstm_gpu_repeat.json
```

Probe updates are disposable; they do not save models for subsequent training.

## Start a new contextual suite

Only use this when starting a new experiment in an unused output directory:

```bash
nohup python -u scripts/train_context_lstm.py --device cuda --output-dir runs/context_lstm_suite_v2 > context_lstm_new.log 2>&1 &
```

Each width trains fresh for 50 epochs. Candidate setup, protocol and GPU precision
are fixed by the runner. No test scores are calculated.

## Resume the current suite

Ensure its earlier process is stopped, then:

```bash
nohup python -u scripts/train_context_lstm.py --device cuda --output-dir runs/context_lstm_suite_v1 --resume-suite >> context_lstm_resume.log 2>&1 &
```

`>>` appends output to preserve earlier resume sessions. A single `>` replaces
the log when launching a command. This affects terminal output, not epoch history
stored in checkpoints and report.json. Earlier overwritten batch-progress lines
cannot be recovered exactly from the per-epoch reports.

Completed candidates are reverified. The incomplete model resumes after its last
saved epoch; a partial epoch is repeated. The command does not restart all models
from epoch zero or grant 50 additional epochs.

## Monitor and pause

```bash
tail -n 30 context_lstm_resume.log
pgrep -af '^python -u scripts/train_context_lstm.py'
```

The process list gives the actual current PID. To pause, replace the example
placeholder with that PID:

```text
kill -TERM <actual-training-PID>
```

Do not copy an old PID from conversation history. Termination keeps completed
checkpoints but does not save partial-epoch updates. Do not run resume immediately
if the intention is to stay paused. A report status 'running' may remain after
termination because it records last saved progress, not live process state.

`nohup` keeps the process alive after terminal disconnect; it does not prevent
Windows sleep, machine shutdown, loss of power, or process failure. Keep the
laptop powered and awake. A shell job number/PID alone is not proof training is
healthy: check the log for progress or an exception.

## Original small LSTM commands

Its training/test experiments are already complete. Read the corresponding
[training](implementations/train_lstm.md), [continuation](implementations/continue_lstm.md)
and [test evaluation](implementations/evaluate_lstm.md) pages before rerunning.
The existing evaluate_lstm.py cannot evaluate contextual checkpoints.
