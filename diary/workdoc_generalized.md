# Propagation repair implementation record

## Problem and implementation

`SAM2VideoPredictor.propagate_in_video` contained a debug branch for frame indices greater
than ten. Its inner block was disabled, while the normal inference call was in an `elif`.
Non-conditioning frames above that boundary therefore reused previous outputs or could
reference an uninitialized local variable, depending on the start frame and direction.

The repair retains each object's conditioning output and always executes inference for
non-conditioning frames. Per-object output storage and tracking flags remain in the normal
path. Forward, reverse and delayed-start cases are covered by a CPU stub regression suite.

## Published changes

- Propagation control-flow repair in `sam2/sam2_video_predictor.py`.
- CPU regression tests in `tests/test_propagate_control_flow.py`, using repository-relative paths.
- Experimental training helper `tools/train_bypass_lora.py` and custom-dataset inference helper
  `tools/vos_inference_custom.py`, retained from the local implementation.
- Recursive exclusion of generated checkpoint archives and evaluation outputs.
- Generalized repair specification and engineering decisions.

The experimental helpers are not an end-to-end certification of adapter checkpoint restore,
annotation quality, GPU compatibility or UI editing. In particular, LoRA training success
must be followed by a restore-and-infer check before claiming a working deployment path.

## Validation scope

Run `python -m pytest tests/test_propagate_control_flow.py -q`. The six cases cover forward
inference, reverse inference, starting above frame ten, per-object conditioning frames,
replacement of stale non-conditioning outputs, and removal of the skipped-inference branch.
These tests exercise the actual method extracted from the source with deterministic CPU
stubs. They do not execute the neural network or measure dataset accuracy.

## Remaining integration checks

The specification defines further checks for tracking array layout, calibration, identity
association, training eligibility, visualization, polygon export and evaluation. Their
implementation may exist in a separate application workspace; this publication does not
claim those adapters or their private datasets are included in this repository.
