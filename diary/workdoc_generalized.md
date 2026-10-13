# Video annotation and experiment-management implementation record

## Behavior

Non-conditioning video frames always execute inference, including frame indices beyond
the former debug boundary. Conditioning outputs and per-object state remain intact.

Training outputs belong to an explicit run. Resolved settings, command, logs, checkpoint
inventory and selection status are recorded. TensorBoard receives loss samples even when
W&B is disabled. A run remains pending selection until every candidate has a finite
validation score. Selection never compares unrelated runs or treats a larger epoch number
as evidence of higher quality. Associated adapter files move with their checkpoints.

Image-encoder LoRA uses explicit projection targets and a frozen dense reference path.
Saved adapter files contain configuration and evidence of gradients and changed weights.
Restoration validates all adapter keys, shapes and finite values. Custom inference respects
the requested output path and can restore the adapter before propagation.

Generic evaluation helpers report missing objects, label continuity, ambiguous spatial
matches and identity-switch candidates. Polygon export retains all connected components;
degenerate components use occupied pixel-row rectangles to avoid inflated background.
PNG masks remain canonical because polygon holes are approximated.

## Public code

- `sam2/sam2_video_predictor.py`: propagation control-flow repair.
- The three bypass predictor variants: TensorBoard logging independent of W&B.
- `tools/train_experiment.py`, `tools/conf/train_bypass.yaml`: run-managed Hydra entry point.
- `tools/train_bypass_all.py`, `tools/train_bypass_lora.py`, `tools/lora_adapter.py`: training and adapter persistence.
- `tools/vos_inference_custom.py`, `tools/evaluate_checkpoints.py`: explicit inference paths and validation.
- `tools/workflow/`: checkpoint selection, GPU capacity gate, mask and temporal metrics, polygon export and audit.

## Validation and limits

The CPU regression suite covers propagation, run isolation, score completeness, adapter
pair retention, scalar logging, output paths, missing-object metrics, identity diagnostics,
and polygon coverage. A separate CPU adapter contract checks update, frozen-reference
behavior and save/restore output equality with the model dependencies installed.

Local GPU experiments also completed training, scalar logging, checkpoint loading, adapter
restoration and video inference. Their private inputs and artifacts are not published.
Comparison did not establish a quality or speed advantage. Selection validation is not an
independent test. Generated reference labels are not human ground truth. UI round-trip
editing and comprehensive annotation coverage remain outside the verified scope.
