# Video annotation quality repair specification

## Objective and interfaces

Produce per-frame masks with stable clip-local object IDs and explicit validity states,
then train only on accepted annotations. Each run records the ordered image manifest,
model/configuration identifiers, code revision and output hashes. Use zero-based frame
indices and original-image pixel coordinates throughout. Preserve previous run outputs.

This specification is application-independent. Repository paths refer to this checkout.
Integration adapter names below describe the external annotation application; implement or
update equivalent modules there. Do not treat their mention as a claim that they are shipped
in this repository. Object classes and distance thresholds belong in private configuration.

## Propagation control flow

Repository location: `sam2/sam2_video_predictor.py`, `propagate_in_video`.
For each object and frame, use that object's conditioning output when present; otherwise
call `_run_single_frame_inference`. Debug logging must not decide whether inference runs.
Every branch must assign the output for the current object and current frame.

Synthetic acceptance case: thirteen frames, two objects, conditioning only at frame zero.
Expect twenty-four inference calls; frames eleven and twelve must contain their own outputs,
not the final object's frame-ten output. Also test reverse order, a start above ten,
different conditioning frames per object and replacement of stale cached outputs.
After CPU tests, run a short real sequence across the boundary before a longer sequence.

## Tracking array contract

Integration locations: tracking adapter and tracking comparison module.
OpenD4RT returns coordinates `[N,T,2]`, visibility `[N,T]` and confidence `[N,T]`, where
N is the input point count and T the input frame count. Serialize point i using `[i,:,:]`
and `[i,:]`; assert all shapes against the input manifests. Preserve each query ID and its
query frame. Reject comparison inputs with mismatched image identities or query identities.
Split visual trajectory segments at invalid or invisible observations.

Acceptance: use a synthetic two-point, three-frame array with distinguishable coordinates.
Verify every serialized coordinate, not just lengths or global averages. Re-export all
frames after repair; previously omitted frames cannot be recovered by transposing a
truncated export. Retract metrics derived from the incorrect layout.

## Calibration and selection

Integration location: depth-based label selection module.
Convert depth samples to metres, unproject with depth intrinsics, apply the extrinsic
rotation and metre-valued translation, then project with RGB intrinsics. Exclude invalid
depth, points behind the camera and projections outside the RGB image. Do not clamp outside
coordinates to the image edge. Resolve multiple projected samples with an explicit depth
ordering rule where an aligned depth image is constructed.

Define the configured distance criterion explicitly, for example median positive RGB-camera
Z within a valid object mask. Record the statistic, valid sample count and selection reason.
Insufficient valid samples produce `depth_unknown`, not verified background.

Acceptance: synthetic known translation, invalid samples, out-of-bounds projections and
threshold-boundary cases. Inspect an actual depth/RGB overlay before regenerating labels.

## Prompt association and identity

Integration locations: prompt-to-mask adapter and identity association module.
Represent each observation with `track_id`, `frame_index`, `point_xy`, `class_id`,
`association_status` and `association_evidence`. Status is `matched`, `new` or `unresolved`.
Associate later prompts against existing tracks at the same frame, using configured position,
mask and depth gates. A matched prompt reuses the existing ID. An ambiguous match remains
unresolved. Failure to match alone does not establish a new object; new-object assignment
requires an explicit determination and recorded evidence. Enforce one-to-one assignments.

Preserve per-frame visibility and missing-state information, including reappearance. A state
at the final frame alone cannot represent the sequence. Inspect identity labels before and
after each correction frame. Tests cover ID reuse, distinct objects, conflicting candidates,
new-object assignment and a gap followed by reappearance.

## Training eligibility

Integration locations: annotation validity and dataset export modules.
Distinguish verified foreground, verified background, verified object absence and unannotated
regions. A failed inference, unknown depth or unresolved identity is not a negative label.
Use the loader's supported ignore representation; when unsupported, exclude incomplete
training samples. Record reasons and retain source-frame and global-ID mappings.

When a loader requires initial masks for every supervised object, create episodes beginning
at valid initial observations. Do not silently erase later objects to satisfy the loader.
Validate eligibility per frame and per supervised object. An empty annotation file or missing
file is meaningful only under a documented loader contract.

Acceptance: all included samples satisfy the loader contract; unknown annotations cannot
become background; a later appearing object is either represented in an appropriate episode
or explicitly excluded. Splitting a video must preserve original identities and frame mapping.

## Visualization

Integration location: overlay renderer.
Choose a deterministic color from `track_id`, independent of the set of visible IDs. Display
ID, class, frame index, timestamp and prompt-frame status. Produce synchronized raw RGB,
unfiltered-mask and filtered-mask views. Mark uncertainty and missing states explicitly.

Acceptance: removing one object does not change another object's color. Decode every output
frame and inspect prompt boundaries plus intermediate non-prompted frames for frozen masks,
missing visible objects and identity changes. A mask-count check alone cannot certify quality.

## Polygon export and editing

Integration location: annotation UI export module.
Pin the UI schema/version, validate category and ID types, and preserve disconnected regions
under a common object ID. Keep PNG masks authoritative. If the target cannot represent holes
losslessly, declare the polygon as an approximation and preserve the source mask separately.
Do not replace shapes merely to pass vertex-count validation without measuring added area.

Acceptance: disconnected components, holes and degenerate small components have explicit
handling and round-trip tests. Measure both omitted and added pixels. Validate the generated
JSON with the target schema and verify UI load, edit and save. If UI execution is unavailable,
record only format validation as complete; do not claim an editing workflow is verified.

## Evaluation and retraining

Integration locations: mask evaluator, training launcher and run reporting.
For a shared ID namespace, include reference objects missing from predictions with IoU zero;
report extra predicted objects separately. If namespaces differ, define the association rule.
Use explicit source-frame mappings rather than sorted-file zip. Separate unannotated frames
from verified empty backgrounds; evaluate false positives on the latter. Report exclusions,
coverage and denominators. Handle all-background cases without division or formatting errors.

Acceptance: a reference with two objects and a prediction containing one penalizes the missing
object. Verify extra-object and empty-background cases. Evaluate a held-out sequence, describe
reference labels as generated when applicable, and report agreement separately from accuracy.
Train again only after label validity and visual review pass. Save model/settings references,
training outputs and evaluation manifest together. Adapter training additionally requires
checkpoint restoration followed by successful inference before deployment is considered done.

## Execution order

1. Repair propagation and pass CPU regression plus a short real-sequence boundary check.
2. Repair tracking layout and regenerate aligned trajectories and comparison metrics.
3. Correct depth projection and selection validity.
4. Connect tracking evidence to prompt association and stable IDs.
5. Regenerate a full label sequence and review fixed-ID visualizations.
6. Build eligible training episodes and validate annotation export and editing.
7. Retrain, restore checkpoints, evaluate held-out inputs and record remaining limitations.

File creation, a zero exit code or unit-test success alone does not establish end-to-end
annotation quality. Each acceptance check must be linked to its actual output.
