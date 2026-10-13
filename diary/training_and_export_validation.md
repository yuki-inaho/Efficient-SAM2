# Training, adapter restoration, and annotation export

A training job could finish successfully while its wrapper failed to manage its outputs.
Checkpoint selection also compared epoch numbers across unrelated experiments, and the
TensorBoard file could exist without containing any loss samples. These behaviors made
training completion and checkpoint quality difficult to verify.

Training now supports an explicit run identifier and output directory. The wrapper records
resolved Hydra settings, the command, logs, checkpoint inventory, and selection status.
TensorBoard loss logging is independent of W&B. Checkpoint selection requires finite
validation scores for every candidate in one run; equal scores prefer the earlier epoch.
Unrelated experiments are untouched, and archive collisions fail before files are moved.
Nonselected candidates are archived for audit rather than silently deleted.

The adapter experiment now targets explicit image-encoder projection modules. The dense
reference path runs without gradients and with adapters disabled. The sparse path trains
the adapter and bypass. Saved adapters include configuration and evidence of nonzero
gradients and parameter changes. Restoration checks every adapter key, shape, and finite
value before copying weights. Training success alone is not treated as restoration evidence.

Local validation completed a bypass run and an image-adapter run, then restored the saved
weights for video inference. Comparison did not establish a quality or speed advantage.
Selection validation is distinguished from an independent test set, and agreement with a
model-generated reference is distinguished from accuracy against human annotation.

Annotation diagnostics now distinguish label continuity from spatial identity-switch
candidates. Tied spatial matches remain ambiguous. A candidate is not a confirmed switch.
Polygon export was also checked in both directions: source coverage and added background.
Degenerate components use small rectangles along occupied pixel rows instead of oversized
triangles. Holes remain a documented approximation, so indexed PNG masks remain canonical.

The public helpers and regression tests preserve generic engineering behavior. Local
capture names, application categories, media, credentials, experiment weights, and raw
conversation records are excluded from publication. UI round-trip editing and comprehensive
annotation coverage are not claimed by these checks.

A completion audit also found that visibility state existed only at the end of the clip.
The generic lifecycle helper now records causal state for each frame: a seen object is
visible, then lost within the configured absence gap, then gone beyond it. Reappearance
restores the same identifier. Future appearances do not create earlier state, and depth
eligibility remains independent of raw-mask visibility. The local annotation adapter and
overlay use this state sequence.
