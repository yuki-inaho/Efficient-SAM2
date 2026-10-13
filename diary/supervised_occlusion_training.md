# Supervised segmentation under synthetic occlusion

This experimental workflow overlays thin green curves on a reviewed image and keeps the
pre-occlusion object mask unchanged. It trains the actual SAM mask decoder with supervised
BCE and Dice losses. Feature distillation alone does not establish this behavior.

The image encoder, prompt encoder and cached high-resolution projections stay frozen.
Original positive points, verified negatives on other objects/background, and boxes are
preserved. Never turn the artificial occluder into a negative prompt when its underlying
object should be recovered. Resolve duplicate hypotheses before using neighboring object
centers as negatives; otherwise one physical object can be cut into multiple masks.

## Reviewed input contract

The manifest contains `status: REVIEWED_PILOT_ONLY` or `REVIEWED_DATASET`, plus `items`.
Each item requires:

- `object_id`: unique physical target identity; no identity may occur in multiple splits.
- `split`: `train`, `val` or `test`.
- `image_path`, `mask_path`, `image_sha256`, `mask_sha256`.
- `review_status: accepted_pilot`, `reviewer`, and concrete `observations`.
- `prompt_points`: crop-local pixel `[x,y]` pairs, `prompt_labels`: matching 1/0 labels.
- `local_box`: crop-local `[x0,y0,x1,y1]`.

Images are RGB; masks are nonempty single-object 0/255 PNGs with matching dimensions.
The loader rejects stale hashes, missing reviews, duplicate identities, invalid prompts,
and negative points that lie inside the reviewed target. `accepted_pilot` acknowledges
sample-level review only; it is not a claim of full dataset completeness.

An identity string is not proof that physical objects differ. Review identity assignments,
shared context and temporal overlap separately. A same-scene, target-disjoint pilot is not
an independent-scene benchmark. Generated SAM proposals require visual correction; agreement
with those proposals cannot by itself establish real annotation accuracy.

## Run

Use an environment with SAM 2, PyTorch, NumPy, Pillow, OpenCV and TensorBoard installed:

```sh
python tools/train_occlusion_decoder.py \
  --manifest /path/to/reviewed_pairs.json \
  --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --out /path/to/new_run --steps 320 --lr 0.00001 --negative-weight 8
```

The initial configuration uses 70% augmented samples and 30% clean samples. Curve width is
2.5–8% of the target bounding-box short side, and target occlusion is restricted to 2.5–25%.
Seeds, opacity maps, unchanged targets and evaluation images support reproducibility.
Negative point neighborhoods receive additional background loss. Positive targets underneath
synthetic occluders remain foreground. Synthetic geometry is an augmentation, not an invented
annotation of genuinely invisible scene content.

Checkpoint selection uses validation synthetic IoU with at most 0.02 absolute clean-IoU
regression. This is an experimental selection rule, not a deployment quality gate. Inspect
false-positive area, fragmentation, neighboring-object leakage and real occlusions separately.
An improvement in average IoU can coexist with worse background leakage.

The run asserts nonzero finite gradients, actual parameter updates and identical predictions
after checkpoint reload. It saves both the final and validation-selected decoder state. If
no checkpoint improves the selection rule, the selected checkpoint remains the base decoder
at step zero. Reports must not describe that outcome as an improved trained model.

After selection is frozen, test pairs receive clean and fixed synthetic comparisons. Once
those test results guide further development, pass `--evaluation-role reused_development`
and create a new untouched physical-object/scene holdout for final evaluation.

```sh
python tools/render_occlusion_comparison.py \
  --run /path/to/new_run --out /path/to/comparison.mp4
```

The comparison includes every evaluated case, including failures. Review actual output
images; a successful training command, test count, saved checkpoint or video file does not
establish annotation quality.

## Whole-dataset acceptance

`tools/workflow/annotation_quality_gate.py` checks complete frame manifests, image hashes,
instance-center correspondence and artifact-bound visual review records. It fails closed
on missing, merged or unreferenced instances and absent/stale reviews. Detector candidates
are not ground truth: visual review must also account for detector misses and false positives.
One-to-one center correspondence is necessary, but does not certify boundaries or completeness.
