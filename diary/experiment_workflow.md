# Generic experiment workflow

Run commands from the repository root in an environment containing the repository's model
dependencies, Hydra, TensorBoard, and pytest. LoRA additionally uses peft. The example assumes
the model environment is `.venv`; override `python` when it is elsewhere.

## Training inputs

Provide JPEG frame sequences and per-object PNG annotations in the layout expected by
`tools/train_bypass_all.py`. Frame names must be integer sequences. The upstream training
loader currently selects video-list indices whose remainder modulo four is one; it also
requires the prompted object set to exist throughout each selected training sequence.
Construct the list deliberately and verify that every selected sequence has usable labels.

```sh
python tools/train_experiment.py run_name=example \
  data.base_video_dir=/path/to/JPEGImages \
  data.input_mask_dir=/path/to/Annotations \
  data.video_list_file=/path/to/train_list.txt
```

The three input paths and run name are mandatory. Use `dry_run=true` to inspect the command.
An existing run directory is rejected. GPU capacity is checked before training. Outputs are
under `runs/example/`: resolved `config.yaml`, `command.json`, `run.json`, `train.log`,
`tensorboard/` and `checkpoints/`. The trained candidates remain pending validation.

For the image-encoder adapter trial, add `train_script=tools/train_bypass_lora.py` and
`train.epochs=1`, with a different run name. The current trial targets the last global
attention block's explicit projection modules. It fails if those targets do not exist or
if no nonzero adapter gradients and parameter changes are observed.

## Validation and selection

`python -m tools.evaluate_checkpoints --run-dir runs/example --dataset-root /path/to/validation --video sample_clip`
requires a base+ model experiment and a prepared validation directory with `JPEGImages/`,
packed indexed-PNG `Annotations/`, `val_sel.txt`, and `manifest_sample_clip.json`.
`val_sel.txt` must contain only the requested video name for this per-video comparison.
The manifest's `files` mapping uses integer frame names as keys; each value may contain
`source_rgb`. Missing frame correspondences fail validation. Add `--lora` to restore the
matching `.lora.pt` adapter for every candidate. Use the same base model weights as training.

The resulting `scores.json` records candidate commands, hashes, reference and scores.
The validation set is used for selection, so do not report it as an independent test.
Select the best candidate explicitly:

```python
import json
from pathlib import Path
from tools.workflow.checkpoints import keep_best

run = Path("runs/example")
scores = json.loads((run / "scores.json").read_text())
selection = keep_best(
    run / "checkpoints",
    run_id=scores["run_id"],
    bypass_type="bottleneck",
    scores=scores["scores"],
)
info = json.loads((run / "run.json").read_text())
info.update(selection_status="selected", best=selection["best"],
            checkpoints=[selection["best"]], archived_checkpoints=selection["superseded"])
(run / "run.json").write_text(json.dumps(info, indent=2) + "\n")
print(selection)
```

Selection writes `best_selection.json`. Nonselected candidates and their adapter companions
move into a run-specific archive. Pretrained weights and unrelated runs are untouched.
Archive collisions and incomplete adapter pairs are errors rather than overwrite requests.

## Checks

```sh
python -m pytest tests -q
python scripts/verify_lora_adapter.py
python -m tools.workflow.export_scalabel --help
python -m tools.workflow.audit_polygon_export --help
```

The adapter contract uses small CPU modules; it does not download model weights or prove
application accuracy. The export audit checks both source coverage and added background
at pixel centers. Hole approximation must remain explicit when polygons are used for display.

`tools.workflow.track_lifecycle.add_lifecycle_states(frames, gone_gap=25)` accepts
chronologically increasing frame dictionaries with `frame` and `objects`. Each object entry
contains its raw-mask `area`. It adds a `states` mapping to each frame and returns the final
state mapping. Zero-area entries are absent; the configured gap counts frame indices.
