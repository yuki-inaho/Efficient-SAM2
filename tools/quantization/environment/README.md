# CPU reference environment

This independent Linux x86-64/Python 3.12 environment does not change the base
project package or build its optional CUDA extension. FFmpeg must also be on
`PATH`. W&B is imported by upstream code, but evaluation disables its logging.

From the repository root:

```sh
uv sync --project tools/quantization/environment --locked
uv run --project tools/quantization/environment python -m pytest tests/test_cpu_evaluation.py tests/test_convrot*.py
```

Download the official base+ checkpoint to a directory **outside** this checkout:

```sh
curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -o "$CHECKPOINT"
sha256sum "$CHECKPOINT"
```

Expected SHA-256:
`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`.

The preregistered video smoke protocol uses the bundled SA-V **training** example,
native 1024 input, independent objects 0, 1, 2, 4, an initial manual mask at frame
0, and evaluation on frames 4, 8, 12, 16. Frame 0 is excluded from scores. It is
not held-out benchmark evidence or downstream accuracy evidence.

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_cpu.py \
  --checkpoint "$CHECKPOINT" --work-dir "$EVALUATION_DIR" \
  --frame-count 17 --objects 0 1 2 4 --threads 4 --name protocol-v1-fp32
uv run --project tools/quantization/environment python tools/quantization/evaluate_cpu.py \
  --checkpoint "$CHECKPOINT" --work-dir "$EVALUATION_DIR" \
  --frame-count 17 --objects 0 1 2 4 --threads 4 --name protocol-v1-w8a16 \
  --variant convrot-w8a16 \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32" \
  --export-gguf "$EVALUATION_DIR/efficient-sam2-base-plus-convrot-w8a16.gguf"
```

`CHECKPOINT` and `EVALUATION_DIR` are user-supplied paths outside the checkout.
The harness enforces that separation. Do not stage generated masks, logits,
downloaded weights, or private inputs. CPU ConvRot evaluation is a floating-point
dequantization reference, not an optimized integer-GEMM speed claim.

Apply the predeclared quality gate and audit sampled semantic IDs:

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_quality_gate.py \
  --report-dir "$EVALUATION_DIR/results/protocol-v1-w8a16" \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32"
uv run --project tools/quantization/environment python tools/quantization/evaluate_identity.py \
  --report-dir "$EVALUATION_DIR/results/protocol-v1-w8a16"
```

The ID audit saves every pairwise prediction/GT IoU and uses Hungarian matching
with a fixed 0.5 IoU gate. Repeated emitted integers alone are not identity
evidence. A clean sampled audit does not establish long-term tracking, downstream
identity, or occlusion reentry.

For native image parity, generate a separate original-pixel box-prompt reference.
This uses the patched `WB_Hiera` first-image path with all 25 packed windows:

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_image_cpu.py \
  --checkpoint "$CHECKPOINT" --image "$EVALUATION_DIR/fixture/frames_17/00000.jpg" \
  --box 73 245 309 783 --output-dir "$EVALUATION_DIR/image-parity/fp32"
```

Adding `--gguf "$GGUF" --mode w8a16` dumps the same reference after GGUF reload.
Outputs include raw preprocessed FP32 input, encoder features, unclamped decoder
logits, IoU score, and the thresholded mask. Native image parity is separate from
the temporal SWR/SMR video gate.

W8A8 simulation is an explicit experimental option and has not passed the
accepted full-model/native deployment gates. The commands above use W8A16.
