# Retained experimental Q8 CPU evaluation

Q8 is **opt-in**, with no end-to-end speedup or native fidelity acceptance claim.
The default Python restoration policy remains `dense`. The optional
`native-h4-f32-v1` policy reproduces the native inverse-ConvRot operation order
before the second lossy Q8_0 quantization. It does not change the GGUF or its
stored weights. Both policies retain the original 191 eligible Linear / eight
width-112 FP32-fallback selection on the measured base+ artifact.

The evaluator reuses `emulator.py` and `manifest.py` from the matching sam3.cpp
checkout. Native restoration additionally imports `restore_policy.py`; the helper
contains no second Q8 quantizer. Its H4 tensor-power tests use synthetic values,
including width112. All downloaded weights, real-layer fixtures and outputs stay
outside Git. Run names must be new so prior failures remain available.

## Recorded limits

On the fixed public TRAIN fixture, both Python Q8 policies passed the unchanged
16-case full-manual-mask-initialized gate (agreement mean/p5/min at least
0.98/0.95/0.90; J/F mean losses at most one percentage point; finite outputs).
The native-H4 policy obtained agreement 0.9937004490 / 0.9742343650 /
0.9373868983, J 0.9285580344 and F 0.9840674163. Sampled identity-error counts
were zero; this does not establish occlusion/reentry or general tracking quality.

The separate **68-case box-prompt native fidelity gate failed** for both policies:

| Python comparison policy | Mean mask IoU | Minimum mask IoU | Result |
|---|---:|---:|---|
| Dense restoration | 0.9972512518 | 0.9734237740 | FAIL |
| Native-H4 restoration | 0.9959965973 | 0.9411180946 | FAIL |

That gate requires mean >=0.99, minimum >=0.98 and all finite. Matching restored
weight bytes does not guarantee matching temporal predictions: activation
quantization, accumulation and recurrence can amplify small differences. The
remaining cause is not established. Do not relabel these failures as passes.

A different question is direct task quality against manual masks. For the saved
native box-seeded trajectories, scored only on frames4/8/12/16 (16 cases):

| Native computation | Mean J (region IoU) | Mean F (boundary) | Mean J&F |
|---|---:|---:|---:|
| Restored FP32 from ConvRot GGUF | 0.9108902637 | 0.9750822755 | 0.9429862696 |
| Q8 second quantization | 0.9134281122 | 0.9751776370 | 0.9443028746 |

These are GT-derived-box-assisted TRAIN observations, not generic pixel accuracy,
a held-out benchmark, or quality on a downstream private dataset. The restored
FP32 row is already based on compressed weights, not the untouched checkpoint.
Direct native Q8-versus-restored-FP32 agreement mean/p5/min was
0.9923943263 / 0.9768474100 / 0.9456099456. That descriptive comparison does not
retroactively change the fidelity gate or demonstrate a speedup.

## Prepare the environment and host-specific fixtures

Follow [the main reproduction instructions](README.md) first to create the
pinned Python environment, FP32 reference and accepted W8A16 GGUF. Set:

```sh
: "${EVALUATION_DIR:?Use a directory outside both checkouts}"
: "${SAM3_REPO:?Path to the matching sam3.cpp checkout}"
: "${CHECKPOINT:?Official base+ checkpoint outside Git}"
: "${GGUF:?W8A16 artifact outside Git}"
PYTHON="$EVALUATION_DIR/venv/bin/python"
EMULATOR_DIR="$SAM3_REPO/efficient_sam2/cpu_quant"
FIXTURES="$EVALUATION_DIR/results/host-kernel-fixtures"
mkdir -p "$FIXTURES"
cmake -S "$EMULATOR_DIR" -B "$EVALUATION_DIR/cpu-quant-build" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$EVALUATION_DIR/cpu-quant-build" -j4
"$EVALUATION_DIR/cpu-quant-build/sam2_cpu_quant_benchmark" 1 3 "$FIXTURES" \
  > "$EVALUATION_DIR/results/host-kernel-probe.csv"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" \
  "$EMULATOR_DIR/test_emulator.py" "$FIXTURES"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -m pytest -q \
  "$EMULATOR_DIR/test_restore_policy.py" tests/test_q8_cpu_evaluation.py
```

Build flags must be appropriate to the target CPU; retain them, compiler version
and loaded-library identities with results. The current activation emulator
explicitly targets measured x86 AVX/AVX2 nearest-even behavior. Other ISA kernels
may differ. Fresh native fixtures are mandatory; if fixture verification fails,
stop and implement a separately named, reviewed arithmetic profile rather than
skipping checks or copying fixtures from another machine. Python emulation is
not an integer-kernel speed benchmark.

## Unchanged 16-case full-mask quality and sampled identity

From the Efficient-SAM2 checkout, use a distinct run for the native policy:

```sh
NAME=protocol-v1-q8-native-h4-new-host
"$PYTHON" tools/quantization/evaluate_q8_0_cpu.py \
  --checkpoint "$CHECKPOINT" --gguf "$GGUF" --work-dir "$EVALUATION_DIR" \
  --name "$NAME" --threads 4 --objects 0 1 2 4 \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32" \
  --emulator-dir "$EMULATOR_DIR" --kernel-fixtures "$FIXTURES" \
  --restore-policy native-h4-f32-v1
"$PYTHON" tools/quantization/evaluate_quality_gate.py \
  --report-dir "$EVALUATION_DIR/results/$NAME" \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32"
"$PYTHON" tools/quantization/evaluate_identity.py \
  --report-dir "$EVALUATION_DIR/results/$NAME"
```

For the original policy, use `--restore-policy dense` and another new run name.
Never compare or share caches solely on GGUF hash: both policies use the same
container. Keep the restore policy, candidate manifest, emulator source hashes,
quantization/activation mode, preprocessing and native library identities.

## Four matched 17-frame box references

Proceed only after the chosen full-mask quality gate passes. The checked-in
[public box protocol](native_box_protocol.public.json) freezes four boxes,
frames0..16, image hashes, artifact hash and unchanged fidelity gates. Its SHA256
is `6bdb7e8d2d2fa6d764a8b4346789bedaf65649152ba5ddf547382bfe01c8bc7a`.
It is tied to the recorded GGUF hash. For a different artifact, preregister a
separate protocol explicitly before measurements; do not relabel old results.

```sh
cp tools/quantization/native_box_protocol.public.json \
  "$EVALUATION_DIR/native-box-protocol.json"
for spec in '0 165 747 253 797' '1 73 245 309 783' \
            '2 258 61 337 241' '4 115 393 201 522'; do
  read -r id x0 y0 x1 y1 <<EOF
$spec
EOF
  "$PYTHON" tools/quantization/evaluate_q8_0_cpu.py \
    --checkpoint "$CHECKPOINT" --gguf "$GGUF" --work-dir "$EVALUATION_DIR" \
    --name "box-q8-native-h4-new-host-object$id" --threads 4 --objects "$id" \
    --initial-box "$x0" "$y0" "$x1" "$y1" --dump-temporal \
    --parity-protocol "$EVALUATION_DIR/native-box-protocol.json" \
    --manifest-reference "$EVALUATION_DIR/results/$NAME/candidate_manifest.json" \
    --emulator-dir "$EMULATOR_DIR" --kernel-fixtures "$FIXTURES" \
    --restore-policy native-h4-f32-v1
done
```

This produces low/original-resolution logits, binary masks and SWR/SMR traces.
Compare all 68 native cases to the matching policy, with the same exact image
bytes, original-pixel boxes and full initialized sequence length17. First verify
correctness, then run the native checkout's `efficient_sam2/benchmark_video.py`
paired whole-model benchmark with cache disabled for both variants. Preserve
load/conversion time, raw samples, peak memory and build/ISA details. A newer CPU
may change the trade-off; its speed and accuracy still require measurements.

The historical full-model numbers above came from archived, source-hashed
isolated evaluation drivers. The subsequently published optional CLI integration
was checked with 102 synthetic/unit regressions and the retained seven actual
native restoration/packed-byte fixtures; no additional model inference was run
solely for this refactor. Keep that distinction when auditing reproduction.
