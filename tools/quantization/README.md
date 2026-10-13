# Reproduce Efficient-SAM2 ConvRot INT8 / GGUF CPU evaluation

This workflow quantizes Efficient-SAM2 Linear weights, exports a standard GGUF
container, reloads it, and compares CPU predictions against a floating-point
reference. It uses the public manual annotations bundled with this repository.

The Python implementation is a **floating-point correctness reference for
INT8-stored weights**. Native kernel, video-runtime, and frontend validation are
separate; the [native adoption result](#native-adoption-result) records the
accepted configuration and rejected Q8_0 candidates.

- [Numerical contract, Python API, and GGUF schema](../../efficient_sam2_quant/README.md)
- [Pinned CPU environment](environment/README.md)
- [Upstream model and inference instructions](../../README.md)
- [Public SA-V dataset and evaluation conventions](../../sav_dataset/README.md)

## 1. Create an isolated environment

Use Linux x86-64, Python 3.12, `uv`, and an FFmpeg executable on `PATH`. The lockfile
pins CPU PyTorch and the evaluation dependencies. The optional upstream CUDA
extension is not built by this environment.

Run all commands from the repository root. Keep checkpoints, generated images,
GGUF files, metrics, and other execution artifacts outside the checkout:

```sh
export EVALUATION_DIR="${EVALUATION_DIR:-$HOME/.cache/efficient-sam2-cpu-eval}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/efficient-sam2-uv-cache}"
export UV_PROJECT_ENVIRONMENT="$EVALUATION_DIR/venv"
export CHECKPOINT="$EVALUATION_DIR/checkpoints/sam2.1_hiera_base_plus.pt"
export GGUF="$EVALUATION_DIR/checkpoints/efficient-sam2-base-plus-convrot-w8a16.gguf"
mkdir -p "$EVALUATION_DIR/checkpoints"
uv sync --project tools/quantization/environment --locked
uv run --project tools/quantization/environment python -m pytest -q \
  tests/test_cpu_evaluation.py tests/test_convrot_reference.py tests/test_convrot_gguf.py
```

If the default directory or cache is not writable, explicitly choose a writable
location outside the repository before running the commands. Use new run names
or a new evaluation directory for independent repetitions. The GGUF writer
refuses to overwrite an existing artifact by default.

## 2. Obtain and verify the base checkpoint

```sh
curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
  -o "$CHECKPOINT"
sha256sum "$CHECKPOINT" bypass/ckpt/bypass_bottleneck_base.pth
```

Expected hashes:

- Base checkpoint: `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`
- Repository-provided bypass: `5e415e3d207f09ffe94216b1472a83518de5c54425ab55bdef8f331761af1c65`

Do not continue with a different or incomplete checkpoint. The evaluator uses
SAM2.1 Hiera base+, the bottleneck bypass, and strict state loading.

## 3. Freeze the comparison protocol

The fixed smoke protocol is:

- Public `sav_dataset/example/sav_000001.mp4` and its `*_manual.json`
- **TRAIN split**, not a held-out benchmark
- Native model input size 1024; CPU FP32; four CPU threads; seed 1234
- Raw frames 0 through 16, processed in order
- Four independent single-object tracks: IDs 0, 1, 2, and 4
- Full manual mask prompt at frame 0, with no subsequent prompts
- Score frames 4, 8, 12, and 16 only: 16 object-frame cases
- Original-resolution output masks from logits strictly greater than zero
- Efficient SWR/bypass and SMR enabled; WB threshold 0.7, dilation kernel 5,
  memory drop ratio 0.95, temporal memory stride 1, and no postprocessing

The evaluator writes the exact frame hashes and protocol to each run. A paired
comparison rejects different protocols. Do not choose cases or relax thresholds
after viewing candidate results. The four score frames are temporally correlated.

The predeclared regression gate requires:

| Check | Threshold |
| --- | --- |
| Paired/reference case coverage | Every declared case exactly once |
| Metrics and outputs | Finite |
| Mean candidate/reference mask IoU | At least 0.98 |
| Fifth-percentile agreement IoU | At least 0.95 |
| Minimum agreement IoU | At least 0.90 |
| Mean J change against manual masks | At least -0.01 |
| Mean F change against manual masks | At least -0.01 |
| Reference-nonempty to candidate-empty mask flips | None |

J and F are on a 0–1 scale, so 0.01 is one percentage point. The fifth percentile
uses NumPy's linear interpolation. The repository's SA-V evaluator supplies the
region J and boundary F definitions.

## 4. Run the FP32 reference

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_cpu.py \
  --checkpoint "$CHECKPOINT" --work-dir "$EVALUATION_DIR" \
  --frame-count 17 --objects 0 1 2 4 --threads 4 \
  --name protocol-v1-fp32
```

## 5. Quantize, export, and evaluate the in-memory candidate

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_cpu.py \
  --checkpoint "$CHECKPOINT" --work-dir "$EVALUATION_DIR" \
  --frame-count 17 --objects 0 1 2 4 --threads 4 \
  --name protocol-v1-w8a16 --variant convrot-w8a16 \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32" \
  --export-gguf "$GGUF"
uv run --project tools/quantization/environment python tools/quantization/evaluate_quality_gate.py \
  --report-dir "$EVALUATION_DIR/results/protocol-v1-w8a16" \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32"
```

This command exports the artifact and then evaluates the same in-memory
quantized model. It is not, by itself, a full-video reload test. The exporter
independently verifies the container and stored tensor payloads.

The `w8a16` mode name denotes the weight-only reference mode. In this CPU FP32
workflow its activations and matrix multiplication remain floating point. To
study dynamic per-token activation quantization separately, use
`--variant convrot-w8a8`, a new run name, and a different output file. W8A8 uses
floating-point dequantization/reference computation too. Do not transfer a
W8A16 result or acceptance decision to W8A8 without measuring it.

## 6. Reload the saved GGUF and repeat the full comparison

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_cpu.py \
  --checkpoint "$CHECKPOINT" --work-dir "$EVALUATION_DIR" \
  --frame-count 17 --objects 0 1 2 4 --threads 4 \
  --name protocol-v1-w8a16-reloaded --variant convrot-w8a16 \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32" \
  --gguf "$GGUF"
uv run --project tools/quantization/environment python tools/quantization/evaluate_quality_gate.py \
  --report-dir "$EVALUATION_DIR/results/protocol-v1-w8a16-reloaded" \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32"
```

The original checkpoint supplies the architecture/base model before strict
restoration of the GGUF state. This path does not re-quantize the checkpoint.
Require a successful reload report before describing persisted-artifact video
quality as validated.

## 7. Inspect artifacts and optional sampled identity checks

Each run saves `protocol.json`, `report.json`, per-case logits, and masks. The
gate command writes `quality_gate.json` and exits with a failure status when any
check fails. Preserve report and artifact hashes along with code, environment,
and command versions.

A separate public-fixture identity audit is available:

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_identity.py \
  --report-dir "$EVALUATION_DIR/results/protocol-v1-w8a16-reloaded"
```

This compares predicted masks with the public manual masks using fixed 0.5 IoU
matching. Repeated emitted ID integers alone are not identity evidence. This
small audit does not establish long-term identity retention, joint multi-object
runtime correctness, or recovery through occlusion/reentry.

For native first-image parity, prepare a separate public box-prompt reference:

```sh
uv run --project tools/quantization/environment python tools/quantization/evaluate_image_cpu.py \
  --checkpoint "$CHECKPOINT" \
  --image "$EVALUATION_DIR/fixture/frames_17/00000.jpg" \
  --box 73 245 309 783 --threads 4 \
  --gguf "$GGUF" --mode w8a16 \
  --output-dir "$EVALUATION_DIR/image-parity/w8a16"
```

This dumps preprocessing, encoder features, raw decoder logits, IoU scores, and
masks for the all-window conditioning-image path. It does not exercise temporal
SWR/SMR, and cannot replace the full video comparison.

## Measured public-fixture result and its limits

One CPU run of the fixed protocol, using 199 quantized Linear layers, produced:

| Quantity | Result |
| --- | --- |
| FP32 mean J / F | 0.92859975 / 0.98405090 |
| In-memory W8A16 mean J / F | 0.92937478 / 0.98484688 |
| Mean / p5 / minimum agreement IoU | 0.99520586 / 0.97565360 / 0.97562658 |
| Exported GGUF size | 85,212,512 bytes |
| Original base checkpoint size | 323,606,802 bytes |
| Original combined FP32 state size | 324,210,824 bytes |
| Exported tensor inventory | 1,019 tensors; 199 quantized Linear layers |

The stored artifact is about 26.3% of the original combined FP32 state size.
This includes residual unquantized tensors and format overhead; it is not an
assertion that every operation or tensor is INT8. The in-memory candidate passed
the declared TRAIN-smoke gate. The small positive J/F changes are observations
on this fixture, not evidence of a general accuracy improvement.

The saved-GGUF full-video reload also passed the hardened paired gate. All 16
reloaded logit arrays were bitwise identical to the in-memory candidate in this
run. The validator recomputes agreement summaries and J/F changes from the
per-case evidence and actual baseline report, recording its SHA-256.

Native CPU, native video, and downstream application validation remain separate
milestones. See their actual reports rather than inferring completion from the
Python results. Shared-machine timing is a functional
observation, not a controlled speed benchmark. A readable GGUF file alone does
not implement Efficient-SAM2's model graph, sparse routing, or memory retrieval.

Do not commit execution artifacts, model files, private inputs, or raw logs.
Publish only reviewed source, synthetic tests, and non-sensitive documentation.

## Optional, separate ggml Q8_0 candidate evaluation

[Explicit native-H4 restoration experiment and retained failure evidence](NATIVE_Q8_EXPERIMENT.md)
provides the optional policy-specific commands; the default below remains dense.

`evaluate_q8_0_cpu.py` studies a second quantization: inverse-ConvRot FP32 weights
are repacked into Q8_0 and activations are dynamically quantized. It first checks
its floating-point emulator against six fixtures produced by the matching native
kernel. The candidate manifest admits only verified Linear consumers; 191 layers
were selected and eight width-112 layers kept the exact FP32 fallback.

Use the matching `efficient_sam2/cpu_quant` sources in the
[sam3.cpp repository](https://github.com/yuki-inaho/sam3.cpp) and generate its
synthetic native fixtures using that component's README. Set `SAM3_CPP_DIR` and
`KERNEL_FIXTURES` to those local locations; fixture/output files stay outside the
source checkouts.

```sh
: "${SAM3_CPP_DIR:?Set the path to the matching sam3.cpp checkout}"
: "${KERNEL_FIXTURES:?Set the external native fixture directory}"
uv run --project tools/quantization/environment python tools/quantization/evaluate_q8_0_cpu.py \
  --checkpoint "$CHECKPOINT" --gguf "$GGUF" --work-dir "$EVALUATION_DIR" \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32" \
  --emulator-dir "$SAM3_CPP_DIR/efficient_sam2/cpu_quant" \
  --kernel-fixtures "$KERNEL_FIXTURES" --threads 4 \
  --name protocol-v1-q8-0-emulated
uv run --project tools/quantization/environment python tools/quantization/evaluate_quality_gate.py \
  --report-dir "$EVALUATION_DIR/results/protocol-v1-q8-0-emulated" \
  --reference-dir "$EVALUATION_DIR/results/protocol-v1-fp32"
```

On the unchanged 16-case protocol this emulator passed, with mean/p5/minimum
agreement IoU 0.99651476 / 0.98786576 / 0.97559172. Mean J and F were 0.92864389
and 0.98389342. Candidate selection, source files, native fixtures, protocol, and
source GGUF hashes are retained in its report.

This is quality evidence for the measured emulator, not native full-model speed
evidence. W8A16 and Q8_0 may use the same input GGUF hash while executing different
arithmetic; preserve the mode and selection-manifest identity in comparisons and
cache keys. Require matched native temporal parity and controlled whole-model
latency/memory measurements before enabling an optimization by default.

### Native adoption result

Full-model Q8_0 remains **experimental, pending application-level review**.
It failed the original native-fidelity gate. Although the emulator passed
the fixed manual-mask smoke gate, the matched four-object, 17-frame native box
protocol failed its unchanged minimum IoU requirement of 0.98. The original
restoration comparison reached 0.97342377; a separately declared native-H4
restoration candidate reached 0.94111809. Both failed results are retained.

The currently accepted native configuration uses the compressed ConvRot GGUF with restored
FP32 arithmetic and lossless shared encoder-prefix caching. Its four-object
native parity comparison reached mean/minimum IoU 0.99999754 / 0.99983286 against
the corresponding Python box oracle. This is graph-parity evidence on the public
fixture, not a general accuracy guarantee or evidence of native INT8 speedup.
Cache performance requires its own controlled workflow measurement. The practical
Q8_0 review reapplies the original quantization agreement and J/F-drop thresholds
to direct native Q8_0 versus restored-FP32 outputs; this is explicitly a post-run
reassessment, and the 16 saved box-prompt cases pass. Before seeing controlled
timings, default adoption additionally requires at least 1.05 median paired
whole-model speedup and the separately fixed downstream identity-quality gates.
Otherwise restored FP32 remains the default. Q8_0 implementation, tests,
experimental results, and portable benchmarks are retained for evaluation on
other CPUs; no controlled whole-model Q8_0 speedup is established here.

A separate post-run measurement used the **saved native box-seeded masks**
against the public manual ground truth, scoring frames 4, 8, 12, and 16 for all
four objects. Mean J/F/J&F were 0.91089026 / 0.97508228 / 0.94298627 for restored
FP32 and 0.91342811 / 0.97517764 / 0.94430287 for Q8_0. Direct native Q8_0 versus
native restored-FP32 mask agreement was mean/minimum 0.99239433 / 0.94560995.
These measures do not change the original fidelity-gate result. Any later
application-level adoption decision must disclose its separate criteria and
post-result basis; downstream identity quality and controlled timing remain
required. Both native variants use the same compressed GGUF; this baseline is not the untouched
official FP32 checkpoint. The box-prompt measurement also differs from the
manual-mask-initialized smoke protocol above. The small TRAIN sample does not
establish 95% general accuracy, and J&F is not pixel-classification accuracy.

### Retained whole-model benchmark for another CPU

The companion checkout retains `efficient_sam2/benchmark_video.py` and
`efficient_sam2/benchmark_protocol.public.json`. With the matching prepared
17-frame public fixture, run from this repository root:

```sh
: "${SAM3_CPP_DIR:?Set the path to the matching sam3.cpp checkout}"
uv run --project tools/quantization/environment python \
  "$SAM3_CPP_DIR/efficient_sam2/benchmark_video.py" \
  --executable "$SAM3_CPP_DIR/build/efficient_sam2_track" \
  --model "$GGUF" --frames "$EVALUATION_DIR/fixture/frames_17" \
  --protocol "$SAM3_CPP_DIR/efficient_sam2/benchmark_protocol.public.json" \
  --output "$EVALUATION_DIR/results/native-q8-paired-new"
```

Use a new empty output directory and stop competing heavy jobs. The runner
validates the fixed model/input hashes, four CPU threads, a full initialized
17-frame sequence, and the first four consumed frames. It excludes one warmup
per mode and retains all three AB/BA/AB paired samples, phase timings and peak
RSS where supported. See the companion `efficient_sam2/cpu_quant/README.md` for
build and fixture details. Its orchestration tests passed; this whole-model Q8
benchmark was not run for this delivery. Recheck both quality and performance on
the new CPU: different hardware alone does not guarantee a speedup.

## Controlled exact-prefix cache result

A separate accepted restored-FP32 workflow compared cache-off with a prewarmed
persistent cache plus one 49 MiB RAM entry. Two public objects each consumed
three frames from an initialized 17-frame sequence. The frozen executable,
libraries, model and settings matched; four CPU threads and affinity 0–3 were
configured on an AMD EPYC 9V74 host, with other heavy task work paused.

After one excluded warmup per mode, preregistered AB/BA/AB pairs gave native
cache-off/cache-on ratios **1.5007 / 1.4103 / 1.5038**, median **1.5007×** (33.37%
less native time). Median complete-process wall ratio was **1.4830×**. All 48
warmup/trial predictions had identical recorded masks, IDs, scores, logit
fingerprints and routing/memory diagnostics. No sample was discarded.

Persistent reads and feature copies are included. Each measured cached run had
three disk hits, three RAM hits and no misses/writes. Peak RSS was 949.21–949.39
MiB uncached and 905.45–905.72 MiB cached. This is a short, warm-cache measurement,
not cold-start, full-dataset or Q8_0 performance. Direct native OS thread-count
telemetry was unavailable in the sandbox; affinity/settings, host-load samples,
RSS and profiles were retained. External completion polling had up to 0.5-second
granularity. Do not extrapolate the ratio to other CPUs or workloads.

## Reusable optimization planning skills

Repository-distributed skills are available for related CPU work:

- [Write a CPU optimization workdoc](skills/write-cpu-optimization-workdoc/SKILL.md)
- [Review a CPU optimization workdoc](skills/review-cpu-optimization-workdoc/SKILL.md)

They cover profiling, unchanged numerical gates, paired controlled timing,
conversion/setup costs, memory scope, portable fallback, backend reuse, and
immediate evidence records. These are source deliverables; this repository does
not install them into a personal skill catalog. Do not infer that a proposed
optimization is implemented or faster from the presence of its plan.
