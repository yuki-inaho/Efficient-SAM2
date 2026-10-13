---
name: write-cpu-optimization-workdoc
description: Create or refine a Japanese repository workdoc for precision-preserving CPU inference optimization, including profiling, ggml kernel reuse, controlled benchmarks, portability, regression gates, and immediate execution records. Use for CPU optimization plans or work records; not for unrelated document writing or a promise of speedup.
---

# Write CPU Optimization Workdoc

Create an executable Japanese work plan and record. Keep the user's model,
required features, accuracy thresholds, and authorized publication scope intact.
Use [references/workdoc-template.md](references/workdoc-template.md) for the
required structure; fill it with repository facts rather than blank placeholders.

## Establish the task

1. Read repository instructions (`AGENTS.md`, `CODEX.md`, `CLAUDE.md`) and the
   existing workdoc. Extend its traceability IDs instead of replacing prior work.
2. Run `date '+%Y-%m-%d %H:%M:%S %Z%z'` and `git rev-parse --show-toplevel`.
   Record the actual timezone, HEAD, branch, working tree, and single-writer
   ownership. Do not reset unrelated changes.
3. Inspect build/test entry points, `justfile`, Python/uv dependencies, native
   backend versions, and relevant execution graphs. Use documented wrappers and
   `uv run` for Python. Record unavailable dependencies explicitly.
4. Write `temp/workdoc_<Mon><DD>-<YYYY>_<task>.md`, using
   `LC_TIME=C date '+%b%d-%Y'`. Preserve the existing path when updating a plan.

## Separate evidence levels

- Distinguish stored quantization, dequantized reference arithmetic, actual
  integer kernels, graph integration, and end-to-end application execution.
- Preserve existing accuracy/data/prompt/frame gates exactly. If none exist,
  define them before candidate measurements and record their authority. A faster
  kernel does not authorize reduced model features or weaker quality targets.
- Profile first where a runnable baseline exists. If the graph is blocked, label
  synthetic/operator work as such; do not invent an end-to-end profile.
- Pin model/config/input hashes, preprocessing, prompts, seeds, threads, and
  output selection. Do not use candidate results as their own reference truth.
- Treat changes of quantization format, rotation layout, clipping, accumulation,
  normalization, and attention precision as possible numerical changes.

## Plan measured, portable optimization

1. Locate suitable existing ggml/llama.cpp primitives and document the pinned
   source/version, operation semantics, shape/stride constraints, and license.
   Reuse their dispatch and buffer lifetime conventions before adding kernels.
2. Define small interfaces with explicit ownership. Separate loading, execution,
   and measurement. Avoid duplicated numerical logic, global mutable state, or
   abstractions without a measured use. Apply SOLID/KISS/DRY within repository
   style, including repositories that deliberately use structs and free functions.
3. Preserve a portable reference/fallback for unsupported ISA, shape, alignment,
   or backend. Test fallback selection and results; never silently pad, drop
   features, or reduce the graph to make a fast path fit.
4. Predeclare representative shapes, threads, warmups, repetitions, A/B order,
   numerical tolerances, and summary statistics. Use the same input, build flags,
   CPU, and resource budget. Alternate paired baseline/candidate runs and retain
   every observation, not only the best run.
5. Separate conversion, allocation/planning, warmup, steady-state inference, and
   complete application latency. Record peak memory and its measurement scope.
   Process-wide high-water RSS is not incremental per-kernel memory.
6. Make noisy/concurrent measurements inconclusive or repeat them under control.
   A microbenchmark gain does not establish model or application speedup.
7. Require regression tests and the unchanged model-level gates after integration.
   Record a justified non-adoption if a candidate is slower, unstable, or less
   accurate. Do not promise a gain in advance.

## For encoder reuse or stateful video caches

Identify a provably prompt-independent tensor boundary before sharing features.
Keep object-conditioned routing and temporal/memory state isolated. Cache keys
must encode every dependency: model/artifact, backend/quantization, preprocessing,
image bytes, feature layout, policy, and any required route/state identity. Build
fingerprints must cover transitive kernel/loader/header dependencies and actual
per-configuration compiler flags; thread identities use resolved runtime values.
Identify actually loaded runtime libraries after backend initialization, not only
link-time paths, and disable persistent reuse when identity cannot be verified.

Prefer bounded files plus a versioned checksum manifest when that meets the
workflow; add a database only for a demonstrated need. Plan atomic writes,
validation, eviction, export/import/resume, and stale/corrupt-cache rejection.
A saved-project causal replay may satisfy resume; disclose replay cost and avoid
adding live-state serialization unless required. Test cold/warm output equivalence,
cross-object isolation, arbitrary start/reset, absence/nonpositive-presence, and
known-length versus streaming sequence semantics at length boundaries,
and edited-prompt invalidation from the earliest affected frame onward. Preserve
unaffected image features only when their dependency keys still match. Measure
cache I/O, memory/disk budgets, hit rates, recomputation, and complete workflow
latency. Preserve raw results separately from supervised corrections.

## Make execution auditable

Use atomic steps, each with exactly these four checkboxes:

```markdown
### 手順 n: 具体的な一操作の目的 (TR-ID)
- [ ] 🖐 **操作**: 実行する単一コマンドまたは編集。
- [ ] 🔎 **確認**: 観測可能な期待結果。
- [ ] 🧪 **テスト**: 失敗から成功への確認、またはTDD対象外の理由。
- [ ] 🛠 **エラー時対処**: 失敗時の安全な次の行動と停止条件。
```

Replace the sample labels with concrete IDs and commands in the output. Split
large tasks before execution. Keep investigation, design/preregistration,
implementation, validation, and publication in separate phases. Link each goal
to a step, evidence artifact, and definition-of-done item.

When executing the workdoc, select one unchecked item, record the start time,
perform only its operation, verify it, and append the result immediately after
checking that item. Keep failures unchecked and log the blocker/restart condition.
Do not batch checkmarks or silently stop at a pending result. Follow any existing
project reminder/action-count convention instead of inventing a conflicting one.

## Privacy and delivery

Keep inputs, weights, generated outputs, and raw logs outside the repository when
required. Private execution records may contain necessary local paths; publish
only separately reviewed generic commands and public-fixture evidence. Preserve
upstream copyright/license notices and record adapted source provenance. A
permission to publish source does not imply permission to publish model weights
or private data. Stage only explicitly reviewed paths under the granted scope.

Before delivery, verify file existence, commands, linked artifacts, checklist
structure, and current evidence. Report complete, failed, running, and not-run
states separately. Never use an all-green checklist to cover a missing runtime,
quality result, portability check, or user-required feature.
