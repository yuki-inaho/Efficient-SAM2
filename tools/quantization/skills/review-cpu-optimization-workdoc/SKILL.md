---
name: review-cpu-optimization-workdoc
description: Review and safely improve Japanese CPU optimization workdocs against repository code and measured evidence, checking unchanged accuracy gates, fair benchmarks, ggml reuse, portability, and complete execution records. Use for optimization-plan review or completion audits, not unrelated prose editing.
---

# Review CPU Optimization Workdoc

Audit whether another engineer can execute the plan and whether its completion
claims follow from actual evidence. Read
[references/review-rubric.md](references/review-rubric.md) before assigning a
verdict. Preserve the user's scope and existing numerical gates.

## Read the right evidence

1. Read repository instructions and the whole named workdoc. If no path is given,
   identify the current task's workdoc; ask only if several plausible documents
   would change the review target.
2. Record the current date, HEAD, branch, and existing differences. Inspect the
   actual CLI, tests, graph call sites, output metadata, and referenced artifacts.
   Read results at their real paths; a report filename alone is not evidence.
3. Distinguish proposal, worker report, reproduced command, measured artifact,
   and untested assertion. Do not convert missing evidence into a pass.

## Check the reasoning that matters

- Goals map to atomic steps, tests, artifacts, and DoD. Required video/state/UI
  behavior cannot be replaced by a smaller image/operator milestone.
- Quality thresholds, data, prompts, preprocessing, and metric definitions were
  fixed before candidate evaluation. A post-run result file is acceptable only
  when its threshold provenance is independently recorded.
- Quantized storage, floating-point reconstruction, actual integer compute,
  backend integration, and application performance are separate claims.
- The baseline and candidate use the same model graph, inputs, resource budget,
  backend/build, and applicable settings. Different precision is disclosed as an
  experimental variable, never hidden under a storage-format name.
- Profiling justifies the target. Timings retain all paired observations and
  separate conversion, setup, warmup, steady execution, and end-to-end costs.
  Shared-load or high-variance runs cannot support a confident speedup claim.
- Memory scope and units are explicit. Process-wide maximum RSS is not assigned
  to one operator without isolation or a justified method.
- Existing backend primitives are reused where suitable, with clear ownership
  and layout contracts. Check unsupported shape/ISA fallback, thread safety,
  buffer reuse/lifetime, and copies. Avoid abstraction or duplication that does
  not improve a concrete use case.
- New numerical error sources, including requantization and implicit internal
  casts or approximate activations, are tested against an independent oracle and
  unchanged model-level gates after graph integration. A graph labeled FP32 must
  be checked at its actual operator implementation, not only its inputs.
- Preserve original license/attribution for copied or adapted material. Do not
  label adapted code solely with the destination repository's license by default.
- Source-publication approval does not authorize private inputs, checkpoint
  redistribution, raw logs, or unrelated changes.

For video reuse/cache work, inspect the exact shared tensor boundary and key
construction. Reused features must be independent of the target prompt or carry
all conditioning dependencies. Check transitive build fingerprints, actually
loaded-library identity, effective thread identities, object-state isolation,
absence/reset and short-sequence boundaries, bounded RAM/disk,
atomic manifests, corruption/invalidation handling, and cold/warm/resumed output
equivalence. Prompt edits must invalidate the affected suffix without silently
changing other objects. Separate raw tracker accuracy from post-run corrections
and count interventions. Disclose causal replay costs when resume rebuilds temporal state. Exportable files
do not imply that a database or live-state serialization API is needed.

## Review and patch

Classify findings as Blocker, Major, or Minor. Cite concrete file/line or artifact
references. Explain how each finding can affect execution or a claimed result.

Default to review-and-fix for unambiguous, scope-preserving improvements. Do not
edit when the user requests review-only. Do not invent measurements, choose a
weaker goal, rewrite historical evidence, or mark an unperformed step complete.
If resolving a finding needs a new product decision, leave it explicit and ask
for that decision. Re-read the patched document before the final verdict.

Report:

1. Verdict: PASS, PASS_WITH_NOTES, REVISE, or FAIL
2. Mode: review-and-fix or review-only
3. Findings ordered by severity, with evidence
4. Applied changes and verification
5. Residual blockers, next evidence needed, and stopping/restart conditions

A plan can pass review while its implementation is unfinished. Say which is
being judged. A completed optimization requires both trustworthy performance
measurement and retained correctness; a faster isolated kernel is not enough.
