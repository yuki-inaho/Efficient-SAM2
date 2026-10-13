# Annotation and training engineering records

These records describe reusable engineering behavior. Application categories, capture
identifiers, private repository names, local paths, images, credentials, and experiment
weights are excluded.

- [Repair specification](annotation_quality_repair.md): contracts and acceptance checks.
- [Implementation record](workdoc_generalized.md): current changes and validation scope.
- [Decisions](decisions_generalized.md): identity, validity and publication rules.
- [Training and export validation](training_and_export_validation.md): concrete defects and fixes.
- [Generic experiment workflow](experiment_workflow.md): commands and required inputs.

Dataset-specific adapters and local experiment evidence remain outside this repository.
The generic helpers are under `tools/workflow/`. Neural-network execution requires the
model environment; most regression tests use CPU fixtures without loading model weights.
