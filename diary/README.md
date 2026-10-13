# Annotation quality review records

This directory contains generalized engineering records for video annotation quality.
Capture identifiers, object categories, private repository names, local paths, images,
and credentials are excluded. Numerical examples in the specification are synthetic
regression fixtures rather than dataset results.

- [Repair specification](annotation_quality_repair.md): data contracts, changes and acceptance checks.
- [Implementation record](workdoc_generalized.md): scope of this publication and validation.
- [Decisions](decisions_generalized.md): identity, validity and publication rules.

The application adapters mentioned in the specification are external workspace integration
points. They are not bundled in this repository; the repository implementation change is
`sam2/sam2_video_predictor.py`. Run its CPU regression suite from the repository root with
`python -m pytest tests/test_propagate_control_flow.py -q` in an environment with pytest.
