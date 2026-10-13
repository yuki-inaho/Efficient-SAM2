# Annotation quality decisions

1. Use a stable clip-local identifier for each physical object. A new prompt is not automatically a new object.
2. Keep raw masks separate from masks filtered by calibrated depth. Unknown depth does not establish background.
3. Convert depth to metres before applying an extrinsic transform whose translation is in metres. Reject projections outside the image.
4. Keep unannotated regions distinct from verified background and verified absence.
5. PNG masks remain authoritative. Polygon representations must declare approximation and topology limitations.
6. Report agreement with generated reference labels separately from accuracy against independently verified ground truth.
7. Validate non-prompted frames and prompt boundaries before accepting a label sequence for training.
8. Preserve old artifacts locally when regenerating outputs. Record code, model, settings and input hashes in local run manifests.
9. Published records omit application categories, capture identifiers, private paths and raw artifacts.
10. A successful training exit code is not evidence that labels, identity continuity or exported UI data are correct.
