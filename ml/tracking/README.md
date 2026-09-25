# Deterministic Candidate Tracking

The tracker associates detections using:

- center-distance
- bounding-box IoU
- class consistency
- bounded frame gap

The production ingestion pipeline must provide a verified
`frame_index`.

The tracker does not infer frame order from filenames.

Track state includes:

- first_frame
- last_frame
- detection_count
- class_votes
- mean_confidence
- max_confidence
- persistence_score

`persistence_score` is an engineering signal, not a probability.

A candidate with only one detection remains stored with a low
persistence score.
