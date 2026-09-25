from __future__ import annotations

from math import hypot

from .schema import CandidateTrack, TrackDetection


def bbox_iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


class DeterministicTracker:
    """
    Center-distance + IoU association tracker.

    Production ingestion must provide a verified frame_index.
    The tracker never derives sequence from filenames.
    """

    def __init__(
        self,
        max_center_distance: float = 80.0,
        min_iou: float = 0.05,
        max_frame_gap: int = 1,
    ) -> None:
        if max_center_distance <= 0:
            raise ValueError(
                "max_center_distance must be positive"
            )

        if not 0.0 <= min_iou <= 1.0:
            raise ValueError(
                "min_iou must be between 0 and 1"
            )

        if max_frame_gap < 1:
            raise ValueError(
                "max_frame_gap must be >= 1"
            )

        self.max_center_distance = (
            max_center_distance
        )
        self.min_iou = min_iou
        self.max_frame_gap = max_frame_gap

        self._next_track_id = 1
        self._tracks: dict[int, CandidateTrack] = {}

    @staticmethod
    def _association_score(
        track: CandidateTrack,
        detection: TrackDetection,
    ) -> tuple[float, float, float]:
        last_detection = track.detections[-1]

        last_center = last_detection.center()
        current_center = detection.center()

        center_distance = hypot(
            current_center[0] - last_center[0],
            current_center[1] - last_center[1],
        )

        iou = bbox_iou(
            last_detection.bbox_xyxy,
            detection.bbox_xyxy,
        )

        return (
            center_distance,
            -iou,
            float(track.track_id),
        )

    def _candidate_matches(
        self,
        track: CandidateTrack,
        detection: TrackDetection,
    ) -> bool:
        if (
            detection.frame_index
            - track.last_frame
            > self.max_frame_gap
        ):
            return False

        last_detection = track.detections[-1]

        last_center = last_detection.center()
        current_center = detection.center()

        center_distance = hypot(
            current_center[0] - last_center[0],
            current_center[1] - last_center[1],
        )

        iou = bbox_iou(
            last_detection.bbox_xyxy,
            detection.bbox_xyxy,
        )

        return (
            center_distance
            <= self.max_center_distance
            and iou >= self.min_iou
        ) or (
            center_distance
            <= self.max_center_distance / 2.0
            and detection.class_id
            == last_detection.class_id
        )

    def _create_track(
        self,
        detection: TrackDetection,
    ) -> CandidateTrack:
        track = CandidateTrack(
            track_id=self._next_track_id,
            first_frame=detection.frame_index,
            last_frame=detection.frame_index,
            detection_count=1,
            class_votes={
                detection.class_name: 1,
            },
            confidence_sum=detection.confidence,
            max_confidence=detection.confidence,
            detections=[detection],
        )

        self._next_track_id += 1
        self._tracks[track.track_id] = track

        return track

    def update(
        self,
        frame_index: int,
        detections: list[TrackDetection],
    ) -> list[CandidateTrack]:
        if frame_index < 0:
            raise ValueError(
                "frame_index must be >= 0"
            )

        for detection in detections:
            if detection.frame_index != frame_index:
                raise ValueError(
                    "detection frame_index does not "
                    "match update frame_index"
                )

        active_tracks = [
            track
            for track in self._tracks.values()
            if (
                frame_index - track.last_frame
                <= self.max_frame_gap
            )
        ]

        unmatched_tracks = set(
            track.track_id
            for track in active_tracks
        )

        # Process detections in deterministic order:
        # highest confidence first, then detection index.
        ordered_detections = sorted(
            detections,
            key=lambda detection: (
                -detection.confidence,
                detection.detection_index,
            ),
        )

        for detection in ordered_detections:
            candidates = [
                track
                for track in active_tracks
                if track.track_id
                in unmatched_tracks
                and self._candidate_matches(
                    track,
                    detection,
                )
            ]

            if not candidates:
                self._create_track(
                    detection
                )
                continue

            candidates.sort(
                key=lambda track: (
                    self._association_score(
                        track,
                        detection,
                    )
                )
            )

            selected = candidates[0]

            selected.last_frame = frame_index
            selected.detection_count += 1
            selected.confidence_sum += (
                detection.confidence
            )
            selected.max_confidence = max(
                selected.max_confidence,
                detection.confidence,
            )

            selected.class_votes[
                detection.class_name
            ] = (
                selected.class_votes.get(
                    detection.class_name,
                    0,
                )
                + 1
            )

            selected.detections.append(
                detection
            )

            unmatched_tracks.remove(
                selected.track_id
            )

        return [
            self._tracks[track_id]
            for track_id in sorted(
                self._tracks
            )
        ]

    def tracks(self) -> list[CandidateTrack]:
        return [
            self._tracks[track_id]
            for track_id in sorted(
                self._tracks
            )
        ]

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()
