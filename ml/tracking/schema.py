from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrackDetection:
    frame_index: int
    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "detection_index": self.detection_index,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(
                self.confidence,
                6,
            ),
            "bbox_xyxy": [
                round(value, 3)
                for value in self.bbox_xyxy
            ],
        }


@dataclass
class CandidateTrack:
    track_id: int
    first_frame: int
    last_frame: int
    detection_count: int
    class_votes: dict[str, int]
    confidence_sum: float
    max_confidence: float
    detections: list[TrackDetection]

    @property
    def mean_confidence(self) -> float:
        if self.detection_count <= 0:
            return 0.0

        return (
            self.confidence_sum
            / self.detection_count
        )

    @property
    def persistence_score(self) -> float:
        """
        Deterministic bounded persistence signal.

        More repeated detections increase the score, but the
        value is never presented as a probability.
        """
        return min(
            1.0,
            self.detection_count / 5.0,
        )

    @property
    def dominant_class(self) -> str:
        if not self.class_votes:
            return "unknown"

        return max(
            self.class_votes.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
        )[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "detection_count": self.detection_count,
            "class_votes": dict(self.class_votes),
            "dominant_class": self.dominant_class,
            "mean_confidence": round(
                self.mean_confidence,
                6,
            ),
            "max_confidence": round(
                self.max_confidence,
                6,
            ),
            "persistence_score": round(
                self.persistence_score,
                6,
            ),
            "detections": [
                item.to_dict()
                for item in self.detections
            ],
        }
