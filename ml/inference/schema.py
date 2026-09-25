from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "bbox_xyxy": [
                round(value, 3)
                for value in self.bbox_xyxy
            ],
        }


@dataclass(frozen=True)
class DetectionResult:
    model_path: str
    model_name: str
    image_width: int
    image_height: int
    inference_ms: float
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_ms": round(self.inference_ms, 3),
            "detection_count": len(self.detections),
            "detections": [
                detection.to_dict()
                for detection in self.detections
            ],
        }
