from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AcousticEvidence:
    bbox_xyxy: tuple[float, float, float, float]
    crop_width: int
    crop_height: int

    object_mean_intensity: float
    object_std_intensity: float

    background_mean_intensity: float
    background_std_intensity: float

    intensity_contrast: float
    edge_density: float
    shape_compactness: float

    shadow_candidate_support: float
    shadow_candidate_direction: str | None
    physical_shadow_direction_available: bool

    evidence_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": [
                round(value, 3)
                for value in self.bbox_xyxy
            ],
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "object_mean_intensity": round(
                self.object_mean_intensity,
                4,
            ),
            "object_std_intensity": round(
                self.object_std_intensity,
                4,
            ),
            "background_mean_intensity": round(
                self.background_mean_intensity,
                4,
            ),
            "background_std_intensity": round(
                self.background_std_intensity,
                4,
            ),
            "intensity_contrast": round(
                self.intensity_contrast,
                6,
            ),
            "edge_density": round(
                self.edge_density,
                6,
            ),
            "shape_compactness": round(
                self.shape_compactness,
                6,
            ),
            "shadow_candidate_support": round(
                self.shadow_candidate_support,
                6,
            ),
            "shadow_candidate_direction": (
                self.shadow_candidate_direction
            ),
            "physical_shadow_direction_available": (
                self.physical_shadow_direction_available
            ),
            "evidence_available": self.evidence_available,
        }
