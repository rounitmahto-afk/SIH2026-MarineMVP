from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceFusion:
    detector_confidence: float

    intensity_support: float
    edge_support: float
    shape_support: float
    shadow_candidate_support: float

    supporting_signal_count: int

    heuristic_evidence_index: float

    interpretation: str

    shadow_direction_available: bool
    evidence_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_confidence": round(
                self.detector_confidence,
                6,
            ),
            "intensity_support": round(
                self.intensity_support,
                6,
            ),
            "edge_support": round(
                self.edge_support,
                6,
            ),
            "shape_support": round(
                self.shape_support,
                6,
            ),
            "shadow_candidate_support": round(
                self.shadow_candidate_support,
                6,
            ),
            "supporting_signal_count": (
                self.supporting_signal_count
            ),
            "heuristic_evidence_index": round(
                self.heuristic_evidence_index,
                6,
            ),
            "interpretation": self.interpretation,
            "shadow_direction_available": (
                self.shadow_direction_available
            ),
            "evidence_available": self.evidence_available,
        }
