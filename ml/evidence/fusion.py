from __future__ import annotations

from .fusion_schema import EvidenceFusion
from .schema import AcousticEvidence


class EvidenceFusionEngine:
    """
    Transparent rule-based evidence summarization.

    This is NOT a calibrated probability model.

    The resulting heuristic_evidence_index is an engineering
    signal used for downstream triage/review ordering. It must
    not be presented as probability of object identity.
    """

    INTENSITY_THRESHOLD = 0.20
    EDGE_THRESHOLD = 0.05
    SHAPE_THRESHOLD = 0.20
    SHADOW_THRESHOLD = 0.10

    def fuse(
        self,
        detector_confidence: float,
        evidence: AcousticEvidence,
    ) -> EvidenceFusion:
        if not 0.0 <= detector_confidence <= 1.0:
            raise ValueError(
                "detector_confidence must be between 0 and 1"
            )

        if not evidence.evidence_available:
            return EvidenceFusion(
                detector_confidence=detector_confidence,
                intensity_support=0.0,
                edge_support=0.0,
                shape_support=0.0,
                shadow_candidate_support=0.0,
                supporting_signal_count=0,
                heuristic_evidence_index=0.0,
                interpretation="evidence_unavailable",
                shadow_direction_available=(
                    evidence.physical_shadow_direction_available
                ),
                evidence_available=False,
            )

        intensity_support = max(
            0.0,
            min(
                1.0,
                evidence.intensity_contrast,
            ),
        )

        edge_support = max(
            0.0,
            min(
                1.0,
                evidence.edge_density,
            ),
        )

        shape_support = max(
            0.0,
            min(
                1.0,
                evidence.shape_compactness,
            ),
        )

        shadow_support = max(
            0.0,
            min(
                1.0,
                evidence.shadow_candidate_support,
            ),
        )

        signals = [
            intensity_support >= self.INTENSITY_THRESHOLD,
            edge_support >= self.EDGE_THRESHOLD,
            shape_support >= self.SHAPE_THRESHOLD,
            shadow_support >= self.SHADOW_THRESHOLD,
        ]

        supporting_signal_count = sum(
            1
            for signal in signals
            if signal
        )

        # Transparent heuristic:
        # detector confidence contributes 50%;
        # four image-derived evidence channels contribute 50%.
        heuristic_index = (
            0.50 * detector_confidence
            + 0.125 * intensity_support
            + 0.125 * edge_support
            + 0.125 * shape_support
            + 0.125 * shadow_support
        )

        if supporting_signal_count >= 3:
            interpretation = "multiple_supporting_signals"
        elif supporting_signal_count >= 1:
            interpretation = "limited_supporting_signals"
        else:
            interpretation = "weak_supporting_signals"

        return EvidenceFusion(
            detector_confidence=detector_confidence,
            intensity_support=intensity_support,
            edge_support=edge_support,
            shape_support=shape_support,
            shadow_candidate_support=shadow_support,
            supporting_signal_count=supporting_signal_count,
            heuristic_evidence_index=float(
                heuristic_index
            ),
            interpretation=interpretation,
            shadow_direction_available=(
                evidence.physical_shadow_direction_available
            ),
            evidence_available=True,
        )
