from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ml.evidence import AcousticEvidenceExtractor, EvidenceFusionEngine
from ml.inference import MarineDetector
from ml.quality import SSSQualityAssessor


@dataclass(frozen=True)
class FramePipelineResult:
    image_path: str
    quality: object
    detection_result: object
    evidence: tuple[object, ...]
    fusion: tuple[object, ...]


class SSSFramePipeline:
    def __init__(
        self,
        detector: MarineDetector | None = None,
        quality_assessor: SSSQualityAssessor | None = None,
        evidence_extractor: AcousticEvidenceExtractor | None = None,
        fusion_engine: EvidenceFusionEngine | None = None,
    ) -> None:
        self.quality_assessor = (
            quality_assessor
            or SSSQualityAssessor()
        )
        self.detector = (
            detector
            or MarineDetector()
        )
        self.evidence_extractor = (
            evidence_extractor
            or AcousticEvidenceExtractor()
        )
        self.fusion_engine = (
            fusion_engine
            or EvidenceFusionEngine()
        )

    @staticmethod
    def _load_image(image_path: str | Path) -> np.ndarray:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"SSS image not found: {path}"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise ValueError(
                f"Could not decode SSS image: {path}"
            )

        return image

    def process_frame(
        self,
        image_path: str | Path,
    ) -> FramePipelineResult:
        path = Path(image_path)

        image = self._load_image(path)

        quality = self.quality_assessor.assess(image)

        if not quality.usable:
            return FramePipelineResult(
                image_path=str(path),
                quality=quality,
                detection_result=None,
                evidence=(),
                fusion=(),
            )

        detection_result = self.detector.predict(image)

        evidence_items = []
        fusion_items = []

        for detection in detection_result.detections:
            evidence = self.evidence_extractor.extract(
                image,
                detection.bbox_xyxy,
            )

            fusion = self.fusion_engine.fuse(
                detection.confidence,
                evidence,
            )

            evidence_items.append(evidence)
            fusion_items.append(fusion)

        return FramePipelineResult(
            image_path=str(path),
            quality=quality,
            detection_result=detection_result,
            evidence=tuple(evidence_items),
            fusion=tuple(fusion_items),
        )
