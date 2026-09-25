from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

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
    preprocessing_ms: float
    evidence_ms: float
    total_ms: float


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
        frame_started = perf_counter()

        preprocessing_started = perf_counter()
        image = self._load_image(path)
        quality = self.quality_assessor.assess(image)
        preprocessing_ms = (
            perf_counter() - preprocessing_started
        ) * 1000.0

        if not quality.usable:
            total_ms = (
                perf_counter() - frame_started
            ) * 1000.0

            return FramePipelineResult(
                image_path=str(path),
                quality=quality,
                detection_result=None,
                evidence=(),
                fusion=(),
                preprocessing_ms=preprocessing_ms,
                evidence_ms=0.0,
                total_ms=total_ms,
            )

        detection_result = self.detector.predict(image)

        evidence_started = perf_counter()
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

        evidence_ms = (
            perf_counter() - evidence_started
        ) * 1000.0

        total_ms = (
            perf_counter() - frame_started
        ) * 1000.0

        return FramePipelineResult(
            image_path=str(path),
            quality=quality,
            detection_result=detection_result,
            evidence=tuple(evidence_items),
            fusion=tuple(fusion_items),
            preprocessing_ms=preprocessing_ms,
            evidence_ms=evidence_ms,
            total_ms=total_ms,
        )
