from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from services.api.config import settings
from .schema import Detection, DetectionResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "storage"
    / "runs"
    / "crab_pot_full"
    / "weights"
    / "best.pt"
)

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640


class MarineDetector:
    """
    Reusable YOLO detector.

    The model is loaded once when the detector is constructed.
    No model reload occurs for individual frames.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: str = "cpu",
    ) -> None:
        configured_path = (
            model_path
            or os.getenv("MARINE_MODEL_PATH")
            or settings.model_path
            or DEFAULT_MODEL_PATH
        )

        self.model_path = Path(configured_path)

        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        if confidence <= 0.0 or confidence >= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1"
            )

        if iou <= 0.0 or iou >= 1.0:
            raise ValueError(
                "iou must be between 0 and 1"
            )

        if image_size <= 0:
            raise ValueError(
                "image_size must be positive"
            )

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device

        self.model = YOLO(str(self.model_path))

        names = self.model.names

        if isinstance(names, dict):
            self.class_names = {
                int(class_id): str(name)
                for class_id, name in names.items()
            }
        else:
            self.class_names = {
                class_id: str(name)
                for class_id, name in enumerate(names)
            }

        self.model_name = settings.model_name

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray"
            )

        if image.size == 0:
            raise ValueError("image is empty")

        if image.ndim not in (2, 3):
            raise ValueError(
                f"unsupported image dimensions: {image.shape}"
            )

    def _run_model(self, image: np.ndarray):
        return self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

    def warm_up(self, image: np.ndarray) -> float:
        """
        Warm the loaded model/runtime using a real source image.

        The warm-up result is intentionally discarded and is never
        exposed as a detection.
        """
        self._validate_image(image)

        started = time.perf_counter()

        self._run_model(image)

        return (
            time.perf_counter() - started
        ) * 1000.0

    def predict(
        self,
        image: np.ndarray,
    ) -> DetectionResult:
        self._validate_image(image)

        height, width = image.shape[:2]

        started = time.perf_counter()

        results = self._run_model(image)

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        if not results:
            detections: tuple[Detection, ...] = ()
        else:
            result = results[0]

            if result.boxes is None:
                detections = ()
            else:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = (
                    result.boxes.cls.cpu().numpy().astype(int)
                )

                parsed: list[Detection] = []

                for box, confidence, class_id in zip(
                    xyxy,
                    confidences,
                    class_ids,
                ):
                    class_name = self.class_names.get(
                        int(class_id),
                        f"class_{class_id}",
                    )

                    parsed.append(
                        Detection(
                            class_id=int(class_id),
                            class_name=class_name,
                            confidence=float(confidence),
                            bbox_xyxy=(
                                float(box[0]),
                                float(box[1]),
                                float(box[2]),
                                float(box[3]),
                            ),
                        )
                    )

                detections = tuple(parsed)

        return DetectionResult(
            model_path=str(self.model_path),
            model_name=self.model_name,
            image_width=int(width),
            image_height=int(height),
            inference_ms=float(elapsed_ms),
            detections=detections,
        )


def load_default_detector() -> MarineDetector:
    return MarineDetector()
