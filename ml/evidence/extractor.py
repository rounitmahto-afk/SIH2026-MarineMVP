from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schema import AcousticEvidence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AcousticEvidenceExtractor:
    """
    Deterministic image-derived evidence extractor.

    Important:
    - These features are evidence candidates, not physical sonar labels.
    - Shadow direction is unavailable without sonar look-direction metadata.
    - Four-direction shadow-candidate analysis therefore reports only
      relative dark-region support around the detected object.
    """

    def __init__(
        self,
        ring_scale: float = 1.8,
        directional_scale: float = 2.5,
    ) -> None:
        if ring_scale <= 1.0:
            raise ValueError(
                "ring_scale must be greater than 1"
            )

        if directional_scale <= 1.0:
            raise ValueError(
                "directional_scale must be greater than 1"
            )

        self.ring_scale = ring_scale
        self.directional_scale = directional_scale

    @staticmethod
    def _clip_bbox(
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox

        x1_i = max(
            0,
            min(width - 1, int(round(x1))),
        )
        y1_i = max(
            0,
            min(height - 1, int(round(y1))),
        )
        x2_i = max(
            x1_i + 1,
            min(width, int(round(x2))),
        )
        y2_i = max(
            y1_i + 1,
            min(height, int(round(y2))),
        )

        return x1_i, y1_i, x2_i, y2_i

    @staticmethod
    def _safe_mean(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0

        return float(np.mean(values))

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0

        return float(np.std(values))

    def _background_ring(
        self,
        gray: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        pad_x = max(
            2,
            int(round(width * (self.ring_scale - 1.0) / 2.0)),
        )
        pad_y = max(
            2,
            int(round(height * (self.ring_scale - 1.0) / 2.0)),
        )

        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(gray.shape[1], x2 + pad_x)
        ry2 = min(gray.shape[0], y2 + pad_y)

        region = gray[ry1:ry2, rx1:rx2]

        if region.size == 0:
            return np.asarray([], dtype=np.uint8)

        mask = np.ones(
            region.shape,
            dtype=np.uint8,
        )

        inner_x1 = x1 - rx1
        inner_y1 = y1 - ry1
        inner_x2 = x2 - rx1
        inner_y2 = y2 - ry1

        mask[
            max(0, inner_y1):min(region.shape[0], inner_y2),
            max(0, inner_x1):min(region.shape[1], inner_x2),
        ] = 0

        return region[mask.astype(bool)]

    @staticmethod
    def _edge_density(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0

        edges = cv2.Canny(
            crop,
            threshold1=50,
            threshold2=150,
        )

        return float(
            np.count_nonzero(edges) / edges.size
        )

    @staticmethod
    def _shape_compactness(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0

        _, binary = cv2.threshold(
            crop,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return 0.0

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))

        if perimeter <= 0.0:
            return 0.0

        compactness = (
            4.0 * np.pi * area
        ) / (perimeter * perimeter)

        return float(
            max(0.0, min(1.0, compactness))
        )

    def _shadow_candidates(
        self,
        gray: np.ndarray,
        bbox: tuple[int, int, int, int],
        background_mean: float,
        background_std: float,
    ) -> tuple[float, str | None]:
        x1, y1, x2, y2 = bbox

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        extension_x = max(
            2,
            int(round(width * self.directional_scale)),
        )
        extension_y = max(
            2,
            int(round(height * self.directional_scale)),
        )

        candidates: list[tuple[str, float]] = []

        regions = {
            "left": gray[
                y1:y2,
                max(0, x1 - extension_x):x1,
            ],
            "right": gray[
                y1:y2,
                x2:min(gray.shape[1], x2 + extension_x),
            ],
            "up": gray[
                max(0, y1 - extension_y):y1,
                x1:x2,
            ],
            "down": gray[
                y2:min(gray.shape[0], y2 + extension_y),
                x1:x2,
            ],
        }

        denominator = max(
            background_std,
            1.0,
        )

        for direction, region in regions.items():
            if region.size == 0:
                continue

            mean_value = float(
                np.mean(region)
            )

            darkness = max(
                0.0,
                background_mean - mean_value,
            )

            support = darkness / denominator

            candidates.append(
                (
                    direction,
                    float(
                        min(
                            1.0,
                            support / 5.0,
                        )
                    ),
                )
            )

        if not candidates:
            return 0.0, None

        direction, support = max(
            candidates,
            key=lambda item: item[1],
        )

        return support, direction

    def extract(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> AcousticEvidence:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray"
            )

        if image.size == 0:
            raise ValueError("image is empty")

        if image.ndim == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(
                f"unsupported image shape: {image.shape}"
            )

        height, width = gray.shape[:2]

        clipped = self._clip_bbox(
            bbox,
            width,
            height,
        )

        x1, y1, x2, y2 = clipped

        crop = gray[y1:y2, x1:x2]

        background = self._background_ring(
            gray,
            clipped,
        )

        if crop.size == 0 or background.size == 0:
            return AcousticEvidence(
                bbox_xyxy=tuple(
                    float(value)
                    for value in bbox
                ),
                crop_width=int(max(0, x2 - x1)),
                crop_height=int(max(0, y2 - y1)),
                object_mean_intensity=0.0,
                object_std_intensity=0.0,
                background_mean_intensity=0.0,
                background_std_intensity=0.0,
                intensity_contrast=0.0,
                edge_density=0.0,
                shape_compactness=0.0,
                shadow_candidate_support=0.0,
                shadow_candidate_direction=None,
                physical_shadow_direction_available=False,
                evidence_available=False,
            )

        object_mean = self._safe_mean(crop)
        object_std = self._safe_std(crop)

        background_mean = self._safe_mean(background)
        background_std = self._safe_std(background)

        intensity_contrast = abs(
            object_mean - background_mean
        ) / max(background_std, 1.0)

        intensity_contrast = float(
            min(
                1.0,
                intensity_contrast / 5.0,
            )
        )

        edge_density = self._edge_density(crop)

        shape_compactness = self._shape_compactness(
            crop
        )

        shadow_support, shadow_direction = (
            self._shadow_candidates(
                gray,
                clipped,
                background_mean,
                background_std,
            )
        )

        return AcousticEvidence(
            bbox_xyxy=tuple(
                float(value)
                for value in bbox
            ),
            crop_width=int(x2 - x1),
            crop_height=int(y2 - y1),
            object_mean_intensity=object_mean,
            object_std_intensity=object_std,
            background_mean_intensity=background_mean,
            background_std_intensity=background_std,
            intensity_contrast=float(
                min(
                    1.0,
                    intensity_contrast,
                )
            ),
            edge_density=float(
                min(
                    1.0,
                    max(
                        0.0,
                        edge_density,
                    ),
                )
            ),
            shape_compactness=shape_compactness,
            shadow_candidate_support=shadow_support,
            shadow_candidate_direction=shadow_direction,
            physical_shadow_direction_available=False,
            evidence_available=True,
        )
