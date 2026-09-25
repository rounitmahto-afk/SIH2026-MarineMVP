from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schema import SSSQualityResult


class SSSQualityAssessor:
    """
    Deterministic quality assessment for side-scan sonar imagery.

    This produces an engineering quality signal from observable image
    properties. It is not a calibrated probability and does not prove
    that an image is physically valid sonar data.

    SSS modality must already be established by the ingestion contract.
    """

    def __init__(
        self,
        low_clip_threshold: int = 3,
        high_clip_threshold: int = 252,
        near_uniform_delta: int = 3,
        minimum_quality_index: float = 0.35,
    ) -> None:
        if not 0 <= low_clip_threshold <= 255:
            raise ValueError("low_clip_threshold must be in [0, 255]")

        if not 0 <= high_clip_threshold <= 255:
            raise ValueError("high_clip_threshold must be in [0, 255]")

        if low_clip_threshold >= high_clip_threshold:
            raise ValueError(
                "low_clip_threshold must be lower than high_clip_threshold"
            )

        if near_uniform_delta < 0:
            raise ValueError("near_uniform_delta must be >= 0")

        if not 0.0 <= minimum_quality_index <= 1.0:
            raise ValueError(
                "minimum_quality_index must be in [0, 1]"
            )

        self.low_clip_threshold = low_clip_threshold
        self.high_clip_threshold = high_clip_threshold
        self.near_uniform_delta = near_uniform_delta
        self.minimum_quality_index = minimum_quality_index

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if image.size == 0:
            raise ValueError("image is empty")

        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            if image.shape[2] == 1:
                gray = image[:, :, 0]
            elif image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError(
                    f"unsupported channel count: {image.shape}"
                )
        else:
            raise ValueError(
                f"unsupported image shape: {image.shape}"
            )

        if gray.dtype != np.uint8:
            gray = cv2.normalize(
                gray,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)

        return gray

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        edges = cv2.Canny(
            gray,
            threshold1=50,
            threshold2=150,
        )

        return float(
            np.count_nonzero(edges) / edges.size
        )

    def assess(self, image: np.ndarray) -> SSSQualityResult:
        gray = self._to_gray(image)

        height, width = gray.shape[:2]

        pixels = gray.astype(np.float32)

        mean_intensity = float(np.mean(pixels))
        std_intensity = float(np.std(pixels))

        p01 = float(np.percentile(pixels, 1))
        p99 = float(np.percentile(pixels, 99))

        dynamic_range = p99 - p01

        low_clip_fraction = float(
            np.mean(gray <= self.low_clip_threshold)
        )

        high_clip_fraction = float(
            np.mean(gray >= self.high_clip_threshold)
        )

        edge_density = self._edge_density(gray)

        near_uniform_fraction = float(
            np.mean(
                np.abs(
                    pixels - mean_intensity
                )
                <= self.near_uniform_delta
            )
        )

        dynamic_score = min(
            1.0,
            max(
                0.0,
                dynamic_range / 100.0,
            ),
        )

        variance_score = min(
            1.0,
            max(
                0.0,
                std_intensity / 50.0,
            ),
        )

        clipping_fraction = (
            low_clip_fraction
            + high_clip_fraction
        )

        clipping_score = 1.0 - min(
            1.0,
            clipping_fraction / 0.20,
        )

        texture_score = min(
            1.0,
            max(
                0.0,
                edge_density / 0.08,
            ),
        )

        uniformity_penalty = min(
            1.0,
            near_uniform_fraction,
        )

        quality_index = (
            0.30 * dynamic_score
            + 0.25 * variance_score
            + 0.20 * clipping_score
            + 0.15 * texture_score
            + 0.10 * (1.0 - uniformity_penalty)
        )

        quality_index = float(
            max(
                0.0,
                min(
                    1.0,
                    quality_index,
                ),
            )
        )

        if near_uniform_fraction >= 0.98:
            status = "unusable"
        elif quality_index < self.minimum_quality_index:
            status = "degraded"
        else:
            status = "good"

        usable = status != "unusable"

        return SSSQualityResult(
            image_width=width,
            image_height=height,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            p01_intensity=p01,
            p99_intensity=p99,
            dynamic_range=dynamic_range,
            low_clip_fraction=low_clip_fraction,
            high_clip_fraction=high_clip_fraction,
            edge_density=edge_density,
            near_uniform_fraction=near_uniform_fraction,
            quality_index=quality_index,
            status=status,
            usable=usable,
        )
