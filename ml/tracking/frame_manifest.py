from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SurveyFrame:
    frame_index: int
    image_path: str
    source_id: str
    sequence_id: str

    latitude: float | None = None
    longitude: float | None = None

    heading_deg: float | None = None

    timestamp_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "image_path": self.image_path,
            "source_id": self.source_id,
            "sequence_id": self.sequence_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading_deg": self.heading_deg,
            "timestamp_utc": self.timestamp_utc,
        }


def validate_frame(frame: SurveyFrame) -> None:
    if frame.frame_index < 0:
        raise ValueError(
            "frame_index must be >= 0"
        )

    if not frame.image_path:
        raise ValueError(
            "image_path is required"
        )

    if not frame.source_id:
        raise ValueError(
            "source_id is required"
        )

    if not frame.sequence_id:
        raise ValueError(
            "sequence_id is required"
        )

    if (
        frame.latitude is not None
        and not -90.0 <= frame.latitude <= 90.0
    ):
        raise ValueError(
            "latitude is outside valid range"
        )

    if (
        frame.longitude is not None
        and not -180.0 <= frame.longitude <= 180.0
    ):
        raise ValueError(
            "longitude is outside valid range"
        )

    if (
        frame.heading_deg is not None
        and not 0.0 <= frame.heading_deg < 360.0
    ):
        raise ValueError(
            "heading_deg must be in [0, 360)"
        )


def validate_sequence(
    frames: list[SurveyFrame],
) -> None:
    if not frames:
        raise ValueError(
            "sequence contains no frames"
        )

    ordered = sorted(
        frames,
        key=lambda frame: frame.frame_index,
    )

    expected_indices = list(
        range(len(ordered))
    )

    actual_indices = [
        frame.frame_index
        for frame in ordered
    ]

    if actual_indices != expected_indices:
        raise ValueError(
            "frame indices must be contiguous "
            "starting at 0"
        )

    sequence_ids = {
        frame.sequence_id
        for frame in ordered
    }

    if len(sequence_ids) != 1:
        raise ValueError(
            "all frames must belong to one sequence"
        )

    for frame in ordered:
        validate_frame(frame)

        if not Path(frame.image_path).exists():
            raise FileNotFoundError(
                f"Frame image does not exist: "
                f"{frame.image_path}"
            )
