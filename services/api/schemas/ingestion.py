from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SSSFrameManifest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    frame_index: int = Field(ge=0)
    timestamp: datetime | None = None
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    heading_deg: float | None = Field(default=None, ge=0.0, lt=360.0)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        from pathlib import Path

        if Path(value).name != value:
            raise ValueError("filename must not contain a directory path")

        return value


class SSSIngestionManifest(BaseModel):
    schema_version: str = Field(default="1", min_length=1, max_length=20)
    modality: Literal["side_scan_sonar"]
    source_id: str = Field(min_length=1, max_length=255)
    sequence_id: str = Field(min_length=1, max_length=255)
    frames: list[SSSFrameManifest] = Field(min_length=1)

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, value: list[SSSFrameManifest]) -> list[SSSFrameManifest]:
        frame_indices = [frame.frame_index for frame in value]
        filenames = [frame.filename for frame in value]

        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("frame_index values must be unique")

        if len(filenames) != len(set(filenames)):
            raise ValueError("frame filenames must be unique")

        return value


class IngestionJobStatusResponse(BaseModel):
    id: int
    status: str
    modality: Literal["side_scan_sonar"]
    source_id: str
    source_checksum: str
    sequence_id: str | None

    pipeline_version: str
    model_name: str
    model_version: str

    frame_count: int
    processed_frame_count: int
    detection_count: int

    started_at: datetime | None
    completed_at: datetime | None

    preprocessing_ms: float | None
    inference_ms: float | None
    evidence_ms: float | None
    tracking_ms: float | None

    error_message: str | None


class IngestionJobResponse(BaseModel):
    id: int
    status: str
    modality: Literal["side_scan_sonar"]
    source_id: str
    sequence_id: str
    frame_count: int
    source_checksum: str
