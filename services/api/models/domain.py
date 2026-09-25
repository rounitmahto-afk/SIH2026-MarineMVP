from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from services.api.models.base import Base


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    __table_args__ = (
        CheckConstraint(
            "modality = 'side_scan_sonar'",
            name="ck_ingestion_jobs_modality_sss",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)

    modality: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="side_scan_sonar",
        server_default="side_scan_sonar",
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="queued",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    preprocessing_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    inference_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    evidence_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    tracking_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    frames: Mapped[list["SonarFrame"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class SonarFrame(Base):
    __tablename__ = "sonar_frames"

    __table_args__ = (
        CheckConstraint(
            "modality = 'side_scan_sonar'",
            name="ck_sonar_frames_modality_sss",
        ),
        Index(
            "idx_sonar_frames_location",
            "location",
            postgresql_using="gist",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    job_id: Mapped[int] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )

    source_id: Mapped[str] = mapped_column(String(255), nullable=False)

    modality: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="side_scan_sonar",
        server_default="side_scan_sonar",
    )

    sequence_id: Mapped[str] = mapped_column(String(255), nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)

    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    quality_index: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    quality_status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )
    quality_usable: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    heading_deg: Mapped[float | None] = mapped_column(Float, nullable=True)

    location: Mapped[object | None] = mapped_column(
        Geometry(
            geometry_type="POINT",
            srid=4326,
            spatial_index=False,
        ),
        nullable=True,
    )

    metadata_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    job: Mapped["IngestionJob"] = relationship(
        back_populates="frames",
    )

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="frame",
        cascade="all, delete-orphan",
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    frame_id: Mapped[int] = mapped_column(
        ForeignKey("sonar_frames.id", ondelete="CASCADE"),
        nullable=False,
    )

    class_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    class_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)

    inference_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    frame: Mapped["SonarFrame"] = relationship(
        back_populates="detections",
    )

    evidence: Mapped["AcousticEvidence | None"] = relationship(
        back_populates="detection",
        uselist=False,
        cascade="all, delete-orphan",
    )

    track_id: Mapped[int | None] = mapped_column(
        ForeignKey("tracks.id", ondelete="SET NULL"),
        nullable=True,
    )


class AcousticEvidence(Base):
    __tablename__ = "acoustic_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    detection_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    object_mean_intensity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    object_std_intensity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    background_mean_intensity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    background_std_intensity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    intensity_contrast: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    edge_density: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    shape_compactness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    shadow_candidate_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    shadow_candidate_direction: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    physical_shadow_direction_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    evidence_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    evidence_index: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    interpretation: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    detection: Mapped["Detection"] = relationship(
        back_populates="evidence",
    )


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    sequence_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    class_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    first_frame: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    last_frame: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    detection_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    mean_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    max_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    persistence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
