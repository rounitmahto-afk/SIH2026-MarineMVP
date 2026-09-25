from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from sqlalchemy import select

from services.api.config import settings
from services.api.db import SessionLocal
from services.api.models import (
    AcousticEvidence,
    Detection,
    IngestionJob,
    SonarFrame,
)
from services.worker.pipeline import SSSFramePipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_VERSION = "sss-frame-pipeline-v1"

_pipeline: SSSFramePipeline | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_pipeline() -> SSSFramePipeline:
    global _pipeline

    if _pipeline is None:
        _pipeline = SSSFramePipeline()

    return _pipeline


def process_ingestion_job(
    job_id: int,
) -> dict[str, int | float | str]:
    db = SessionLocal()

    try:
        job = db.get(IngestionJob, job_id)

        if job is None:
            raise ValueError(
                f"Ingestion job not found: {job_id}"
            )

        if job.modality != "side_scan_sonar":
            raise ValueError(
                f"Unsupported ingestion modality: {job.modality}"
            )

        if job.status != "queued":
            return {
                "job_id": job.id,
                "status": job.status,
                "frames_processed": 0,
                "detections_total": 0,
            }

        frames = list(
            db.scalars(
                select(SonarFrame)
                .where(SonarFrame.job_id == job.id)
                .order_by(SonarFrame.frame_index)
            )
        )

        if not frames:
            raise ValueError(
                f"Ingestion job {job.id} contains no SSS frames"
            )

        indices = [
            frame.frame_index
            for frame in frames
        ]

        if indices != list(range(len(frames))):
            raise ValueError(
                "SSS frame indices must be contiguous starting at 0"
            )

        if any(
            frame.modality != "side_scan_sonar"
            for frame in frames
        ):
            raise ValueError(
                "Job contains a non-SSS frame"
            )

        job.status = "running"
        job.started_at = _utcnow()
        job.error_message = None
        job.pipeline_version = PIPELINE_VERSION
        job.model_name = settings.model_name
        job.model_version = settings.model_version

        db.commit()

        pipeline = _get_pipeline()

        frames_processed = 0
        detections_total = 0
        total_inference_ms = 0.0
        total_preprocessing_ms = 0.0
        total_evidence_ms = 0.0
        total_processing_ms = 0.0

        for frame in frames:
            image_path = (
                PROJECT_ROOT / frame.image_path
            )

            started = perf_counter()

            result = pipeline.process_frame(
                image_path
            )

            elapsed_ms = (
                perf_counter() - started
            ) * 1000.0

            frame.quality_index = (
                result.quality.quality_index
            )
            frame.quality_status = (
                result.quality.status
            )
            frame.quality_usable = (
                result.quality.usable
            )

            frames_processed += 1
            total_processing_ms += elapsed_ms
            total_preprocessing_ms += (
                result.preprocessing_ms
            )
            total_evidence_ms += (
                result.evidence_ms
            )

            detection_result = (
                result.detection_result
            )

            if detection_result is None:
                continue

            total_inference_ms += (
                detection_result.inference_ms
            )

            detections_total += len(
                detection_result.detections
            )

            for detection_index, detection in enumerate(
                detection_result.detections
            ):
                stored_detection = Detection(
                    frame_id=frame.id,
                    class_name=detection.class_name,
                    class_id=detection.class_id,
                    confidence=detection.confidence,
                    x1=detection.bbox_xyxy[0],
                    y1=detection.bbox_xyxy[1],
                    x2=detection.bbox_xyxy[2],
                    y2=detection.bbox_xyxy[3],
                    inference_ms=(
                        detection_result.inference_ms
                    ),
                    model_name=settings.model_name,
                    model_version=settings.model_version,
                )

                db.add(stored_detection)
                db.flush()

                evidence = (
                    result.evidence[detection_index]
                )
                fusion = (
                    result.fusion[detection_index]
                )

                db.add(
                    AcousticEvidence(
                        detection_id=stored_detection.id,
                        object_mean_intensity=(
                            evidence.object_mean_intensity
                        ),
                        object_std_intensity=(
                            evidence.object_std_intensity
                        ),
                        background_mean_intensity=(
                            evidence.background_mean_intensity
                        ),
                        background_std_intensity=(
                            evidence.background_std_intensity
                        ),
                        intensity_contrast=(
                            evidence.intensity_contrast
                        ),
                        edge_density=(
                            evidence.edge_density
                        ),
                        shape_compactness=(
                            evidence.shape_compactness
                        ),
                        shadow_candidate_score=(
                            evidence.shadow_candidate_support
                        ),
                        shadow_candidate_direction=(
                            evidence.shadow_candidate_direction
                        ),
                        physical_shadow_direction_available=(
                            evidence.physical_shadow_direction_available
                        ),
                        evidence_available=(
                            evidence.evidence_available
                        ),
                        evidence_index=(
                            fusion.heuristic_evidence_index
                        ),
                        interpretation=(
                            fusion.interpretation
                        ),
                    )
                )

        job.inference_ms = (
            total_inference_ms
        )

        job.preprocessing_ms = (
            total_preprocessing_ms
        )
        job.evidence_ms = (
            total_evidence_ms
        )
        job.tracking_ms = None

        job.status = "succeeded"
        job.completed_at = _utcnow()
        job.error_message = None

        db.commit()

        return {
            "job_id": job.id,
            "status": job.status,
            "frames_processed": frames_processed,
            "detections_total": detections_total,
            "inference_ms": round(
                total_inference_ms,
                3,
            ),
            "processing_ms": round(
                total_processing_ms,
                3,
            ),
        }

    except Exception as exc:
        db.rollback()

        failed_job = db.get(
            IngestionJob,
            job_id,
        )

        if failed_job is not None:
            failed_job.status = "failed"
            failed_job.completed_at = _utcnow()
            failed_job.error_message = str(exc)
            db.commit()

        raise

    finally:
        db.close()
