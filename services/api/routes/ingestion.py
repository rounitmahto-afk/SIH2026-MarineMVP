from pathlib import Path
import json
import shutil
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import ValidationError
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from geoalchemy2.elements import WKTElement

from services.api.config import settings
from services.api.db import get_db
from services.api.models import (
    Detection,
    IngestionJob,
    SonarFrame,
)
from services.api.schemas.ingestion import (
    IngestionJobListResponse,
    IngestionJobResponse,
    IngestionJobStatusResponse,
    IngestionJobSummaryResponse,
    SSSIngestionManifest,
)
from services.api.services.ingestion import (
    PROJECT_ROOT,
    STAGING_ROOT,
    UPLOAD_ROOT,
    IngestionValidationError,
    calculate_bundle_checksum,
    canonical_manifest_bytes,
    remove_directory,
    save_and_validate_image,
)


router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post(
    "/jobs",
    response_model=IngestionJobResponse,
    status_code=202,
)
async def create_sss_ingestion_job(
    manifest: UploadFile = File(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> IngestionJobResponse:
    if not manifest.filename:
        raise HTTPException(
            status_code=422,
            detail="SSS manifest filename is required.",
        )

    if manifest.filename.lower() != "manifest.json":
        raise HTTPException(
            status_code=422,
            detail="Manifest must be named manifest.json.",
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=422,
            detail="At least one SSS image file is required.",
        )

    if len(files) > settings.max_files_per_job:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Job contains {len(files)} files, exceeding "
                f"MAX_FILES_PER_JOB={settings.max_files_per_job}."
            ),
        )

    try:
        manifest_bytes = await manifest.read()
        manifest_data = SSSIngestionManifest.model_validate(
            json.loads(manifest_bytes.decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid SSS manifest: {exc}",
        ) from exc

    if manifest_data.modality != "side_scan_sonar":
        raise HTTPException(
            status_code=422,
            detail="Only side_scan_sonar ingestion is supported.",
        )

    expected_files = {
        frame.filename: frame
        for frame in manifest_data.frames
    }

    uploaded_names = [
        Path(upload.filename or "").name
        for upload in files
    ]

    if len(uploaded_names) != len(set(uploaded_names)):
        raise HTTPException(
            status_code=422,
            detail="Uploaded SSS filenames must be unique.",
        )

    if set(uploaded_names) != set(expected_files):
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploaded filenames must exactly match filenames "
                "declared in manifest.json."
            ),
        )

    staging_dir = STAGING_ROOT / str(uuid4())

    records: list[dict] = []
    file_hashes: list[tuple[str, str]] = []
    total_bytes = 0

    try:
        for upload in files:
            filename = Path(upload.filename or "").name

            frame_meta = expected_files[filename]
            destination = staging_dir / filename

            width, height, sha256, total_bytes = (
                await save_and_validate_image(
                    upload=upload,
                    destination=destination,
                    total_bytes=total_bytes,
                )
            )

            records.append(
                {
                    "filename": filename,
                    "frame_index": frame_meta.frame_index,
                    "timestamp": frame_meta.timestamp,
                    "latitude": frame_meta.latitude,
                    "longitude": frame_meta.longitude,
                    "heading_deg": frame_meta.heading_deg,
                    "width": width,
                    "height": height,
                    "sha256": sha256,
                }
            )

            file_hashes.append((filename, sha256))

        source_checksum = calculate_bundle_checksum(
            canonical_manifest_bytes(manifest_data),
            sorted(file_hashes),
        )

        job = IngestionJob(
            source_id=manifest_data.source_id,
            source_checksum=source_checksum,
            pipeline_version="ingestion-v1",
            model_name=settings.model_name,
            model_version=settings.model_version,
            modality="side_scan_sonar",
            status="queued",
        )

        db.add(job)
        db.flush()

        final_dir = UPLOAD_ROOT / f"job_{job.id}"
        final_dir.parent.mkdir(parents=True, exist_ok=True)

        if final_dir.exists():
            raise IngestionValidationError(
                f"Storage directory already exists for job {job.id}."
            )

        staging_dir.rename(final_dir)

        for record in records:
            latitude = record["latitude"]
            longitude = record["longitude"]

            location = None

            if latitude is not None and longitude is not None:
                location = WKTElement(
                    f"POINT({longitude} {latitude})",
                    srid=4326,
                )

            frame = SonarFrame(
                job_id=job.id,
                source_id=manifest_data.source_id,
                modality="side_scan_sonar",
                sequence_id=manifest_data.sequence_id,
                frame_index=record["frame_index"],
                image_path=str(
                    final_dir.joinpath(record["filename"]).relative_to(
                        PROJECT_ROOT
                    )
                ),
                image_sha256=record["sha256"],
                width=record["width"],
                height=record["height"],
                timestamp=record["timestamp"],
                latitude=latitude,
                longitude=longitude,
                heading_deg=record["heading_deg"],
                location=location,
                metadata_verified=False,
            )

            db.add(frame)

        db.commit()

        return IngestionJobResponse(
            id=job.id,
            status=job.status,
            modality=job.modality,
            source_id=job.source_id,
            sequence_id=manifest_data.sequence_id,
            frame_count=len(records),
            source_checksum=source_checksum,
        )

    except IngestionValidationError as exc:
        db.rollback()
        remove_directory(staging_dir)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()
        remove_directory(staging_dir)

        final_dir = (
            UPLOAD_ROOT / f"job_{getattr(locals().get('job', None), 'id', 'unknown')}"
        )
        remove_directory(final_dir)

        raise HTTPException(
            status_code=500,
            detail="SSS ingestion failed.",
        ) from exc


@router.get(
    "/jobs",
    response_model=IngestionJobListResponse,
)
def list_sss_ingestion_jobs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db),
) -> IngestionJobListResponse:
    try:
        total = (
            db.query(
                func.count(IngestionJob.id)
            ).scalar()
            or 0
        )

        rows = (
            db.query(
                IngestionJob.id,
                IngestionJob.status,
                IngestionJob.modality,
                IngestionJob.source_id,
                IngestionJob.pipeline_version,
                IngestionJob.model_name,
                IngestionJob.model_version,
                func.count(
                    func.distinct(SonarFrame.id)
                ).label("frame_count"),
                func.count(
                    func.distinct(SonarFrame.id)
                ).filter(
                    SonarFrame.quality_index.is_not(None)
                ).label("processed_frame_count"),
                func.count(
                    Detection.id
                ).label("detection_count"),
                IngestionJob.created_at,
                IngestionJob.started_at,
                IngestionJob.completed_at,
                IngestionJob.error_message,
            )
            .outerjoin(
                SonarFrame,
                SonarFrame.job_id == IngestionJob.id,
            )
            .outerjoin(
                Detection,
                Detection.frame_id == SonarFrame.id,
            )
            .group_by(
                IngestionJob.id,
                IngestionJob.status,
                IngestionJob.modality,
                IngestionJob.source_id,
                IngestionJob.pipeline_version,
                IngestionJob.model_name,
                IngestionJob.model_version,
                IngestionJob.created_at,
                IngestionJob.started_at,
                IngestionJob.completed_at,
                IngestionJob.error_message,
            )
            .order_by(IngestionJob.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ingestion job status temporarily unavailable.",
        ) from exc

    items = [
        IngestionJobSummaryResponse(
            id=row.id,
            status=row.status,
            modality=row.modality,
            source_id=row.source_id,
            pipeline_version=row.pipeline_version,
            model_name=row.model_name,
            model_version=row.model_version,
            frame_count=row.frame_count,
            processed_frame_count=row.processed_frame_count,
            detection_count=row.detection_count,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
            error_message=row.error_message,
        )
        for row in rows
    ]

    pages = (
        (total + page_size - 1) // page_size
        if total
        else 0
    )

    return IngestionJobListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobStatusResponse,
)
def get_sss_ingestion_job_status(
    job_id: int,
    db: Session = Depends(get_db),
) -> IngestionJobStatusResponse:
    job = db.get(IngestionJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Ingestion job not found.",
        )

    frames = (
        db.query(SonarFrame)
        .filter(SonarFrame.job_id == job.id)
        .order_by(SonarFrame.frame_index)
        .all()
    )

    processed_frame_count = sum(
        1
        for frame in frames
        if frame.quality_index is not None
    )

    detection_count = (
        db.query(Detection)
        .join(
            SonarFrame,
            Detection.frame_id == SonarFrame.id,
        )
        .filter(SonarFrame.job_id == job.id)
        .count()
    )

    sequence_id = (
        frames[0].sequence_id
        if frames
        else None
    )

    return IngestionJobStatusResponse(
        id=job.id,
        status=job.status,
        modality=job.modality,
        source_id=job.source_id,
        source_checksum=job.source_checksum,
        sequence_id=sequence_id,
        pipeline_version=job.pipeline_version,
        model_name=job.model_name,
        model_version=job.model_version,
        frame_count=len(frames),
        processed_frame_count=processed_frame_count,
        detection_count=detection_count,
        started_at=job.started_at,
        completed_at=job.completed_at,
        preprocessing_ms=job.preprocessing_ms,
        inference_ms=job.inference_ms,
        evidence_ms=job.evidence_ms,
        tracking_ms=job.tracking_ms,
        error_message=job.error_message,
    )
