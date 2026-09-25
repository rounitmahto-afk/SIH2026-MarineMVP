"""create core SSS marine detection schema

Revision ID: c4e9a1b72d3f
Revises:
Create Date: 2026-09-25

"""

from alembic import op
import sqlalchemy as sa
import geoalchemy2


revision = "c4e9a1b72d3f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("source_checksum", sa.String(length=128), nullable=False),
        sa.Column("pipeline_version", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column(
            "modality",
            sa.String(length=50),
            nullable=False,
            server_default=sa.text("'side_scan_sonar'"),
        ),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("preprocessing_ms", sa.Float(), nullable=True),
        sa.Column("inference_ms", sa.Float(), nullable=True),
        sa.Column("evidence_ms", sa.Float(), nullable=True),
        sa.Column("tracking_ms", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "modality = 'side_scan_sonar'",
            name="ck_ingestion_jobs_modality_sss",
        ),
    )

    op.create_table(
        "tracks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sequence_id", sa.String(length=255), nullable=False),
        sa.Column("class_name", sa.String(length=100), nullable=False),
        sa.Column("first_frame", sa.Integer(), nullable=False),
        sa.Column("last_frame", sa.Integer(), nullable=False),
        sa.Column("detection_count", sa.Integer(), nullable=False),
        sa.Column("mean_confidence", sa.Float(), nullable=False),
        sa.Column("max_confidence", sa.Float(), nullable=False),
        sa.Column("persistence_score", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "sonar_frames",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column(
            "modality",
            sa.String(length=50),
            nullable=False,
            server_default=sa.text("'side_scan_sonar'"),
        ),
        sa.Column("sequence_id", sa.String(length=255), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("heading_deg", sa.Float(), nullable=True),
        sa.Column(
            "location",
            geoalchemy2.types.Geometry(
                geometry_type="POINT",
                srid=4326,
                dimension=2,
                spatial_index=False,
                from_text="ST_GeomFromEWKT",
                name="geometry",
            ),
            nullable=True,
        ),
        sa.Column("metadata_verified", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "modality = 'side_scan_sonar'",
            name="ck_sonar_frames_modality_sss",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["ingestion_jobs.id"],
            ondelete="CASCADE",
        ),
    )

    op.create_index(
        "idx_sonar_frames_location",
        "sonar_frames",
        ["location"],
        unique=False,
        postgresql_using="gist",
    )

    op.create_table(
        "detections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("frame_id", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(length=100), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("inference_ms", sa.Float(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["frame_id"],
            ["sonar_frames.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["track_id"],
            ["tracks.id"],
            ondelete="SET NULL",
        ),
    )

    op.create_table(
        "acoustic_evidence",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("detection_id", sa.Integer(), nullable=False, unique=True),
        sa.Column("object_mean_intensity", sa.Float(), nullable=True),
        sa.Column("object_std_intensity", sa.Float(), nullable=True),
        sa.Column("background_mean_intensity", sa.Float(), nullable=True),
        sa.Column("background_std_intensity", sa.Float(), nullable=True),
        sa.Column("intensity_contrast", sa.Float(), nullable=True),
        sa.Column("edge_density", sa.Float(), nullable=True),
        sa.Column("shape_compactness", sa.Float(), nullable=True),
        sa.Column("shadow_candidate_score", sa.Float(), nullable=True),
        sa.Column("shadow_candidate_direction", sa.String(length=30), nullable=True),
        sa.Column(
            "physical_shadow_direction_available",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column("evidence_available", sa.Boolean(), nullable=False),
        sa.Column("evidence_index", sa.Float(), nullable=True),
        sa.Column("interpretation", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(
            ["detection_id"],
            ["detections.id"],
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    op.drop_table("acoustic_evidence")
    op.drop_table("detections")
    op.drop_index(
        "idx_sonar_frames_location",
        table_name="sonar_frames",
        postgresql_using="gist",
    )
    op.drop_table("sonar_frames")
    op.drop_table("tracks")
    op.drop_table("ingestion_jobs")
