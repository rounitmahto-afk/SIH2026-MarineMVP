"""add persisted SSS quality fields

Revision ID: e1a7c2d94f06
Revises: c4e9a1b72d3f
"""

from alembic import op
import sqlalchemy as sa


revision = "e1a7c2d94f06"
down_revision = "c4e9a1b72d3f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sonar_frames",
        sa.Column(
            "quality_index",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "sonar_frames",
        sa.Column(
            "quality_status",
            sa.String(length=20),
            nullable=True,
        ),
    )
    op.add_column(
        "sonar_frames",
        sa.Column(
            "quality_usable",
            sa.Boolean(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "sonar_frames",
        "quality_usable",
    )
    op.drop_column(
        "sonar_frames",
        "quality_status",
    )
    op.drop_column(
        "sonar_frames",
        "quality_index",
    )
