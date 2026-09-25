from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from services.api.db import engine


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, str]:
    try:
        with engine.connect() as conn:
            database_name = conn.execute(
                text("SELECT current_database()")
            ).scalar_one()

            postgis_version = conn.execute(
                text("SELECT PostGIS_Version()")
            ).scalar_one()

        return {
            "status": "ok",
            "database": database_name,
            "postgis": postgis_version,
        }

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database readiness check failed",
        ) from exc
