from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.api.config import settings
from services.api.routes.health import router as health_router
from services.api.routes.ingestion import router as ingestion_router


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(ingestion_router)

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"

app.mount(
    "/dashboard/static",
    StaticFiles(directory=WEB_ROOT),
    name="dashboard-static",
)


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "dashboard.html",
        media_type="text/html",
    )
