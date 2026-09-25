from fastapi import FastAPI

from services.api.config import settings
from services.api.routes.health import router as health_router
from services.api.routes.ingestion import router as ingestion_router


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(ingestion_router)
