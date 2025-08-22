from fastapi import FastAPI

from app.config.settings import settings
from app.controllers.embedding import router as embedding_router
from app.controllers.chunk import router as chunk_router
from app.controllers.health import router as health_router
from app.utils.logger import configure_logging


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version=settings.version)
    app.include_router(health_router)
    app.include_router(chunk_router, prefix="/chunk")
    app.include_router(embedding_router, prefix="/embedding")
    return app


app = create_app()


