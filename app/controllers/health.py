from fastapi import APIRouter

from app.config.settings import settings


router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.version}


