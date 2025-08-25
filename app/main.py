from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.controllers.embedding import router as embedding_router
from app.controllers.chunk import router as chunk_router
from app.controllers.health import router as health_router
from app.utils.logger import configure_logging


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version=settings.version)
    
    # CORS 설정 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 모든 도메인 허용 (개발용)
        allow_credentials=True,
        allow_methods=["*"],  # 모든 HTTP 메서드 허용
        allow_headers=["*"],  # 모든 헤더 허용
    )
    
    app.include_router(health_router)
    app.include_router(chunk_router, prefix="/chunk")
    app.include_router(embedding_router, prefix="/embedding")
    return app


app = create_app()


