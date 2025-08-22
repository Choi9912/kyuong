import logging
from typing import Any, Dict, Optional

import requests

from app.config.settings import settings


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging(settings.log_level)
    return logging.getLogger(name)


def log_admin_event(event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """관리자 로깅 API로 비동기 전송 (베이스 URL이 설정된 경우에만)."""
    base = (settings.admin_log_url or "").rstrip("/")
    if not base:
        return
    url = f"{base}/admin/log/{event_name}"
    try:
        requests.post(url, json=payload or {}, timeout=10)
    except Exception:  # noqa: BLE001
        # 로깅 실패는 서비스 흐름을 막지 않음
        pass


