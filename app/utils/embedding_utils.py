"""임베딩 관련 공통 유틸리티 함수들"""
from datetime import datetime
from typing import Any, Dict, Optional


def generate_output_name(base_name: Optional[str] = None, prefix: str = "embed") -> str:
    """
    통일된 출력 파일명 생성
    
    Args:
        base_name: 사용자가 지정한 파일명 (없으면 자동 생성)
        prefix: 자동 생성 시 사용할 접두사
    
    Returns:
        생성된 파일명
    """
    if base_name:
        return base_name
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}"


def get_utc_timestamp() -> str:
    """UTC 타임스탬프를 ISO 형식으로 반환"""
    return datetime.utcnow().isoformat() + "Z"


def create_base_manifest(
    output_name: str,
    chunks_count: int,
    dimension: int,
    model: str,
    embeddings_format: str = "npy"
) -> Dict[str, Any]:
    """
    기본 매니페스트 구조 생성
    
    Args:
        output_name: 출력 파일명
        chunks_count: 처리된 청크 수
        dimension: 임베딩 차원
        model: 사용된 모델명
        embeddings_format: 임베딩 저장 형식
    
    Returns:
        기본 매니페스트 딕셔너리
    """
    return {
        "output_name": output_name,
        "created_at": get_utc_timestamp(),
        "chunks": chunks_count,
        "dim": dimension,
        "model": model,
        "onnx": True,
        "embeddings_format": embeddings_format,
    }


def create_admin_log_payload(
    event_type: str,
    output_name: str,
    processed_count: int,
    **extra_fields
) -> Dict[str, Any]:
    """
    통일된 어드민 로그 페이로드 생성
    
    Args:
        event_type: 이벤트 타입 (batch, url, query 등)
        output_name: 출력 파일명
        processed_count: 처리된 항목 수
        **extra_fields: 추가 필드들
    
    Returns:
        어드민 로그 페이로드
    """
    payload = {
        "event_type": event_type,
        "output_name": output_name,
        "processed_count": processed_count,
        "timestamp": get_utc_timestamp(),
    }
    payload.update(extra_fields)
    return payload


def get_admin_event_name(event_type: str) -> str:
    """
    통일된 어드민 이벤트명 생성
    
    Args:
        event_type: 이벤트 타입
    
    Returns:
        표준화된 이벤트명
    """
    return f"embedding_{event_type}_completed"
