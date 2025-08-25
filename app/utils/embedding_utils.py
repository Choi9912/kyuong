"""임베딩 관련 공통 유틸리티 함수들"""
from datetime import datetime
from typing import Any, Dict, Optional


def generate_embedding_output_name(base_name: Optional[str] = None, prefix: str = "embed") -> str:
    """
    임베딩용 출력 파일명 생성
    
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


def create_embedding_manifest(
    output_name: str,
    chunks_count: int,
    dimension: int,
    model: str,
) -> Dict[str, Any]:
    """
    임베딩용 매니페스트 구조 생성
    
    Args:
        output_name: 출력 파일명
        chunks_count: 처리된 청크 수
        dimension: 임베딩 차원
        model: 사용된 모델명
        embeddings_format: 임베딩 저장 형식
    
    Returns:
        임베딩 매니페스트 딕셔너리
    """
    return {
        "output_name": output_name,
        "created_at": get_utc_timestamp(),
        "chunks": chunks_count,
        "dim": dimension,
        "model": model,
        "onnx": True,
        "embeddings_format": "npy",
        "type": "embedding_manifest"
    }


def create_embedding_admin_log_payload(
    chunks_folder_path: str,
    json_files_count: int,
    total_chunks: int,
    output_name: str,
    **extra_fields
) -> Dict[str, Any]:
    """
    임베딩용 어드민 로그 페이로드 생성
    
    Args:
        chunks_folder_path: 청크 폴더 경로
        json_files_count: 처리된 JSON 파일 수
        total_chunks: 총 청크 수
        output_name: 출력 파일명
        **extra_fields: 추가 필드들
    
    Returns:
        임베딩 어드민 로그 페이로드
    """
    payload = {
        "event_type": "embedding_generation",
        "chunks_folder_path": chunks_folder_path,
        "json_files_count": json_files_count,
        "total_chunks": total_chunks,
        "output_name": output_name,
        "timestamp": get_utc_timestamp(),
    }
    payload.update(extra_fields)
    return payload


def get_embedding_admin_event_name() -> str:
    """임베딩용 어드민 이벤트명 반환"""
    return "embedding_generation_completed"
