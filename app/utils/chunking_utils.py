"""청킹 관련 공통 유틸리티 함수들"""
from datetime import datetime
from typing import Any, Dict


def generate_chunk_output_name(base_name: str = None, prefix: str = "chunks") -> str:
    """
    청킹용 출력 파일명 생성
    
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


def create_chunk_manifest(
    input_file: str,
    total_chunks: int,
    chunker_settings: dict
) -> dict:
    """청킹용 매니페스트 생성"""
    return {
        "input_file": input_file,
        "total_chunks": total_chunks,
        "chunker_settings": chunker_settings,
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "type": "chunking_manifest"
    }


def create_chunk_admin_log_payload(
    mode: str,
    folder_path: str,
    json_files_count: int,
    chunk_files: list,
    total_chunks: int,
    **extra_fields
) -> Dict[str, Any]:
    """
    청킹용 어드민 로그 페이로드 생성
    
    Args:
        mode: 청킹 모드
        folder_path: 입력 폴더 경로
        json_files_count: 처리된 JSON 파일 수
        chunk_files: 생성된 청크 파일 목록
        total_chunks: 총 청크 수
        **extra_fields: 추가 필드들
    
    Returns:
        어드민 로그 페이로드
    """
    payload = {
        "event_type": "file_chunking",
        "mode": mode,
        "folder_path": folder_path,
        "json_files_count": json_files_count,
        "chunk_files": chunk_files,
        "total_chunks": total_chunks,
        "timestamp": datetime.now().isoformat(),
    }
    payload.update(extra_fields)
    return payload


def get_chunk_admin_event_name() -> str:
    """청킹용 어드민 이벤트명 반환"""
    return "file_chunking_completed"
