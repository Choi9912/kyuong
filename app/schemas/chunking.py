"""청킹 관련 스키마"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentSchema(BaseModel):
    """청킹할 문서 스키마"""
    model_config = {"extra": "allow"}  # 추가 필드 허용 (RDBMS 유연성)
    
    id: str = Field(..., description="문서 ID")
    # content 필드는 동적으로 처리 (settings.embedding_text_field 사용)


class ChunkingRequest(BaseModel):
    """청킹 요청"""
    documents: List[DocumentSchema]


class ChunkSchema(BaseModel):
    """청크 결과 스키마"""
    chunk_id: str = Field(..., description="청크 ID")
    document_id: str = Field(..., description="원본 문서 ID")
    content: str = Field(..., description="청크 텍스트 내용")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="청크 메타데이터")
    start_index: Optional[int] = Field(default=None, description="원본에서의 시작 위치")
    end_index: Optional[int] = Field(default=None, description="원본에서의 종료 위치")


class ChunkingResponse(BaseModel):
    """청킹 응답"""
    chunks: List[ChunkSchema]
    count: int
    total_documents: int
