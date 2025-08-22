"""임베딩 관련 스키마"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmbeddingChunkSchema(BaseModel):
    """임베딩용 청크 스키마"""
    chunk_id: str = Field(..., description="청크 ID")
    id: str = Field(..., description="원본 문서 ID")
    content: str = Field(..., description="임베딩할 텍스트 내용")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    row_index: Optional[int] = Field(default=None, description="NPY 파일에서의 행 인덱스")


class BatchEmbeddingRequest(BaseModel):
    """배치 임베딩 요청 (파일 기반)"""
    chunks_file: str = Field(..., description="청크 파일명 (예: 001)")
    embeddings_format: Optional[str] = Field(default=None, description="저장 포맷: json|npy (기본 설정 따름)")
    output_name: Optional[str] = Field(default=None, description="출력 파일 식별자")


class QueryEmbeddingRequest(BaseModel):
    """쿼리 임베딩 요청 (실시간 검색용)"""
    query: str = Field(..., description="임베딩할 쿼리 텍스트", min_length=1)
    normalize: bool = Field(default=True, description="벡터 정규화 여부")


class BatchEmbeddingResponse(BaseModel):
    """배치 임베딩 응답"""
    chunks: List[EmbeddingChunkSchema]
    count: int
    embeddings_file: Optional[str] = Field(default=None, description="생성된 임베딩 파일")


class QueryEmbeddingResponse(BaseModel):
    """쿼리 임베딩 응답"""
    query: str
    vector: List[float]
    dimension: int
    model: str


