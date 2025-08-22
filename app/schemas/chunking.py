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


class AdvancedChunkingRequest(BaseModel):
    """고급 청킹 요청 (embed_export.py 기능)"""
    documents: List[DocumentSchema]
    mode: str = Field(default="sentence", description="청킹 모드: sentence, library, window")
    
    # 공통 옵션
    id_key: str = Field(default="doc_id", description="문서 ID 필드명")
    title_key: str = Field(default="title", description="제목 필드명")  
    text_key: str = Field(default="text", description="본문 필드명")
    link_key: str = Field(default="link", description="링크 필드명")
    normalize_whitespace: bool = Field(default=True, description="공백 정규화 여부")
    
    # sentence 모드 옵션
    max_sentences: int = Field(default=3, description="청크당 최대 문장 수")
    overlap_sentences: int = Field(default=1, description="청크 간 겹칠 문장 수")
    
    # library 모드 옵션  
    max_tokens: int = Field(default=500, description="청크 최대 길이(토큰)")
    overlap_tokens: int = Field(default=50, description="청크 겹침 길이(토큰)")
    
    # window 모드 옵션
    max_chars: int = Field(default=1200, description="청크 최대 길이(문자)")
    overlap_chars: int = Field(default=200, description="청크 겹침 길이(문자)")


class AdvancedChunkingResponse(BaseModel):
    """고급 청킹 응답"""
    chunks: List[Dict[str, Any]]
    count: int
    mode: str
    total_documents: int