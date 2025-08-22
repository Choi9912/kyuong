"""청킹 관련 스키마"""
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class AdvancedChunkingRequest(BaseModel):
    """고급 청킹 요청 (파일 기반)"""
    mode: str = Field(default="sentence", description="청킹 모드: sentence, library, window")
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