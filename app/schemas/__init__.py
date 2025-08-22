"""스키마 패키지 - 기능별 분리"""

# 청킹 관련
from app.schemas.chunking import (
    DocumentSchema,
    ChunkingRequest, 
    ChunkSchema,
    ChunkingResponse
)

# 임베딩 관련  
from app.schemas.embedding import (
    EmbeddingChunkSchema,
    BatchEmbeddingRequest,
    QueryEmbeddingRequest,
    BatchEmbeddingResponse, 
    QueryEmbeddingResponse,
    UrlEmbeddingRequest,
    UrlEmbeddingResponse
)

# 하위 호환성을 위한 별칭 (기존 코드가 깨지지 않도록)
FlexibleDocumentSchema = DocumentSchema  # 기존 이름 유지
ChunkOnlyResponse = ChunkingResponse      # 기존 이름 유지

__all__ = [
    # 청킹
    "DocumentSchema",
    "ChunkingRequest",
    "ChunkSchema", 
    "ChunkingResponse",
    
    # 임베딩
    "EmbeddingChunkSchema",
    "BatchEmbeddingRequest",
    "QueryEmbeddingRequest", 
    "BatchEmbeddingResponse",
    "QueryEmbeddingResponse",
    "UrlEmbeddingRequest",
    "UrlEmbeddingResponse",
    
    # 하위 호환성
    "FlexibleDocumentSchema",
    "ChunkOnlyResponse",
]