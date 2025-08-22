from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.clients.storage_client import StorageClient
from app.config.settings import settings
from app.schemas import BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingChunkSchema, QueryEmbeddingRequest, QueryEmbeddingResponse
# TextChunker 제거됨
from app.services.embedder import Embedder
from app.utils.logger import get_logger, log_admin_event
from app.utils.embedding_utils import (
    generate_output_name,
    create_base_manifest,
    create_admin_log_payload,
    get_admin_event_name
)


router = APIRouter()
logger = get_logger(__name__)


## chunk 전용 엔드포인트는 app.controllers.chunk로 분리됨


@router.post("/", response_model=BatchEmbeddingResponse)
def embedding(
    background_tasks: BackgroundTasks,
    chunks_file: str = Query(..., description="청크 파일명 (예: legal_batch_001)"),
    embeddings_format: str = Query(None, description="임베딩 저장 포맷: json|npy (기본 설정 따름)"),
    output_name: Optional[str] = Query(None, description="출력 파일 식별자 (기본값: chunks_file과 동일)"),
) -> BatchEmbeddingResponse:
    # 청크 파일 읽기
    storage_client = StorageClient(base_dir=settings.output_dir)
    file_id = generate_output_name(output_name or chunks_file, prefix="batch")
    
    # JSON 또는 JSONL 파일 찾기
    chunks_json_path = storage_client._output_dir(chunks_file, settings.output_use_batch_subdir) / f"chunks_{chunks_file}.json"
    chunks_jsonl_path = storage_client._output_dir(chunks_file, settings.output_use_batch_subdir) / f"chunks_{chunks_file}.jsonl"
    
    all_chunks: List[EmbeddingChunkSchema] = []
    
    if chunks_json_path.exists():
        # JSON 파일 읽기
        import json
        with chunks_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for chunk_data in data["chunks"]:
                # chunk_content 필드에서 임베딩할 텍스트 내용 가져오기
                if "chunk_content" in chunk_data:
                    content = chunk_data["chunk_content"]
                elif "content" in chunk_data:
                    content = chunk_data["content"]
                else:
                    # content가 없으면 에러
                    raise HTTPException(status_code=400, detail="청크 파일에 chunk_content 또는 content 필드가 없습니다.")
                
                all_chunks.append(
                    EmbeddingChunkSchema(
                        chunk_id=chunk_data.get("chunk_id") or str(chunk_data.get("id", "unknown")),
                        id=str(chunk_data.get("id", "unknown")),
                        content=content,
                        metadata=chunk_data.get("metadata", {}),
                        row_index=chunk_data.get("row_index"),
                    )
                )
    elif chunks_jsonl_path.exists():
        # JSONL 파일 읽기
        import json
        with chunks_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                chunk_data = json.loads(line.strip())
                if "chunk_content" in chunk_data:
                    content = chunk_data["chunk_content"]
                elif "content" in chunk_data:
                    content = chunk_data["content"]
                else:
                    raise HTTPException(status_code=400, detail="청크 파일에 chunk_content 또는 content 필드가 없습니다.")
                
                all_chunks.append(
                    EmbeddingChunkSchema(
                        chunk_id=chunk_data.get("chunk_id") or str(chunk_data.get("id", "unknown")),
                        id=str(chunk_data.get("id", "unknown")),
                        content=content,
                        metadata=chunk_data.get("metadata", {}),
                        row_index=chunk_data.get("row_index"),
                    )
                )
    else:
        raise HTTPException(status_code=404, detail=f"청크 파일을 찾을 수 없습니다: {chunks_file}")

    embedder = Embedder(model_name=settings.embedding_model, require=settings.require_embedding_model)

    # 임베딩 생성
    vectors = embedder.embed_texts([c.content for c in all_chunks])

    # 백그라운드 처리: 임베딩 저장, 어드민 로그
    # storage_client는 이미 위에서 초기화됨
    
    # 임베딩 파일 저장 (json: id+vector, npy: 행렬)
    chosen_emb_fmt = (embeddings_format or settings.output_embeddings_format).lower()
    if chosen_emb_fmt == "npy":
        import numpy as np

        mat = np.array(vectors, dtype=np.float32)
        background_tasks.add_task(
            storage_client.save_embeddings_npy,
            file_id,
            mat,
            settings.output_use_batch_subdir,
        )
        # 매니페스트 저장(권장): N, D, 모델/포맷 기록
        manifest_data = create_base_manifest(
            output_name=file_id,
            chunks_count=len(all_chunks),
            dimension=int(mat.shape[1]) if mat.ndim == 2 else 0,
            model=settings.embedding_model,
            embeddings_format="npy"
        )
        manifest_data.update({
            "source_type": "batch_file",
            "chunks_file": chunks_file,
        })
        background_tasks.add_task(
            storage_client.save_manifest,
            file_id,
            manifest_data,
            settings.output_use_batch_subdir,
        )
    else:
        background_tasks.add_task(
            storage_client.save_embeddings_json,
            file_id,
            [
                {
                    "id": c.chunk_id,
                    "vector": vectors[i],
                }
                for i, c in enumerate(all_chunks)
            ],
            settings.output_use_batch_subdir,
        )
        manifest_data = create_base_manifest(
            output_name=file_id,
            chunks_count=len(all_chunks),
            dimension=int(len(vectors[0])) if vectors else 0,
            model=settings.embedding_model,
            embeddings_format="json"
        )
        manifest_data.update({
            "source_type": "batch_file",
            "chunks_file": chunks_file,
        })
        background_tasks.add_task(
            storage_client.save_manifest,
            file_id,
            manifest_data,
            settings.output_use_batch_subdir,
        )

    # 로그 전송
    admin_payload = create_admin_log_payload(
        event_type="batch",
        output_name=file_id,
        processed_count=len(all_chunks),
        chunks_file=chunks_file,
        dimension=int(len(vectors[0])) if vectors else 0
    )
    background_tasks.add_task(
        log_admin_event,
        event_name=get_admin_event_name("batch"),
        payload=admin_payload,
    )

    return BatchEmbeddingResponse(chunks=all_chunks, count=len(all_chunks))


@router.post("/query", response_model=QueryEmbeddingResponse)
def embed_query(request: QueryEmbeddingRequest) -> QueryEmbeddingResponse:
    """
    단일 쿼리 텍스트를 임베딩합니다. (Milvus 검색용)
    
    - 실시간 검색을 위한 빠른 단일 쿼리 임베딩
    - 벡터 정규화 옵션 제공
    - 파일 저장 없이 즉시 벡터 반환
    """
    try:
        # Embedder 초기화
        embedder = Embedder(model_name=settings.embedding_model, require=settings.require_embedding_model)
        
        # 쿼리 임베딩 생성
        vectors = embedder.embed_texts([request.query])
        vector = vectors[0]  # 단일 벡터
        
        # 정규화 옵션 적용
        if request.normalize:
            logger.info("ONNX 모델에서 이미 정규화되어 추가 정규화 생략")
        
        logger.info(f"쿼리 임베딩 완료: '{request.query}' -> {len(vector)}차원 벡터")
        
        return QueryEmbeddingResponse(
            query=request.query,
            vector=vector,
            dimension=len(vector),
            model=settings.embedding_model
        )
        
    except Exception as e:
        logger.error(f"쿼리 임베딩 실패: {e}")
        raise HTTPException(status_code=500, detail=f"쿼리 임베딩 실패: {str(e)}")



