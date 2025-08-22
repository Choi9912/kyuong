from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.clients.storage_client import StorageClient
from app.config.settings import settings
from app.schemas import BatchEmbeddingRequest, ChunkOnlyResponse, EmbeddingChunkSchema
from app.services.chunker import TextChunker
from app.utils.logger import get_logger, log_admin_event


router = APIRouter()
logger = get_logger(__name__)


@router.post("/", response_model=ChunkOnlyResponse)
def chunk(
    request: BatchEmbeddingRequest,
    background_tasks: BackgroundTasks,
    save_to_storage: bool = Query(False, description="청크를 로컬 JSON으로 저장할지 여부"),
    chunks_format: str = Query(None, description="저장 포맷: json|jsonl (기본 설정 따름)"),
    output_name: Optional[str] = Query(None, description="출력 파일 식별자(예: 001). 지정 시 파일명에 사용"),
) -> ChunkOnlyResponse:
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents가 비어있습니다")

    chunker = TextChunker(
        max_chars=settings.chunk_size,
        overlap_chars=settings.chunk_overlap,
        prefer_sentence_boundary=settings.prefer_sentence_boundary,
        sentence_overlap=settings.sentence_overlap,
    )

    all_chunks: List[EmbeddingChunkSchema] = []
    for doc in request.documents:
        # 동적 필드 매핑
        doc_dict = doc.model_dump()
        text_field = settings.embedding_text_field
        
        if text_field not in doc_dict:
            raise HTTPException(status_code=400, detail=f"임베딩 대상 필드 '{text_field}'가 없습니다")
        
        content_text = doc_dict[text_field]
        
        # metadata 생성 (id와 임베딩 필드 제외)
        exclude_fields = {"id", text_field}
        if settings.metadata_exclude_fields:
            exclude_fields.update(settings.metadata_exclude_fields)
        
        metadata = {k: v for k, v in doc_dict.items() if k not in exclude_fields}
        
        chunks = chunker.chunk_text(content_text)
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc.id}::chunk_{idx:05d}"
            all_chunks.append(
                EmbeddingChunkSchema(
                    id=chunk_id,
                    document_id=doc.id,
                    metadata=metadata,
                )
            )

    storage_client = StorageClient(base_dir=settings.output_dir)
    if save_to_storage:
        file_id = output_name or "batch"
        fmt = (chunks_format or settings.output_chunks_format).lower()
        if fmt == "jsonl":
            background_tasks.add_task(
                storage_client.save_chunks_jsonl,
                file_id,
                (
                    {
                        "id": c.id,
                        "document_id": c.document_id,
                        "metadata": c.metadata or {},
                        "row_index": i,
                    }
                    for i, c in enumerate(all_chunks)
                ),
                settings.output_use_batch_subdir,
            )
        else:
            background_tasks.add_task(
                storage_client.save_chunks_json,
                file_id,
                [
                    {
                        "id": c.id,
                        "document_id": c.document_id,
                        "metadata": c.metadata or {},
                        "row_index": i,
                    }
                    for i, c in enumerate(all_chunks)
                ],
                settings.output_use_batch_subdir,
            )

    background_tasks.add_task(
        log_admin_event,
        event_name="chunking_completed",
        payload={
            "output_name": output_name or "batch",
            "doc_count": len(request.documents),
            "chunk_count": len(all_chunks),
        },
    )

    return ChunkOnlyResponse(chunks=all_chunks, count=len(all_chunks))


