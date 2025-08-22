from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.clients.storage_client import StorageClient
from app.config.settings import settings
from app.schemas import AdvancedChunkingRequest, AdvancedChunkingResponse
from app.services.chunker import AdvancedChunker
from app.utils.logger import get_logger, log_admin_event


router = APIRouter()
logger = get_logger(__name__)



@router.post("/", response_model=AdvancedChunkingResponse)
def advanced_chunk(
    request: AdvancedChunkingRequest,
    background_tasks: BackgroundTasks,
    save_to_storage: bool = Query(False, description="청크를 로컬 JSON으로 저장할지 여부"),
    output_name: Optional[str] = Query(None, description="출력 파일 식별자"),
) -> AdvancedChunkingResponse:
    """
    고급 청킹 기능 (embed_export.py와 동일)
    
    - sentence: 문장 단위 청킹
    - library: LangChain RecursiveCharacterTextSplitter 기반
    - window: 정확한 문자 길이 기반
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents가 비어있습니다")

    chunker = AdvancedChunker()
    all_chunks = []

    for row_idx, doc in enumerate(request.documents):
        doc_dict = doc.model_dump()
        
        # 필드 추출
        doc_id = doc_dict.get(request.id_key) or doc_dict.get("doc_id") or doc_dict.get("news_id")
        title = doc_dict.get(request.title_key, "")
        text = doc_dict.get(request.text_key, "") or ""
        link = (doc_dict.get("metadata") or {}).get(request.link_key) or doc_dict.get(request.link_key)
        
        if not doc_id:
            raise HTTPException(status_code=400, detail=f"문서 ID 필드 '{request.id_key}'가 없습니다")

        # 모드별 청킹
        try:
            if request.mode == "sentence":
                chunks = chunker.build_sentence_chunks(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    max_sentences=request.max_sentences,
                    overlap_sentences=request.overlap_sentences,
                    row_index=row_idx,
                    link=link,
                    normalize_whitespace=request.normalize_whitespace,
                )
            elif request.mode == "library":
                # 토큰값이 없으면 문자 기준에서 역산
                max_tok = request.max_tokens if request.max_tokens > 0 else max(1, request.max_chars // 4)
                overlap_tok = request.overlap_tokens if request.overlap_tokens > 0 else max(0, request.overlap_chars // 4)
                chunks = chunker.build_library_chunks(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    max_tokens=max_tok,
                    overlap_tokens=overlap_tok,
                    row_index=row_idx,
                    link=link,
                    normalize_whitespace=request.normalize_whitespace,
                )
            elif request.mode == "window":
                chunks = chunker.build_window_chunks(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    max_chars=request.max_chars,
                    overlap_chars=request.overlap_chars,
                    row_index=row_idx,
                    link=link,
                )
            else:
                raise HTTPException(status_code=400, detail=f"지원하지 않는 모드: {request.mode}")
                
        except ImportError as e:
            raise HTTPException(status_code=400, detail=str(e))

        all_chunks.extend(chunks)

    # 저장 옵션
    if save_to_storage:
        storage_client = StorageClient(base_dir=settings.output_dir)
        file_id = output_name or f"advanced_{request.mode}"
        
        background_tasks.add_task(
            storage_client.save_chunks_json,
            file_id,
            all_chunks,
            settings.output_use_batch_subdir,
        )

    # 로그
    background_tasks.add_task(
        log_admin_event,
        event_name="advanced_chunking_completed",
        payload={
            "mode": request.mode,
            "output_name": output_name or f"advanced_{request.mode}",
            "doc_count": len(request.documents),
            "chunk_count": len(all_chunks),
        },
    )

    return AdvancedChunkingResponse(
        chunks=all_chunks,
        count=len(all_chunks), 
        mode=request.mode,
        total_documents=len(request.documents)
    )


