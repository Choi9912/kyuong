from typing import List, Optional
import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.clients.storage_client import StorageClient
from app.config.settings import settings
from app.schemas import AdvancedChunkingResponse
from app.services.chunker import AdvancedChunker
from app.utils.logger import get_logger, log_admin_event


router = APIRouter()
logger = get_logger(__name__)



@router.post("/", response_model=AdvancedChunkingResponse)
def chunk_from_file(
    background_tasks: BackgroundTasks,
    file_path: str = Query(..., description="원본 JSON 파일 경로"),
    mode: str = Query("sentence", description="청킹 모드: sentence, library, window"),
    save_to_storage: bool = Query(True, description="청크를 로컬 JSON으로 저장할지 여부"),
    output_name: Optional[str] = Query(None, description="출력 파일 식별자"),
    # 동적 필드 선택
    id_field: str = Query("doc_id", description="문서 ID 필드명 (PK 역할)"),
    text_field: str = Query("text", description="청킹할 텍스트 필드명"),
    # 청킹 옵션
    max_sentences: int = Query(3, description="청크당 최대 문장 수 (sentence 모드)"),
    overlap_sentences: int = Query(1, description="청크 간 겹칠 문장 수 (sentence 모드)"),
    max_tokens: int = Query(500, description="청크 최대 길이(토큰) (library 모드)"),
    overlap_tokens: int = Query(50, description="청크 겹침 길이(토큰) (library 모드)"),
    max_chars: int = Query(1200, description="청크 최대 길이(문자) (window 모드)"),
    overlap_chars: int = Query(200, description="청크 겹침 길이(문자) (window 모드)"),
    normalize_whitespace: bool = Query(True, description="공백 정규화 여부"),
) -> AdvancedChunkingResponse:
    """
    원본 JSON 파일을 읽어서 청킹 수행
    
    크롤링에서 생성된 JSON 파일을 직접 읽어서 청킹합니다.
    """
    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")
    
    try:
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # {"datas": [...]} 형태인지 확인하고 처리
        if isinstance(data, dict) and "datas" in data:
            news_data = data["datas"]
        elif isinstance(data, list):
            news_data = data
        else:
            raise HTTPException(status_code=400, detail="JSON 파일은 배열 형태이거나 {\"datas\": [...]} 형태여야 합니다")
            
        logger.info(f"파일에서 {len(news_data)}개의 뉴스 기사를 읽었습니다: {file_path}")
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 읽기 오류: {str(e)}")

    # 청킹 수행
    chunker = AdvancedChunker()
    all_chunks = []

    for row_idx, doc in enumerate(news_data):
        if not isinstance(doc, dict):
            logger.warning(f"문서 {row_idx}가 딕셔너리가 아닙니다. 건너뜁니다.")
            continue
            
        # 동적 필드 추출
        doc_id = doc.get(id_field) or str(row_idx)
        text_content = doc.get(text_field, "")
        
        if not text_content:
            logger.warning(f"문서 {doc_id}에 '{text_field}' 필드가 없거나 비어있습니다. 건너뜁니다.")
            continue
        
        # 모든 필드를 메타데이터로 보존 (원본 문서의 모든 정보 유지)
        metadata = dict(doc)  # 원본 문서의 모든 필드를 복사

        # 모드별 청킹
        try:
            if mode == "sentence":
                chunks = chunker.build_sentence_chunks(
                    doc_id=doc_id,
                    text=text_content,
                    text_field=text_field,
                    max_sentences=max_sentences,
                    overlap_sentences=overlap_sentences,
                    row_index=row_idx,
                    metadata=metadata,
                    normalize_whitespace=normalize_whitespace,
                )
            elif mode == "library":
                chunks = chunker.build_library_chunks(
                    doc_id=doc_id,
                    text=text_content,
                    text_field=text_field,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    row_index=row_idx,
                    metadata=metadata,
                    normalize_whitespace=normalize_whitespace,
                )
            elif mode == "window":
                chunks = chunker.build_window_chunks(
                    doc_id=doc_id,
                    text=text_content,
                    text_field=text_field,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                    row_index=row_idx,
                    metadata=metadata,
                    normalize_whitespace=normalize_whitespace,

                )
            else:
                raise HTTPException(status_code=400, detail=f"지원하지 않는 모드: {mode}")
                
        except ImportError as e:
            raise HTTPException(status_code=400, detail=str(e))

        all_chunks.extend(chunks)

    # 저장 옵션
    if save_to_storage:
        storage_client = StorageClient(base_dir=settings.output_dir)
        file_id = output_name or f"chunks_{os.path.splitext(os.path.basename(file_path))[0]}"
        
        background_tasks.add_task(
            storage_client.save_chunks_json,
            file_id,
            all_chunks,
            settings.output_use_batch_subdir,
        )

    # 로그
    background_tasks.add_task(
        log_admin_event,
        event_name="file_chunking_completed",
        payload={
            "mode": mode,
            "file_path": file_path,
            "output_name": output_name or f"chunks_{os.path.splitext(os.path.basename(file_path))[0]}",
            "doc_count": len(news_data),
            "chunk_count": len(all_chunks),
        },
    )

    return AdvancedChunkingResponse(
        chunks=all_chunks,
        count=len(all_chunks), 
        mode=mode,
        total_documents=len(news_data)
    )


