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
    folder_path: str = Query(..., description="원본 JSON 파일들이 있는 폴더 경로"),
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
    # 폴더 존재 확인
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"폴더를 찾을 수 없습니다: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"경로가 폴더가 아닙니다: {folder_path}")
    
    # 폴더 내 모든 JSON 파일 찾기
    import glob
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail=f"폴더에 JSON 파일이 없습니다: {folder_path}")
    
    logger.info(f"폴더에서 {len(json_files)}개의 JSON 파일을 발견했습니다: {folder_path}")
    
    # 모든 JSON 파일에서 데이터 수집
    all_news_data = []
    
    try:
        for json_file in json_files:
            logger.info(f"파일 처리 중: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # {"datas": [...]} 형태인지 확인하고 처리
            if isinstance(data, dict) and "datas" in data:
                file_data = data["datas"]
            elif isinstance(data, list):
                file_data = data
            else:
                logger.warning(f"파일 {json_file}의 형태가 올바르지 않습니다. 건너뜁니다.")
                continue
            
            all_news_data.extend(file_data)
            logger.info(f"파일 {json_file}에서 {len(file_data)}개 문서 추가")
        
        news_data = all_news_data
        logger.info(f"총 {len(news_data)}개의 문서를 수집했습니다")
        
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
        file_id = output_name or f"chunks_{os.path.basename(folder_path)}"
        
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
            "folder_path": folder_path,
            "json_files_count": len(json_files),
            "output_name": output_name or f"chunks_{os.path.basename(folder_path)}",
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


