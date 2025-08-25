from typing import List, Optional
import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.clients.storage_client import StorageClient
from app.config.settings import settings
from app.schemas import AdvancedChunkingResponse
from app.services.chunker import AdvancedChunker

from app.utils.logger import get_logger, log_admin_event
from app.utils.chunking_utils import create_chunk_admin_log_payload, get_chunk_admin_event_name


router = APIRouter()
logger = get_logger(__name__)

def process_single_file(
    file_data: list,
    json_file: str,
    mode: str,
    id_field: str,
    text_field: str,
    max_sentences: int,
    overlap_sentences: int,
    max_tokens: int,
    overlap_tokens: int,
    max_chars: int,
    overlap_chars: int,
    normalize_whitespace: bool,
    save_to_storage: bool,
    output_name: str = None
) -> list:
    """단일 파일에 대한 청킹 처리"""
    
    chunker = AdvancedChunker()
    file_chunks = []
    
    # 파일명에서 확장자 제거
    base_filename = os.path.splitext(os.path.basename(json_file))[0]
    
    for row_idx, doc in enumerate(file_data):
        if not isinstance(doc, dict):
            logger.warning(f"문서 {row_idx}가 딕셔너리가 아닙니다. 건너뜁니다.")
            continue
            
        # 동적 필드 추출
        doc_id = doc.get(id_field) or str(row_idx)
        text_content = doc.get(text_field)
        
        if not text_content:
            logger.warning(f"문서 {doc_id}에 텍스트 필드 '{text_field}'가 없습니다. 건너뜁니다.")
            continue
        
        # 메타데이터는 전체 문서 정보 유지
        metadata = dict(doc)
        
        try:
            # 청킹 모드별 처리를 chunker 서비스에 위임
            chunks = chunker.process_document(
                mode=mode,
                doc_id=doc_id,
                text=text_content,
                text_field=text_field,
                row_index=row_idx,
                metadata=metadata,
                normalize_whitespace=normalize_whitespace,
                # 모드별 파라미터들
                max_sentences=max_sentences,
                overlap_sentences=overlap_sentences,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
                
        except ImportError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        file_chunks.extend(chunks)
    
    # 파일별로 저장
    if save_to_storage and file_chunks:
        storage_client = StorageClient(base_dir=settings.output_dir)
        chunk_file_id = f"{base_filename}_chunks"
        
        storage_client.save_chunks_json(
            output_name=chunk_file_id,
            chunks=file_chunks,
            use_subdir=False
        )
        logger.info(f"파일 {json_file}의 청크를 {chunk_file_id}.json으로 저장했습니다")
    
    return file_chunks



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
    
    # 파일별로 개별 처리
    all_chunk_files = []
    total_chunks = 0
    
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
            
            logger.info(f"파일 {json_file}에서 {len(file_data)}개 문서 처리 시작")
            
            # 파일별로 청킹 처리
            file_chunks = process_single_file(
                file_data=file_data,
                json_file=json_file,
                mode=mode,
                id_field=id_field,
                text_field=text_field,
                max_sentences=max_sentences,
                overlap_sentences=overlap_sentences,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                normalize_whitespace=normalize_whitespace,
                save_to_storage=save_to_storage,
                output_name=output_name
            )
            
            base_filename = os.path.splitext(os.path.basename(json_file))[0]
            all_chunk_files.append({
                "original_file": json_file,
                "chunk_count": len(file_chunks),
                "chunk_file": f"{base_filename}_chunks.json"
            })
            total_chunks += len(file_chunks)
        
        logger.info(f"총 {len(all_chunk_files)}개 파일에서 {total_chunks}개의 청크를 생성했습니다")
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 읽기 오류: {str(e)}")

    # 응답용 모든 청크 수집 (첫 번째 파일의 청크만 응답에 포함)
    sample_chunks = []
    if all_chunk_files:
        try:
            first_file = all_chunk_files[0]
            base_filename = os.path.splitext(os.path.basename(first_file["original_file"]))[0]
            chunk_file_path = os.path.join(settings.output_dir, f"{base_filename}_chunks.json")
            if os.path.exists(chunk_file_path):
                with open(chunk_file_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    if isinstance(chunk_data, dict) and "chunks" in chunk_data:
                        sample_chunks = chunk_data["chunks"][:10]  # 처음 10개만 샘플로
        except Exception as e:
            logger.warning(f"응답용 청크 파일 읽기 실패: {e}")

    # 로그
    background_tasks.add_task(
        log_admin_event,
        event_name=get_chunk_admin_event_name(),
        payload=create_chunk_admin_log_payload(
            mode=mode,
            folder_path=folder_path,
            json_files_count=len(json_files),
            chunk_files=[f['chunk_file'] for f in all_chunk_files],
            total_chunks=total_chunks,
        ),
    )

    return AdvancedChunkingResponse(
        chunks=sample_chunks,
        count=total_chunks, 
        mode=mode,
        total_documents=len(json_files),
        message=f"파일별 청킹 완료: {len(all_chunk_files)}개 파일 처리됨. 생성된 청크 파일: {[f['chunk_file'] for f in all_chunk_files]}"
    )


