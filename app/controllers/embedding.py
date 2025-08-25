from typing import List, Optional
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.clients.storage_client import StorageClient
from app.config.settings import settings
from app.schemas import BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingChunkSchema, QueryEmbeddingRequest, QueryEmbeddingResponse
from app.services.embedder import Embedder
from app.utils.logger import get_logger, log_admin_event
from app.utils.embedding_utils import (
    generate_embedding_output_name,
    create_embedding_manifest,
    create_embedding_admin_log_payload,
    get_embedding_admin_event_name
)


router = APIRouter()
logger = get_logger(__name__)


## chunk 전용 엔드포인트는 app.controllers.chunk로 분리됨


@router.post("/", response_model=BatchEmbeddingResponse)
def embedding(
    background_tasks: BackgroundTasks,
    chunks_folder_path: str = Query(..., description="청크 파일들이 있는 폴더 경로"),
    output_name: Optional[str] = Query(None, description="출력 파일 식별자"),
    content_field: str = Query("chunk_content", description="임베딩할 텍스트 필드명 (기본값: chunk_content)"),
    batch_size: int = Query(100, description="배치 크기 (기본값: 100)"),
) -> BatchEmbeddingResponse:
    # 폴더 존재 확인
    import os
    import json
    import glob
    
    if not os.path.exists(chunks_folder_path):
        raise HTTPException(status_code=404, detail=f"폴더를 찾을 수 없습니다: {chunks_folder_path}")
    
    if not os.path.isdir(chunks_folder_path):
        raise HTTPException(status_code=400, detail=f"경로가 폴더가 아닙니다: {chunks_folder_path}")
    
    # 폴더 내 모든 JSON 파일 찾기 (manifest.json 제외)
    json_files = [f for f in glob.glob(os.path.join(chunks_folder_path, "*.json")) 
                  if os.path.basename(f) != "manifest.json"]
    
    if not json_files:
        raise HTTPException(status_code=404, detail=f"폴더에 JSON 파일이 없습니다: {chunks_folder_path}")
    
    logger.info(f"폴더에서 {len(json_files)}개의 JSON 파일을 발견했습니다: {chunks_folder_path}")
    
    # 청크 파일 읽기
    storage_client = StorageClient(base_dir=settings.output_dir)
    
    # 청킹 파일명 기반으로 임베딩 파일명 생성
    if output_name:
        file_id = output_name
    else:
        # 폴더명을 기반으로 생성 (여러 파일 통합)
        folder_name = os.path.basename(chunks_folder_path)
        file_id = f"{folder_name}_embeddings"
    
    # 파일별로 개별 처리
    all_embedding_files = []
    
    # Embedder 초기화
    embedder = Embedder(model_name=settings.embedding_model, require=settings.require_embedding_model)
    
    for json_file in json_files:
        logger.info(f"청크 파일 처리 중: {json_file}")
        
        # 파일별 청크 데이터 수집
        file_chunks: List[EmbeddingChunkSchema] = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # {"chunks": [...]} 형태인지 확인하고 처리
            if isinstance(data, dict) and "chunks" in data:
                file_chunks_data = data["chunks"]
            elif isinstance(data, list):
                file_chunks_data = data
            else:
                logger.warning(f"파일 {json_file}의 형태가 올바르지 않습니다. 건너뜁니다.")
                continue
            
            logger.info(f"파일 {json_file}에서 {len(file_chunks_data)}개 청크 처리")
            
            for chunk_data in file_chunks_data:
                # 지정된 필드에서 임베딩할 텍스트 내용 가져오기
                content = None
                
                # content_field가 metadata 안에 있는지 확인
                if "." in content_field:
                    # metadata.field_name 형태 처리
                    keys = content_field.split(".")
                    current = chunk_data
                    for key in keys:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            current = None
                            break
                    content = current if current is not None else None
                else:
                    # 단순 필드명 또는 metadata 내부 확인
                    if content_field in chunk_data:
                        content = chunk_data[content_field]
                    elif "metadata" in chunk_data and isinstance(chunk_data["metadata"], dict) and content_field in chunk_data["metadata"]:
                        content = chunk_data["metadata"][content_field]
                
                if content is None:
                    raise HTTPException(status_code=400, detail=f"청크 파일에 '{content_field}' 필드가 없습니다.")
                
                # content가 문자열이 아닌 경우 문자열로 변환
                if not isinstance(content, str):
                    content = str(content)
                
                file_chunks.append(
                    EmbeddingChunkSchema(
                        chunk_id=chunk_data.get("chunk_id") or str(chunk_data.get("id", "unknown")),
                        id=str(chunk_data.get("id", "unknown")),
                        content=content,
                        metadata=chunk_data.get("metadata", {}),
                        row_index=chunk_data.get("row_index"),
                    )
                )
            
            # 파일별 임베딩 생성 및 저장
            if file_chunks:
                # 파일명 기반 임베딩 파일명 생성
                base_filename = os.path.splitext(os.path.basename(json_file))[0]
                # chunks_embeddings → embeddings로 단순화
                if base_filename.endswith("_chunks"):
                    base_name = base_filename[:-7]  # "_chunks" 제거
                else:
                    base_name = base_filename
                embedding_file_id = f"{base_name}_embeddings"
                
                # 임베딩 생성
                texts_to_embed = [c.content for c in file_chunks]
                file_vectors = embedder.embed_texts(texts_to_embed)
                
                # NPY 파일 저장
                import numpy as np
                mat = np.array(file_vectors, dtype=np.float32)
                background_tasks.add_task(
                    storage_client.save_embeddings_npy,
                    embedding_file_id,
                    mat,
                    settings.output_use_batch_subdir,
                )
                
                # 매니페스트 저장
                manifest_data = create_embedding_manifest(
                    output_name=embedding_file_id,
                    chunks_count=len(file_chunks),
                    dimension=int(mat.shape[1]) if mat.ndim == 2 else 0,
                    model=settings.embedding_model,
                )
                manifest_data.update({
                    "source_type": "single_file",
                    "source_file": json_file,
                })
                background_tasks.add_task(
                    storage_client.save_manifest,
                    embedding_file_id,
                    manifest_data,
                    settings.output_use_batch_subdir,
                )
                
                all_embedding_files.append({
                    "source_file": json_file,
                    "embedding_file": f"{embedding_file_id}.npy",
                    "chunk_count": len(file_chunks),
                })
                
                logger.info(f"파일 {json_file}의 임베딩을 {embedding_file_id}.npy로 저장했습니다")
            
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON 파싱 오류: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"파일 읽기 오류: {str(e)}")
    
    # 전체 청크 수집 (응답용)
    all_chunks = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "chunks" in data:
                file_chunks_data = data["chunks"]
            elif isinstance(data, list):
                file_chunks_data = data
            else:
                continue
            
            for chunk_data in file_chunks_data:
                content = None
                if "." in content_field:
                    keys = content_field.split(".")
                    current = chunk_data
                    for key in keys:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            current = None
                            break
                    content = current if current is not None else None
                else:
                    if content_field in chunk_data:
                        content = chunk_data[content_field]
                    elif "metadata" in chunk_data and isinstance(chunk_data["metadata"], dict) and content_field in chunk_data["metadata"]:
                        content = chunk_data["metadata"][content_field]
                
                if content is not None:
                    if not isinstance(content, str):
                        content = str(content)
                    
                    all_chunks.append(
                        EmbeddingChunkSchema(
                            chunk_id=chunk_data.get("chunk_id") or str(chunk_data.get("id", "unknown")),
                            id=str(chunk_data.get("id", "unknown")),
                            content=content,
                            metadata=chunk_data.get("metadata", {}),
                            row_index=chunk_data.get("row_index"),
                        )
                    )
        except:
            continue
    
    logger.info(f"폴더에서 총 {len(all_chunks)}개의 청크를 읽었습니다: {chunks_folder_path}")
    logger.info(f"생성된 임베딩 파일: {[f['embedding_file'] for f in all_embedding_files]}")

    # 로그 전송
    admin_payload = create_embedding_admin_log_payload(
        chunks_folder_path=chunks_folder_path,
        json_files_count=len(json_files),
        total_chunks=len(all_chunks),
        output_name="multiple_files",
        embedding_files=all_embedding_files
    )
    background_tasks.add_task(
        log_admin_event,
        event_name=get_embedding_admin_event_name(),
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



