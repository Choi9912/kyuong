from typing import List, Optional
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.clients.storage_client import StorageClient
from app.config.settings import settings
from app.schemas import BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingChunkSchema, QueryEmbeddingRequest, QueryEmbeddingResponse
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
    chunks_folder_path: str = Query(..., description="청크 파일들이 있는 폴더 경로"),
    embeddings_format: str = Query(None, description="임베딩 저장 포맷: json|npy (기본 설정 따름)"),
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
    
    # 폴더 내 모든 JSON 파일 찾기
    json_files = glob.glob(os.path.join(chunks_folder_path, "*.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail=f"폴더에 JSON 파일이 없습니다: {chunks_folder_path}")
    
    logger.info(f"폴더에서 {len(json_files)}개의 JSON 파일을 발견했습니다: {chunks_folder_path}")
    
    # 청크 파일 읽기
    storage_client = StorageClient(base_dir=settings.output_dir)
    file_id = generate_output_name(output_name or os.path.basename(chunks_folder_path), prefix="batch")
    
    all_chunks: List[EmbeddingChunkSchema] = []
    
    try:
        # 모든 JSON 파일에서 청크 데이터 수집
        for json_file in json_files:
            logger.info(f"청크 파일 처리 중: {json_file}")
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
            
            logger.info(f"파일 {json_file}에서 {len(file_chunks_data)}개 청크 추가")
            
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
                
                all_chunks.append(
                    EmbeddingChunkSchema(
                        chunk_id=chunk_data.get("chunk_id") or str(chunk_data.get("id", "unknown")),
                        id=str(chunk_data.get("id", "unknown")),
                        content=content,
                        metadata=chunk_data.get("metadata", {}),
                        row_index=chunk_data.get("row_index"),
                    )
                )
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 읽기 오류: {str(e)}")
    
    logger.info(f"폴더에서 총 {len(all_chunks)}개의 청크를 읽었습니다: {chunks_folder_path}")

    embedder = Embedder(model_name=settings.embedding_model, require=settings.require_embedding_model)

    # 임베딩 생성 (배치 처리)
    start_time = time.time()
    total_chunks = len(all_chunks)
    logger.info(f"임베딩 시작: {total_chunks}개 청크를 배치 크기 {batch_size}로 처리 중...")
    
    all_vectors = []
    texts_to_embed = [c.content for c in all_chunks]
    
    # 배치 단위로 처리
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        
        logger.info(f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch_texts)}개 청크)")
        
        batch_vectors = embedder.embed_texts(batch_texts)
        all_vectors.extend(batch_vectors)
        
        # 진행률 출력
        progress = (i + len(batch_texts)) / len(texts_to_embed) * 100
        logger.info(f"진행률: {progress:.1f}% ({i + len(batch_texts)}/{len(texts_to_embed)})")
    
    vectors = all_vectors
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"임베딩 완료: {total_chunks}개 청크, 소요시간: {elapsed_time:.2f}초")
    logger.info(f"처리 속도: {total_chunks/elapsed_time:.2f} 청크/초")

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
            "source_type": "batch_folder",
            "chunks_folder": chunks_folder_path,
            "json_files_count": len(json_files),
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
            "source_type": "batch_folder",
            "chunks_folder": chunks_folder_path,
            "json_files_count": len(json_files),
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
        chunks_folder=chunks_folder_path,
        json_files_count=len(json_files),
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



