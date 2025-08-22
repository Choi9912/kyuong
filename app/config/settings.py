from typing import Optional, List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    app_name: str = "embedding-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Embedding - ONNX 모델 기반
    embedding_model: str = "bge-m3-ko-onnx-sent"  # ONNX 변환된 모델명
    require_embedding_model: bool = True
    # ONNX runtime
    use_onnx: bool = True
    onnx_model_path: Optional[str] = "models/bge-m3-ko-onnx-sent/model.onnx"  # ONNX 모델 파일 경로
    onnx_tokenizer_id: Optional[str] = "dragonkue/bge-m3-ko"  # 토크나이저용 HuggingFace 모델 ID (폴백)
    onnx_tokenizer_path: Optional[str] = "models/bge-m3-ko-onnx-sent"  # 로컬 토크나이저 경로 (우선)
    onnx_pooling: str = "cls"  # cls | mean
    # 출력 차원 및 차원 맞춤
    embedding_output_dim: int = 768
    embedding_reduce: str = "truncate"  # truncate | none
    chunk_size: int = 800
    chunk_overlap: int = 100
    prefer_sentence_boundary: bool = True
    sentence_overlap: int = 1
    use_kss: bool = True

    # Integrations
    admin_log_url: Optional[str] = None  # 베이스 URL (선택)

    # Storage
    output_dir: str = "outputs"

    # Chunk metadata allowlist (for saved chunks_*.json); None means keep all
    chunk_metadata_allowlist: Optional[List[str]] = None

    # Output layout and formats
    output_use_batch_subdir: bool = True
    output_chunks_format: str = "json"  # one of: json, jsonl
    output_embeddings_format: str = "json"  # one of: json, npy

    # Dynamic schema mapping
    embedding_text_field: str = "content"  # RDBMS에서 임베딩할 필드명
    metadata_exclude_fields: Optional[List[str]] = None  # metadata에서 제외할 필드들 (id, 임베딩 필드 자동 제외)
    include_chunk_content: bool = False  # 청크 내용을 metadata에 포함할지 여부
    chunk_content_max_length: int = 200  # 청크 내용 포함 시 최대 길이

settings = Settings()


