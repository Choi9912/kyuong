from typing import Optional, List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    app_name: str = "embedding-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    # Embedding - ONNX 모델 기반
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # ONNX 변환된 모델명
    require_embedding_model: bool = True
    # ONNX runtime
    use_onnx: bool = True
    onnx_model_path: Optional[str] = "models/all-MiniLM-L6-v2-onnx/model.onnx"  # ONNX 모델 파일 경로
    onnx_tokenizer_id: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"  # 토크나이저용 HuggingFace 모델 ID (폴백)
    onnx_tokenizer_path: Optional[str] = "models/all-MiniLM-L6-v2-onnx"  # 로컬 토크나이저 경로 (우선)
    onnx_pooling: str = "cls"  # cls | mean
    # 출력 차원 및 차원 맞춤
    embedding_output_dim: int = 384
    embedding_reduce: str = "truncate"  # truncate | none


    # Integrations
    admin_log_url: Optional[str] = None  # 베이스 URL (선택)

    # Storage
    output_dir: str = "chunk"

    # Chunk metadata allowlist (for saved chunks_*.json); None means keep all
    chunk_metadata_allowlist: Optional[List[str]] = None

    # Output layout and formats
    output_use_batch_subdir: bool = True
    output_chunks_format: str = "json"  # one of: json, jsonl
    output_embeddings_format: str = "json"  # one of: json, npy

settings = Settings()


