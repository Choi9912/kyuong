Embedding Service

설치 및 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

아키텍처 개요와 폴더별 역할은 `ARCHITECTURE.md`를 참고하세요.

엔드포인트
- GET /health
- POST /embedding/chunk
- POST /embedding/batch

요청 예시
```json
{
  "batch_id": "crawl_2024_08_08_001",
  "documents": [
    {
      "id": "doc-1",
      "content": "여기에 본문 텍스트",
      "metadata": {"source": "crawl", "url": "https://example.com"}
    }
  ]
}
```

쿼리 파라미터
- save_to_storage: 결과를 outputs/embeddings_<batch>.json 으로 저장

.env 예시
```
ADMIN_LOG_URL=
OUTPUT_DIR=outputs
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=100
LOG_LEVEL=INFO
```

