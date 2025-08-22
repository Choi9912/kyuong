# library_chunker.py
from typing import List, Dict, Any, Optional
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter

def n_tokens(s: str) -> int:
    """토큰 수 추정 (문자 수 / 4) — LLM 입력 토큰 대략치."""
    return max(1, len(s) // 4)

def _legacy_chunk_id(doc_id: str, index: int) -> str:
    """
    sentence_chunker와 동일 규칙 가정: {doc_id}-{chunk_index}
    필요한 경우 해시 기반으로 바꾸고 싶으면 아래 주석 해제.
    """
    return f"{doc_id}-{index}"

    # --- 해시 기반 사용 시 (과거 호환 불필요하면 위 한 줄을 이 블록으로 교체) ---
    # h = hashlib.sha1(f"{doc_id}:{index}".encode()).hexdigest()[:12]
    # return f"{doc_id}#{h}"

def build_chunks(
    doc_id: str,
    title: str,
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    row_index: int = 0,
    link: Optional[str] = None,
    include_chunk_in_metadata: bool = True,  # sentence_chunker와 동일하게 중복 포함
) -> List[Dict[str, Any]]:
    """
    LangChain RecursiveCharacterTextSplitter(토큰 길이 함수 기반)로 라이브러리 청킹.
    반환 스키마는 sentence_chunker.py와 동일하게 맞춤.

    출력 스키마:
    {
      "news_id": str,
      "chunk_content": str,
      "chunk_id": f"{news_id}-{chunk_index}",
      "metadata": {
        "news_id": str,
        "title": str,
        "chunk_content": str,       # include_chunk_in_metadata=True일 때
        "link": str | None
      },
      "row_index": int,
      "chunk_index": int,            # 1-based
      "chunk_count": int,
      "chunk_method": "library"
    }
    """
    # RecursiveCharacterTextSplitter 초기화 (토큰 길이 함수로 n_tokens 사용)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=n_tokens,
        separators=["\n\n", "\n", " ", ""],
    )

    pieces = splitter.split_text(text or "")
    chunk_count = len(pieces)

    out: List[Dict[str, Any]] = []
    for i, chunk_text in enumerate(pieces, start=1):
        # sentence_chunker와 동일 규칙의 chunk_id
        cid = _legacy_chunk_id(doc_id, i)

        # metadata 구성 (예시처럼 중복 포함)
        meta: Dict[str, Any] = {
            "news_id": doc_id,
            "title": title,
        }
        if include_chunk_in_metadata:
            meta["chunk_content"] = chunk_text
        if link is not None:
            meta["link"] = link

        rec = {
            "news_id": doc_id,
            "chunk_content": chunk_text,
            "chunk_id": cid,
            "metadata": meta,
            "row_index": row_index,
            "chunk_index": i,            # 1-based 인덱스
            "chunk_count": chunk_count,
            "chunk_method": "library",
        }
        out.append(rec)

    return out
