# chunkers/sentence_chunker.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
import re

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?。！？…])\s+')

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # 줄바꿈도 공백으로 통일 후 문장 분리
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r'\s+', ' ', t).strip()
    parts = _SENT_SPLIT_RE.split(t)
    return [p.strip() for p in parts if p.strip()]

def build_chunks(
    doc_id: str,
    title: str,
    text: str,
    max_sentences: int = 3,
    overlap_sentences: int = 1,
    row_index: int = 0,
    link: Optional[str] = None,
    include_chunk_in_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    문장 기반 청킹.
    반환 스키마는 library/window와 동일:
      news_id, chunk_content, chunk_id, metadata{news_id,title,chunk_content,link},
      row_index, chunk_index(1-based), chunk_count, chunk_method="sentence"
    """
    sents = _split_sentences(text or "")
    if not sents:
        return []

    if max_sentences is None or max_sentences <= 0:
        max_sentences = 3
    if overlap_sentences is None or overlap_sentences < 0:
        overlap_sentences = 0

    step = max(max_sentences - overlap_sentences, 1)

    out: List[Dict[str, Any]] = []
    i = 1
    for start in range(0, len(sents), step):
        end = min(start + max_sentences, len(sents))
        chunk_text = " ".join(sents[start:end])

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
            "chunk_id": f"{doc_id}-{i}",
            "metadata": meta,
            "row_index": row_index,
            "chunk_index": i,   # embed_export.py에서 최종 보정되지만 여기서도 채움
            "chunk_count": 0,   # embed_export.py에서 최종 보정
            "chunk_method": "sentence",
        }
        out.append(rec)

        i += 1
        if end >= len(sents):
            break

    # 최종 chunk_count는 embed_export.py에서 다시 덮어씌워짐
    return out
