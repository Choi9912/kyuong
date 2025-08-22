from typing import List, Dict, Any, Optional
import re

# kss는 AdvancedChunker에서 사용하지 않으므로 제거

try:
    # LangChain for library chunking
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None


class AdvancedChunker:
    """고급 청킹 기능 - sentence, library, window 모드 지원"""
    
    def __init__(self):
        self._sent_split_re = re.compile(r'(?<=[.!?。！？…])\s+')
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        if not text:
            return []
        # 줄바꿈도 공백으로 통일 후 문장 분리
        t = text.replace('\r\n', '\n').replace('\r', '\n')
        t = re.sub(r'\s+', ' ', t).strip()
        parts = self._sent_split_re.split(t)
        return [p.strip() for p in parts if p.strip()]
    
    def _n_tokens(self, s: str) -> int:
        """토큰 수 추정 (문자 수 / 4)"""
        return max(1, len(s) // 4)
    
    def _collapse_whitespace(self, s: str) -> str:
        """연속 공백/개행/탭을 단일 space로 정규화"""
        return " ".join((s or "").split())
    
    def build_sentence_chunks(
        self,
        doc_id: str,
        title: str,
        text: str,
        max_sentences: int = 3,
        overlap_sentences: int = 1,
        row_index: int = 0,
        link: Optional[str] = None,
        normalize_whitespace: bool = True,
    ) -> List[Dict[str, Any]]:
        """문장 기반 청킹"""
        sents = self._split_sentences(text or "")
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
            
            if normalize_whitespace:
                chunk_text = self._collapse_whitespace(chunk_text)

            rec = {
                "news_id": doc_id,
                "chunk_content": chunk_text,
                "chunk_id": f"{doc_id}-{i}",
                "metadata": {
                    "news_id": doc_id,
                    "title": title,
                    "chunk_content": chunk_text,
                    "link": link,
                },
                "row_index": row_index,
                "chunk_index": i,
                "chunk_count": len(range(0, len(sents), step)),  # 총 청크 수 계산
                "chunk_method": "sentence",
            }
            out.append(rec)

            i += 1
            if end >= len(sents):
                break

        return out
    
    def build_library_chunks(
        self,
        doc_id: str,
        title: str,
        text: str,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        row_index: int = 0,
        link: Optional[str] = None,
        normalize_whitespace: bool = True,
    ) -> List[Dict[str, Any]]:
        """LangChain RecursiveCharacterTextSplitter 기반 청킹"""
        if RecursiveCharacterTextSplitter is None:
            raise ImportError("langchain이 설치되지 않았습니다. pip install langchain을 실행하세요.")
        
        # RecursiveCharacterTextSplitter 초기화
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            length_function=self._n_tokens,
            separators=["\n\n", "\n", " ", ""],
        )

        pieces = splitter.split_text(text or "")

        out: List[Dict[str, Any]] = []
        total = len(pieces)
        for i, chunk_text in enumerate(pieces, start=1):
            if normalize_whitespace:
                chunk_text = self._collapse_whitespace(chunk_text)
            
            rec = {
                "news_id": doc_id,
                "chunk_content": chunk_text,
                "chunk_id": f"{doc_id}-{i}",
                "metadata": {
                    "news_id": doc_id,
                    "title": title,
                    "chunk_content": chunk_text,
                    "link": link,
                },
                "row_index": row_index,
                "chunk_index": i,
                "chunk_count": total,
                "chunk_method": "library",
            }
            out.append(rec)

        return out
    
    def build_window_chunks(
        self,
        doc_id: str,
        title: str,
        text: str,
        max_chars: int = 1200,
        overlap_chars: int = 200,
        row_index: int = 0,
        link: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """정확한 문자 길이 기반 윈도우 청킹"""
        s = text or ""
        step = max(max_chars - overlap_chars, 1)
        pieces = []
        for start in range(0, len(s), step):
            end = min(start + max_chars, len(s))
            pieces.append(s[start:end])
            if end >= len(s):
                break

        out = []
        total = len(pieces)
        for i, chunk_text in enumerate(pieces, start=1):
            rec = {
                "news_id": doc_id,
                "chunk_content": chunk_text,
                "chunk_id": f"{doc_id}-{i}",
                "metadata": {
                    "news_id": doc_id,
                    "title": title,
                    "chunk_content": chunk_text,
                    "link": link,
                },
                "row_index": row_index,
                "chunk_index": i,
                "chunk_count": total,
                "chunk_method": "window",
            }
            out.append(rec)
        return out


