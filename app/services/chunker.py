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
    
    def process_document(
        self,
        mode: str,
        doc_id: str,
        text: str,
        text_field: str,
        row_index: int,
        metadata: Optional[Dict[str, Any]] = None,
        normalize_whitespace: bool = True,
        # 모드별 파라미터들
        max_sentences: int = 3,
        overlap_sentences: int = 1,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        max_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        문서를 지정된 모드로 청킹 처리
        
        Args:
            mode: 청킹 모드 (sentence, library, window)
            doc_id: 문서 ID
            text: 청킹할 텍스트
            text_field: 텍스트 필드명
            row_index: 문서 인덱스
            metadata: 메타데이터
            normalize_whitespace: 공백 정규화 여부
            기타 모드별 파라미터들...
            
        Returns:
            청크 리스트
            
        Raises:
            ValueError: 지원하지 않는 모드인 경우
            ImportError: 필요한 라이브러리가 없는 경우
        """
        if mode == "sentence":
            return self.build_sentence_chunks(
                doc_id=doc_id,
                text=text,
                text_field=text_field,
                max_sentences=max_sentences,
                overlap_sentences=overlap_sentences,
                row_index=row_index,
                metadata=metadata,
                normalize_whitespace=normalize_whitespace,
            )
        elif mode == "library":
            return self.build_library_chunks(
                doc_id=doc_id,
                text=text,
                text_field=text_field,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                row_index=row_index,
                metadata=metadata,
                normalize_whitespace=normalize_whitespace,
            )
        elif mode == "window":
            return self.build_window_chunks(
                doc_id=doc_id,
                text=text,
                text_field=text_field,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                row_index=row_index,
                metadata=metadata,
                normalize_whitespace=normalize_whitespace,
            )
        else:
            raise ValueError(f"지원하지 않는 모드: {mode}")
    
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
        text: str,
        text_field: str = "text",
        max_sentences: int = 3,
        overlap_sentences: int = 1,
        row_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
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

            # 원본 문서의 모든 필드를 메타데이터로 보존 + 청킹 필드를 청크된 텍스트로 대체
            chunk_metadata = dict(metadata) if metadata else {}
            # 선택한 텍스트 필드가 있다면 청크된 텍스트로 대체 (임베딩용)
            if text_field in chunk_metadata:
                chunk_metadata[text_field] = chunk_text
            
            rec = {
                "id": doc_id,
                "chunk_content": chunk_text,
                "chunk_id": f"{doc_id}-{i}",
                "metadata": chunk_metadata,
                "row_index": row_index,
                "chunk_index": i,
                "chunk_count": len(range(0, len(sents), step)),
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
        text: str,
        text_field: str = "text",
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        row_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
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
            
            # 원본 문서의 모든 필드를 메타데이터로 보존 + 청킹 필드를 청크된 텍스트로 대체
            chunk_metadata = dict(metadata) if metadata else {}
            # 선택한 텍스트 필드가 있다면 청크된 텍스트로 대체 (임베딩용)
            if text_field in chunk_metadata:
                chunk_metadata[text_field] = chunk_text
            
            rec = {
                "id": doc_id,
                "chunk_content": chunk_text,
                "chunk_id": f"{doc_id}-{i}",
                "metadata": chunk_metadata,
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
        text: str,
        text_field: str = "text",
        max_chars: int = 1200,
        overlap_chars: int = 200,
        row_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        normalize_whitespace: bool = True,
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
            if normalize_whitespace:
                chunk_text = self._collapse_whitespace(chunk_text)
                
            # 원본 문서의 모든 필드를 메타데이터로 보존 + 청킹 필드를 청크된 텍스트로 대체
            chunk_metadata = dict(metadata) if metadata else {}
            # 선택한 텍스트 필드가 있다면 청크된 텍스트로 대체 (임베딩용)
            if text_field in chunk_metadata:
                chunk_metadata[text_field] = chunk_text
            
            rec = {
                "id": doc_id,
                "chunk_content": chunk_text,
                "chunk_id": f"{doc_id}-{i}",
                "metadata": chunk_metadata,
                "row_index": row_index,
                "chunk_index": i,
                "chunk_count": total,
                "chunk_method": "window",
            }
            out.append(rec)
        return out


