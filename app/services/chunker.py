from typing import List

try:
    # 선택 사용: 한국어 문장 분리기
    import kss  # type: ignore
except Exception:  # noqa: BLE001
    kss = None


class TextChunker:
    """문장/문단 경계를 우선하는 청킹.

    - kss 사용 가능 시 문장 단위 분리 → 목표 길이까지 병합
    - 없을 경우 기본 문자 길이 기반으로 폴백
    - 겹침은 문장 단위(`sentence_overlap`) 또는 문자 단위(`overlap_chars`) 지원
    """

    def __init__(
        self,
        max_chars: int = 800,
        overlap_chars: int = 100,
        prefer_sentence_boundary: bool = True,
        sentence_overlap: int = 1,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars는 1 이상이어야 합니다")
        if overlap_chars < 0:
            raise ValueError("overlap_chars는 0 이상이어야 합니다")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.prefer_sentence_boundary = prefer_sentence_boundary
        self.sentence_overlap = max(0, sentence_overlap)

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        if self.prefer_sentence_boundary and kss is not None:
            return self._chunk_by_sentences(text)

        return self._chunk_by_chars(text)

    # 내부: 문자 길이 기반 폴백
    def _chunk_by_chars(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.max_chars, length)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == length:
                break
            start = max(0, end - self.overlap_chars)
            if start >= length:
                break
        return chunks

    # 내부: 문장 경계 병합 방식
    def _chunk_by_sentences(self, text: str) -> List[str]:
        sentences = [s.strip() for s in kss.split_sentences(text) if s and s.strip()]  # type: ignore[attr-defined]
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if current:
                chunks.append(" ".join(current).strip())
                if self.sentence_overlap > 0:
                    overlap = current[-self.sentence_overlap :]
                    current = overlap.copy()
                    current_len = sum(len(s) + 1 for s in current)
                else:
                    current = []
                    current_len = 0

        for sent in sentences:
            sent_len = len(sent) + 1
            if current_len + sent_len > self.max_chars and current:
                flush()
            current.append(sent)
            current_len += sent_len
            if current_len >= self.max_chars:
                flush()

        if current:
            chunks.append(" ".join(current).strip())
        return chunks


