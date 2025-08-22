#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Optional

from chunkers import sentence_chunker, library_chunker


def parse_args():
    ap = argparse.ArgumentParser(description="Make embedding-ready JSON from doc JSON")
    ap.add_argument("--input", "-i", required=True, help="Input docs.json (array of {doc_id,title,text,metadata})")
    ap.add_argument("--output", "-o", required=True, help="Output embedding.json")

    ap.add_argument(
        "--mode",
        choices=["sentence", "library", "window"],
        default="sentence",
        help="sentence: 문장 단위, library: 토큰 기반 윈도우, window: 정확한 문자 길이 윈도우",
    )

    # window(문자 기준) 옵션
    ap.add_argument("--max-chars", type=int, default=1200, help="청크 최대 길이(문자)")
    ap.add_argument("--overlap-chars", type=int, default=200, help="청크 겹침 길이(문자)")

    # 공통 키
    ap.add_argument("--id-key", default="doc_id", help="문서 ID 필드명(예: news_id)")
    ap.add_argument("--title-key", default="title", help="제목 필드명")
    ap.add_argument("--text-key", default="text", help="본문 필드명")
    ap.add_argument("--row-start", type=int, default=0, help="row_index 시작값 (0 또는 1 권장)")
    ap.add_argument("--link-key", default="link", help="링크 키 이름 (metadata.link 또는 최상위 link)")

    # sentence 옵션
    ap.add_argument("--max-sentences", type=int, default=0, help="청크당 최대 문장 수(>0이면 문장 기준 패킹)")
    ap.add_argument("--overlap-sentences", type=int, default=0, help="청크 간 겹칠 문장 수")

    # library(토큰) 옵션
    ap.add_argument("--max-tokens", type=int, default=0, help="청크 최대 길이(토큰)")
    ap.add_argument("--overlap-tokens", type=int, default=0, help="청크 겹침 길이(토큰)")

    # 공백/개행 정규화 토글 (기본 ON)
    ap.add_argument('--normalize-whitespace', dest='normalize_ws', action='store_true',
                    help='공백/개행/탭을 단일 space로 정규화')
    ap.add_argument('--keep-whitespace', dest='normalize_ws', action='store_false',
                    help='원본 개행/공백 유지')
    ap.set_defaults(normalize_ws=True)

    args = ap.parse_args()
    args.output_method = args.mode  # 기록용(출력 정책 분기)
    return args


def _collapse_whitespace(s: str) -> str:
    """연속 공백/개행/탭을 단일 space로 정규화."""
    return " ".join((s or "").split())


def _build_char_windows(
    doc_id: str,
    title: str,
    text: str,
    max_chars: int,
    overlap_chars: int,
    row_index: int,
    link: Optional[str],
):
    """정확한 문자 길이 기반 윈도우 청킹 — sentence/library와 동일 스키마."""
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
            # window 모드에서는 기존처럼 인덱스/카운트 유지
            "chunk_index": i,
            "chunk_count": total,
            "chunk_method": "window",
        }
        out.append(rec)
    return out


def main():
    args = parse_args()

    # 입력 로드 (배열/단일 모두 지원)
    docs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(docs, dict):
        docs = [docs]

    out = []
    for row_idx, doc in enumerate(docs, start=args.row_start):
        doc_id = doc.get(args.id_key) or doc.get("doc_id") or doc.get("news_id")
        title = doc.get(args.title_key, "")
        text = doc.get(args.text_key, "") or ""
        link = (doc.get("metadata") or {}).get(args.link_key) or doc.get(args.link_key)

        # 모드별 청킹
        if args.mode == "sentence":
            chunks = sentence_chunker.build_chunks(
                doc_id=doc_id,
                title=title,
                text=text,
                max_sentences=args.max_sentences or 0,
                overlap_sentences=args.overlap_sentences or 0,
                row_index=row_idx,
                link=link,
            )
        elif args.mode == "library":
            # 토큰값이 없으면 문자 기준에서 역산(≈ /4)
            tok = args.max_tokens if args.max_tokens > 0 else max(1, args.max_chars // 4)
            ovt = args.overlap_tokens if args.overlap_tokens > 0 else max(0, args.overlap_chars // 4)
            chunks = library_chunker.build_chunks(
                doc_id=doc_id,
                title=title,
                text=text,
                max_tokens=tok,
                overlap_tokens=ovt,
                row_index=row_idx,
                link=link,
                include_chunk_in_metadata=True,
            )
        else:  # window
            chunks = _build_char_windows(
                doc_id=doc_id,
                title=title,
                text=text,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
                row_index=row_idx,
                link=link,
            )

        # 공통 후처리
        total = len(chunks)
        for i, rec in enumerate(chunks, start=1):
            # 1) 공백 정규화 (sentence/library에만 기본 적용; 토글 가능)
            if args.normalize_ws and args.output_method in ("sentence", "library"):
                rec["chunk_content"] = _collapse_whitespace(rec.get("chunk_content", ""))
                md = rec.get("metadata")
                if isinstance(md, dict) and "chunk_content" in md:
                    md["chunk_content"] = _collapse_whitespace(md.get("chunk_content", ""))

            # 2) 필드 제거 정책
            if args.output_method in ("sentence", "library"):
                # 요청대로: 이 3개 필드를 완전히 제거
                rec.pop("chunk_index", None)
                rec.pop("chunk_count", None)
                rec.pop("chunk_method", None)
            else:
                # window 모드는 기존 유지/보정
                rec["chunk_index"] = i
                rec["chunk_count"] = total
                rec["chunk_method"] = "window"

            out.append(rec)

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.output} ({len(out)} records) mode={args.output_method}, normalize_ws={args.normalize_ws}")


if __name__ == "__main__":
    main()
