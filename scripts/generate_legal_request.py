import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


CATEGORIES = [
    "형사",
    "민사",
    "행정",
    "상사",
    "노동",
    "개인정보",
    "전자상거래",
    "지식재산",
]


LEGAL_TOPICS = [
    "개인정보 처리 및 보호에 관한 의무와 책임",
    "전자상거래에서의 청약철회, 환불 및 분쟁조정 절차",
    "노동관계에서의 근로계약, 해고 제한 및 부당해고 구제",
    "지식재산권의 보호 범위와 침해에 대한 손해배상",
    "형사절차에서의 피의자 방어권 및 적법절차의 보장",
    "행정처분의 요건, 절차, 그리고 행정심판 및 소송",
    "상법상 이사의 선관주의의무 및 책임 제한",
    "소비자 분쟁 해결을 위한 집단소송의 요건과 절차",
]


def make_long_body(seed: int, target_chars: int) -> str:
    rng = random.Random(seed)
    segments = []
    while sum(len(s) for s in segments) < target_chars:
        topic = rng.choice(LEGAL_TOPICS)
        clause_num = rng.randint(1, 50)
        article_num = rng.randint(1, 200)
        para = (
            f"제{article_num}조({topic}) 본 조항은 관련 법령의 취지를 반영하여 해석한다. "
            f"다음 각 호의 어느 하나에 해당하는 경우, 당사자는 손해의 발생 및 범위를 입증하여 배상을 청구할 수 있다. "
            f"① 정당한 사유 없는 정보의 제공 거부 ② 고지 의무 위반 ③ 부당한 계약 해지. "
            f"또한 분쟁 발생 시 관할 법원은 당사자의 주장과 증거에 따라 {clause_num}항을 참작하여 판단한다. "
            f"개별 약정이 존재하는 경우 그 약정은 법률의 강행규정에 저촉되지 않는 범위에서 효력을 가진다."
        )
        bullets = []
        for k in range(1, 6):
            bullets.append(
                f"({k}) 당사자는 신의성실의 원칙에 따라 협력하여야 하며, 위반 시 상당한 범위의 손해배상을 부담한다."
            )
        tail = (
            " 본 문서는 교육용 예시 텍스트로서 실제 법률 자문을 대체하지 않으며, "
            "구체적 사안에 대해서는 전문가의 검토가 필요하다."
        )
        block = " ".join([para] + bullets + [tail])
        segments.append(block)
    return "\n\n".join(segments)


def generate_documents(count: int, avg_chars: int) -> list:
    docs = []
    now = datetime.utcnow()
    for i in range(count):
        doc_id = f"law-{i+1:04d}"
        category = CATEGORIES[i % len(CATEGORIES)]
        created_at = now - timedelta(days=30 + i)
        updated_at = created_at + timedelta(days=(i % 10))
        law_name = f"가상의 법령집 제{i%7 + 1}판"
        article_number = f"제{(i % 120) + 1}조"
        article_title = random.choice(LEGAL_TOPICS)

        # 본문은 평균 길이 근처로 생성
        body = make_long_body(seed=i, target_chars=avg_chars)

        docs.append(
            {
                "id": doc_id,
                "content": body,
                "metadata": {
                    "title": f"{law_name} {article_number} ({article_title})",
                    "created_at": created_at.replace(microsecond=0).isoformat() + "Z",
                    "updated_at": updated_at.replace(microsecond=0).isoformat() + "Z",
                    "category": category,
                    "law_name": law_name,
                    "article_number": article_number,
                    "article_title": article_title,
                    "effective_date": (created_at.date().isoformat()),
                    "status": "active",
                    "keywords": [category, "분쟁", "손해배상", "의무", "절차"],
                    "source": "rdbms",
                    "lang": "ko",
                },
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legal domain BatchEmbeddingRequest JSON")
    parser.add_argument("--count", type=int, default=50, help="number of documents")
    parser.add_argument("--batch-id", type=str, default="law_demo_001", help="batch id")
    parser.add_argument(
        "--avg-chars",
        type=int,
        default=1600,
        help="approximate content length to trigger chunking",
    )
    parser.add_argument("--output", type=Path, default=Path("samples/request_legal_batch.json"))

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    docs = generate_documents(count=args.count, avg_chars=args.avg_chars)
    payload = {"batch_id": args.batch_id, "documents": docs}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output} (docs={len(docs)})")


if __name__ == "__main__":
    main()


