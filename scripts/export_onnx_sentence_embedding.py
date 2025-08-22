"""
Sentence-Transformers 전체 파이프라인(Transformer + Pooling + Normalize)을 포함한 ONNX 내보내기.

권장: 런타임에서의 풀링/정규화 분기 없이 곧바로 `sentence_embedding`(B, D) 출력 사용.

사용 예 (PowerShell):
  python scripts/export_onnx_sentence_embedding.py \
    --model-id dragonkue/bge-m3-ko \
    --output-dir models/bge-m3-ko-onnx-sent \
    --pooling cls \
    --opset 17

출력:
  - models/bge-m3-ko-onnx-sent/model.onnx (출력 이름: sentence_embedding)
  - 같은 폴더에 토크나이저 파일 저장
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SentenceEmbeddingWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, pooling: str = "cls") -> None:
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling.lower()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # HF 모델의 last_hidden_state: (B, L, H)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            # 패딩 제외 평균 풀링
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            # 기본: CLS 토큰
            pooled = last_hidden[:, 0, :]
        # L2 정규화
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled


def export(model_id: str, output_dir: Path, pooling: str, opset: int, max_length: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModel.from_pretrained(model_id)
    base.eval()

    wrapper = SentenceEmbeddingWrapper(base, pooling=pooling).eval()

    dummy = tokenizer(["hello"], padding=True, truncation=True, max_length=min(max_length, 512), return_tensors="pt")
    input_names = ["input_ids", "attention_mask"]
    output_names = ["sentence_embedding"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "sentence_embedding": {0: "batch"},
    }

    onnx_path = output_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=(dummy["input_ids"], dummy["attention_mask"]),
            f=str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    # 토크나이저 저장 (런타임에서 동일 토크나이저 사용 보장)
    tokenizer.save_pretrained(output_dir)
    print(f"Exported sentence-level ONNX to: {onnx_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="HF model id, e.g., dragonkue/bge-m3-ko")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--max-length", type=int, default=8192)
    args = p.parse_args()

    export(args.model_id, args.output_dir, args.pooling, args.opset, args.max_length)


if __name__ == "__main__":
    main()


