import hashlib
from pathlib import Path
from typing import Any, List, Optional, Dict

import numpy as np

from app.config.settings import settings


def _resolve_onnx_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".onnx":
        return p
    if p.is_dir():
        candidate = p / "model.onnx"
        if candidate.exists():
            return candidate
        for child in p.iterdir():
            if child.is_file() and child.suffix.lower() == ".onnx":
                return child
    raise FileNotFoundError(f"ONNX model not found at: {path_like}")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x if n == 0 else x / n
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n


def _match_target_dim(arr: np.ndarray, target: int, reduce: str) -> np.ndarray:
    if arr.ndim == 1:
        if arr.shape[0] == target:
            return arr
        if arr.shape[0] > target and reduce == "truncate":
            y = arr[:target]
        else:
            y = np.zeros((target,), dtype=np.float32)
            y[: min(target, arr.shape[0])] = arr[: min(target, arr.shape[0])]
        return _l2_normalize(y)
    if arr.ndim == 2:
        if arr.shape[1] == target:
            return arr
        if arr.shape[1] > target and reduce == "truncate":
            y = arr[:, :target]
        else:
            y = np.zeros((arr.shape[0], target), dtype=np.float32)
            y[:, : min(target, arr.shape[1])] = arr[:, : min(target, arr.shape[1])]
        return _l2_normalize(y)
    return arr



class OnnxEmbeddingBackend():
    """ONNXRuntime 백엔드.

    - 토크나이저는 우선 로컬 디렉터리(`tokenizer_path`)에서 로드, 없으면 HF id(`tokenizer_id`)로 로드
    - 출력은 반드시 `sentence_embedding`(B, D) 노드를 기대
    """

    def __init__(self, model_path: str, tokenizer_id: str, tokenizer_path: Optional[str]) -> None:
        import onnxruntime as ort  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        onnx_path = _resolve_onnx_path(model_path)
        
        # ONNX 런타임 최적화 설정
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0  # 모든 CPU 코어 사용
        sess_options.inter_op_num_threads = 0  # 모든 CPU 코어 사용
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 병렬 실행
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 모든 최적화 활성화
        
        # 사용 가능한 프로바이더 (CPU 최적화 우선)
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'TensorrtExecutionProvider')
            
        self._session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        if tokenizer_path:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, int(getattr(self._session, "hidden_size", 0)) or 1), dtype=np.float32)
        batch_size = 64
        outputs_all: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=getattr(self._tokenizer, "model_max_length", 512),
                return_tensors="np",
                add_special_tokens=True,
                return_token_type_ids=False,  # token_type_ids 제외
            )
            ort_inputs: Dict[str, Any] = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            if "token_type_ids" in ort_inputs and ort_inputs["token_type_ids"] is None:
                ort_inputs.pop("token_type_ids")
            ort_outputs = self._session.run(None, ort_inputs)
            out_names = [o.name for o in self._session.get_outputs()]

            sent_idx = None
            for i, name in enumerate(out_names):
                if "sentence_embedding" in name:
                    sent_idx = i
                    break
            if sent_idx is None:
                raise RuntimeError("ONNX 모델 출력에 'sentence_embedding' 노드가 없습니다. sentence-level ONNX로 내보내세요.")
            pooled = ort_outputs[sent_idx]
            if pooled.ndim != 2:
                raise RuntimeError("'sentence_embedding' 출력의 차원이 예상(B, D)과 다릅니다.")
            outputs_all.append(_l2_normalize(pooled))
        return np.vstack(outputs_all)


class Embedder:
    """임베딩 파사드.

    - ONNX sentence-level 모델만 지원(폴백 없음)
    - 출력은 항상 설정된 차원(`embedding_output_dim`)으로 정규화/축소한다
    """

    def __init__(self, model_name: Optional[str] = None, require: bool = False) -> None:
        # 항상 ONNX만 사용한다고 가정
        self.model_name = model_name or settings.embedding_model
        self.target_dim = int(getattr(settings, "embedding_output_dim", 384))
        self.reduce = (getattr(settings, "embedding_reduce", "truncate") or "truncate").lower()

        onnx_path = getattr(settings, "onnx_model_path", None)
        if not onnx_path:
            raise RuntimeError("ONNX 모델 경로(ONNX_MODEL_PATH)가 설정되어 있지 않습니다.")

        try:
            self._backend = OnnxEmbeddingBackend(
                model_path=onnx_path,
                tokenizer_id=(getattr(settings, "onnx_tokenizer_id", None) or self.model_name),
                tokenizer_path=getattr(settings, "onnx_tokenizer_path", None),
            )
        except Exception as e:
            # 다른 폴백은 제공하지 않음
            raise RuntimeError(f"ONNX 임베딩 백엔드 초기화 실패: {e}")


    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._backend.embed(texts)  # (N, D')
        vecs = _match_target_dim(vecs, self.target_dim, self.reduce)
        return [row.astype(float).tolist() for row in vecs]


