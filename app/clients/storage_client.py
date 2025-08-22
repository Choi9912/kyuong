import json
from pathlib import Path
from datetime import datetime
from typing import Iterable
from typing import Any, Dict, List

from app.utils.logger import get_logger


logger = get_logger(__name__)


class StorageClient:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _output_dir(self, output_name: str, use_subdir: bool = True) -> Path:
        if use_subdir:
            d = self.base_dir / output_name
            d.mkdir(parents=True, exist_ok=True)
            return d
        return self.base_dir

    def save_chunks_json(self, output_name: str, chunks: List[Dict[str, Any]], use_subdir: bool = True) -> str:
        path = self._output_dir(output_name, use_subdir) / f"chunks_{output_name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"chunks": chunks, "count": len(chunks)}, f, ensure_ascii=False)
        logger.info("청크 JSON 저장: %s", path)
        return str(path)

    def save_chunks_jsonl(self, output_name: str, chunks: Iterable[Dict[str, Any]], use_subdir: bool = True) -> str:
        path = self._output_dir(output_name, use_subdir) / f"chunks_{output_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for obj in chunks:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        logger.info("청크 JSONL 저장: %s", path)
        return str(path)

    def save_embeddings_json(self, output_name: str, vectors: List[Dict[str, Any]], use_subdir: bool = True) -> str:
        """Save only id + vector list for each chunk."""
        path = self._output_dir(output_name, use_subdir) / f"embeddings_{output_name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"chunks": vectors, "count": len(vectors)}, f, ensure_ascii=False)
        logger.info("임베딩 JSON 저장: %s", path)
        return str(path)

    def save_embeddings_npy(
        self,
        output_name: str,
        matrix: Any,
        use_subdir: bool = True,
    ) -> str:
        import numpy as np  # local import to keep optional

        out_dir = self._output_dir(output_name, use_subdir)
        npy_path = out_dir / f"embeddings_{output_name}.npy"
        with open(npy_path, "wb") as f:
            np.save(f, matrix)
        logger.info("임베딩 NPY 저장: %s", npy_path)
        return str(npy_path)

    def save_manifest(
        self,
        output_name: str,
        meta: Dict[str, Any],
        use_subdir: bool = True,
    ) -> str:
        path = self._output_dir(output_name, use_subdir) / "manifest.json"
        meta = {**meta, "output_name": output_name, "created_at": datetime.utcnow().isoformat() + "Z"}
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("매니페스트 저장: %s", path)
        return str(path)


