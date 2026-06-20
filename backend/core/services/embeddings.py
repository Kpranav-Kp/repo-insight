# backend/core/services/embeddings.py
import json
import os
from collections import OrderedDict
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self.index = None
        self.metadata = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self._cache = OrderedDict()
        self._cache_maxsize = 1000

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 1:
            key = texts[0].strip()
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        texts = [t.strip() if t else "" for t in texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        vectors = embeddings.astype("float32")

        if len(texts) == 1:
            key = texts[0]
            self._cache[key] = vectors
            if len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)

        return vectors

    def build_index(self, items: list[tuple[str, str, dict]]):
        if not items:
            self.index = None
            return

        ids, texts, metadatas = zip(*items, strict=False)
        vectors = self.encode(list(texts))

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)  # type: ignore
        self.metadata = {}
        self.id_to_index = {}
        self.index_to_id = {}

        for idx, (id_, meta) in enumerate(zip(ids, metadatas, strict=False)):
            self.metadata[id_] = meta
            self.id_to_index[id_] = idx
            self.index_to_id[idx] = id_

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.encode([query])
        scores, indices = self.index.search(query_vec, top_k)  # type: ignore

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue

            id_ = self.index_to_id.get(int(idx))
            if id_:
                results.append(
                    SearchResult(
                        id=id_, score=float(score), metadata=self.metadata.get(id_, {})
                    )
                )

        return results

    def save(self, path: str):
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")

        with open(f"{path}.json", "w") as f:
            json.dump(
                {
                    "metadata": self.metadata,
                    "id_to_index": self.id_to_index,
                    "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
                    "model_name": self.model_name,
                    "dimension": self.dimension,
                },
                f,
            )

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        if self.index.d != self.dimension:
            raise ValueError(
                f"Index dimension {self.index.d} doesn't match expected {self.dimension}"
            )
        with open(f"{path}.json") as f:
            data = json.load(f)
            self.metadata = data["metadata"]
            self.id_to_index = {k: int(v) for k, v in data["id_to_index"].items()}
            self.index_to_id = {int(k): v for k, v in data["index_to_id"].items()}
            self.model_name = data.get("model_name", self.model_name)
            self.dimension = data.get("dimension", self.dimension)
