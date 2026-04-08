# embeddings.py
import json
import os
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
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self.index = None
        self.metadata = {}
        self.id_to_index = {}
        self.index_to_id = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        texts = [t.strip() if t else "" for t in texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

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

    def search_by_vector(
        self, vector: np.ndarray, top_k: int = 5
    ) -> list[SearchResult]:
        if self.index is None:
            return []

        vector = vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(vector, top_k)  # type: ignore

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

    def cosine_similarity(self, text1: str, text2: str) -> float:
        vecs = self.encode([text1, text2])
        return float(np.dot(vecs[0], vecs[1]))

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
