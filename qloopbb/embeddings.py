from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_EMBEDDING_CACHE_DIR = Path("models") / "embeddings"


Metadata = Dict[str, str]


@dataclass(frozen=True)
class VectorDocument:
    id: str
    text: str
    metadata: Metadata


@dataclass(frozen=True)
class SearchResult:
    document: VectorDocument
    score: float


class LocalEmbeddingModel:
    """Local text embeddings backed by fastembed."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        threads: Optional[int] = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=threads,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        prepared = [self._prepare_document(text) for text in texts]
        return normalize_matrix(list(self._model.embed(prepared, batch_size=batch_size)))

    def embed_query(self, text: str) -> np.ndarray:
        embedding = next(self._model.embed([self._prepare_query(text)], batch_size=1))
        return normalize_vector(embedding)

    def _prepare_query(self, text: str) -> str:
        if self._uses_e5_prefixes:
            return f"query: {text}"
        return text

    def _prepare_document(self, text: str) -> str:
        if self._uses_e5_prefixes:
            return f"passage: {text}"
        return text

    @property
    def _uses_e5_prefixes(self) -> bool:
        return "e5" in self.model_name.lower()


class InMemoryVectorIndex:
    """Small normalized cosine index for prototype retrieval."""

    def __init__(self, documents: Sequence[VectorDocument], embeddings: np.ndarray) -> None:
        if len(documents) == 0:
            raise ValueError("documents must not be empty")
        if len(documents) != embeddings.shape[0]:
            raise ValueError("documents and embeddings must have the same length")

        self.documents = list(documents)
        self.embeddings = normalize_matrix(embeddings)

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        embedding_model: LocalEmbeddingModel,
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Metadata]] = None,
    ) -> "InMemoryVectorIndex":
        if len(texts) == 0:
            raise ValueError("texts must not be empty")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids must match texts length")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must match texts length")

        documents = [
            VectorDocument(
                id=ids[index] if ids is not None else str(index + 1),
                text=text,
                metadata=metadatas[index] if metadatas is not None else {},
            )
            for index, text in enumerate(texts)
        ]
        embeddings = embedding_model.embed_documents(texts)
        return cls(documents=documents, embeddings=embeddings)

    def search(
        self,
        query: str,
        embedding_model: LocalEmbeddingModel,
        top_k: int = 3,
    ) -> List[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = embedding_model.embed_query(query)
        scores = self.embeddings @ query_embedding
        ranked_indexes = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                document=self.documents[index],
                score=float(scores[index]),
            )
            for index in ranked_indexes
        ]


def normalize_matrix(vectors: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(list(vectors), dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array
    return array / norm
