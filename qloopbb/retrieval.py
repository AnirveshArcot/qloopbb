from pathlib import Path
import re
from time import perf_counter
from typing import List, Optional

from qloopbb.embeddings import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    InMemoryVectorIndex,
    LocalEmbeddingModel,
    SearchResult,
)


SAMPLE_DOCUMENTS = [
    "Orthopedics handles bone, joint, knee, hip, shoulder, and fracture appointments.",
    "Radiology schedules MRI, CT scan, ultrasound, X-ray, and imaging appointments.",
    "Gastroenterology handles stomach pain, digestion issues, colonoscopy, and liver concerns.",
    "Cardiology handles chest pain follow-ups, heart rhythm checks, ECG, and blood pressure concerns.",
    "The billing desk helps with insurance questions, claim status, invoices, and payment receipts.",
]


class LocalRetriever:
    def __init__(
        self,
        documents: List[str],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        threads: Optional[int] = None,
        show_timings: bool = False,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")

        print(f"Loading embedding model: {model_name}")
        model_start = perf_counter()
        self.embedding_model = LocalEmbeddingModel(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
        )
        print_timing(show_timings, "embedding model load", perf_counter() - model_start)

        index_start = perf_counter()
        self.index = InMemoryVectorIndex.from_texts(
            texts=documents,
            embedding_model=self.embedding_model,
        )
        print_timing(show_timings, "retrieval index build", perf_counter() - index_start)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        return self.index.search(
            query=query,
            embedding_model=self.embedding_model,
            top_k=top_k,
        )


def load_retrieval_documents(
    docs: Optional[List[str]] = None,
    docs_file: Optional[Path] = None,
) -> List[str]:
    documents: List[str] = []
    if docs:
        documents.extend(docs)

    if docs_file:
        file_documents = [
            line.strip()
            for line in docs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        documents.extend(file_documents)

    if not documents:
        documents.extend(SAMPLE_DOCUMENTS)

    return documents


def print_search_results(results: List[SearchResult]) -> None:
    print("Retrieval matches:")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result.score:.4f} id={result.document.id}")
        print(f"   {result.document.text}")


def print_timing(enabled: bool, label: str, seconds: float) -> None:
    if enabled:
        print(f"Timing: {label}: {seconds * 1000:.1f} ms")


def normalize_query_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
