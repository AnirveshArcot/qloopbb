import argparse
from pathlib import Path
from time import perf_counter

from qloopbb.embeddings import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from qloopbb.retrieval import LocalRetriever, load_retrieval_documents, print_search_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed English text and run a small retrieval smoke test."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query to embed.",
    )
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        help="Document text to index. Can be passed multiple times.",
    )
    parser.add_argument(
        "--docs-file",
        type=Path,
        help="Optional newline-delimited text file of documents to index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of matches to return.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="fastembed model name.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE_DIR,
        help="Directory used for downloaded embedding models.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Optional number of inference threads.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print model, indexing, and query timing information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_retrieval_documents(
        docs=args.docs,
        docs_file=args.docs_file,
    )
    retriever = LocalRetriever(
        documents=documents,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        threads=args.threads,
        show_timings=args.timings,
    )
    search_start = perf_counter()
    results = retriever.search(args.query, top_k=args.top_k)
    search_seconds = perf_counter() - search_start

    print(f"Query: {args.query}")
    print_search_results(results)
    if args.timings:
        print(f"Timing: retrieval search: {search_seconds * 1000:.1f} ms")


if __name__ == "__main__":
    main()
