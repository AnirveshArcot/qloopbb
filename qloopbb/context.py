from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

from qloopbb.embeddings import SearchResult
from qloopbb.retrieval import LocalRetriever, normalize_query_text
from qloopbb.router import RouteDecision
from qloopbb.tools import LocalHospitalTools, ToolResult


@dataclass(frozen=True)
class ContextResolution:
    tool_result: Optional[ToolResult]
    retrieval_results: Optional[List[SearchResult]]
    tool_seconds: Optional[float] = None
    retrieval_seconds: Optional[float] = None
    reused_prewarmed_results: bool = False


def resolve_route_context(
    route: RouteDecision,
    transcript: str,
    tools: Optional[LocalHospitalTools],
    retriever: Optional[LocalRetriever],
    top_k: int,
    prewarmed_results: Optional[List[SearchResult]] = None,
    prewarmed_query_key: str = "",
) -> ContextResolution:
    tool_result = None
    tool_seconds = None
    if tools is not None and route.tool_name is not None:
        tool_start = perf_counter()
        tool_result = tools.run(route, transcript)
        tool_seconds = perf_counter() - tool_start

    retrieval_results = None
    retrieval_seconds = None
    reused_prewarmed_results = False
    if retriever is not None and route.needs_retrieval and transcript:
        final_query_key = normalize_query_text(transcript)
        if prewarmed_results is not None and final_query_key == prewarmed_query_key:
            retrieval_results = prewarmed_results
            retrieval_seconds = 0.0
            reused_prewarmed_results = True
        else:
            retrieval_start = perf_counter()
            retrieval_results = retriever.search(transcript, top_k=top_k)
            retrieval_seconds = perf_counter() - retrieval_start

    return ContextResolution(
        tool_result=tool_result,
        retrieval_results=retrieval_results,
        tool_seconds=tool_seconds,
        retrieval_seconds=retrieval_seconds,
        reused_prewarmed_results=reused_prewarmed_results,
    )
