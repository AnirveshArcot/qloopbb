from typing import List, Optional

from qloopbb.embeddings import SearchResult
from qloopbb.router import RouteDecision, RouteKind
from qloopbb.tools import ToolResult


FIXED_REPLY = "that's cool"
SAFE_RECEPTIONIST_FALLBACK = "I am a receptionist, let me connect you with a triage nurse."


def build_fixed_reply(_transcript: str) -> str:
    """Return the fixed prototype response for any transcript."""
    return FIXED_REPLY


def build_reply(
    transcript: str,
    route: Optional[RouteDecision] = None,
    tool_result: Optional[ToolResult] = None,
    retrieval_results: Optional[List[SearchResult]] = None,
) -> str:
    if route is None:
        return build_fixed_reply(transcript)

    if route.kind == RouteKind.EMPTY:
        return "I didn't catch that. Could you say it again?"

    if route.safe_handoff:
        return SAFE_RECEPTIONIST_FALLBACK

    if tool_result is not None:
        return build_tool_reply(route, tool_result)

    if route.needs_retrieval and retrieval_results:
        top_match = retrieval_results[0]
        return f"I found this routing note: {top_match.document.text}"

    if route.needs_retrieval:
        return "Let me check that information and route you to the right desk."

    return FIXED_REPLY


def build_tool_reply(route: RouteDecision, tool_result: ToolResult) -> str:
    if route.kind == RouteKind.APPOINTMENT:
        department = tool_result.data.get("department")
        if department:
            return f"I can help with that. {tool_result.summary}"
        return "I can help schedule that. Which department or reason should I use?"

    if route.kind == RouteKind.BILLING:
        location = tool_result.data.get("location", "the billing desk")
        return f"The billing desk can help with that. You can use {location}."

    if route.kind == RouteKind.DEPARTMENT_LOOKUP:
        location = tool_result.data.get("location")
        if location:
            return f"{tool_result.summary} It is at {location}."
        return "Let me check which department is best for that."

    return tool_result.summary
