from typing import Optional

from qloopbb.router import RouteDecision, RouteKind


def build_chatter(route: RouteDecision) -> Optional[str]:
    if route.safe_handoff or route.kind == RouteKind.EMPTY:
        return None
    if route.kind == RouteKind.APPOINTMENT:
        return "Let me check the scheduling details."
    if route.kind == RouteKind.BILLING:
        return "Let me pull up the billing desk details."
    if route.kind == RouteKind.DEPARTMENT_LOOKUP:
        return "Let me check the right department."
    if route.kind == RouteKind.POLICY_LOOKUP:
        return "Let me look that up."
    return None
