from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from qloopbb.router import RouteDecision, RouteKind, normalize_for_routing


@dataclass(frozen=True)
class Department:
    name: str
    location: str
    handles: Sequence[str]


@dataclass(frozen=True)
class ToolResult:
    name: str
    summary: str
    data: Dict[str, str]


DEPARTMENTS: Sequence[Department] = (
    Department(
        name="Orthopedics",
        location="Wing B, level 2",
        handles=("knee", "hip", "shoulder", "fracture", "bone", "joint", "orthopedic"),
    ),
    Department(
        name="Radiology",
        location="Wing C, level 1",
        handles=("mri", "ct scan", "ultrasound", "x ray", "x-ray", "imaging", "scan"),
    ),
    Department(
        name="Gastroenterology",
        location="Wing A, level 3",
        handles=("stomach", "digestion", "colonoscopy", "liver", "gastroenterology"),
    ),
    Department(
        name="Cardiology",
        location="Wing D, level 2",
        handles=("heart", "cardiology", "ecg", "blood pressure", "chest pain follow up"),
    ),
)


class LocalHospitalTools:
    """Deterministic local tools until real hospital integrations exist."""

    def run(self, route: RouteDecision, query: str) -> Optional[ToolResult]:
        if route.kind in (RouteKind.EMERGENCY, RouteKind.MEDICAL_ADVICE):
            return self.safe_transfer(route)
        if route.kind == RouteKind.APPOINTMENT:
            return self.appointment_lookup(query)
        if route.kind == RouteKind.BILLING:
            return self.billing_lookup()
        if route.kind == RouteKind.DEPARTMENT_LOOKUP:
            return self.department_lookup(query)
        return None

    def safe_transfer(self, route: RouteDecision) -> ToolResult:
        if route.kind == RouteKind.EMERGENCY:
            summary = "Immediate triage handoff is required."
            destination = "triage nurse"
        else:
            summary = "Medical advice request must be handed to qualified clinical staff."
            destination = "triage nurse"
        return ToolResult(
            name="safe_transfer",
            summary=summary,
            data={"destination": destination, "reason": route.reason},
        )

    def appointment_lookup(self, query: str) -> ToolResult:
        department = best_department_match(query)
        if department is None:
            return ToolResult(
                name="appointment_lookup",
                summary="Scheduling intent detected, but the department is unclear.",
                data={"next_step": "ask_department_or_reason"},
            )

        return ToolResult(
            name="appointment_lookup",
            summary=f"Likely scheduling destination is {department.name}.",
            data={
                "department": department.name,
                "location": department.location,
                "next_step": "collect scheduling details",
            },
        )

    def billing_lookup(self) -> ToolResult:
        return ToolResult(
            name="billing_lookup",
            summary="Billing desk handles insurance, claims, invoices, receipts, and payments.",
            data={"department": "Billing", "location": "Main lobby, counter 4"},
        )

    def department_lookup(self, query: str) -> ToolResult:
        department = best_department_match(query)
        if department is None:
            return ToolResult(
                name="department_lookup",
                summary="Department routing intent detected, but no local department matched.",
                data={"next_step": "use_retrieval_or_clarify"},
            )

        return ToolResult(
            name="department_lookup",
            summary=f"{department.name} is the likely department.",
            data={"department": department.name, "location": department.location},
        )


def best_department_match(query: str) -> Optional[Department]:
    normalized_query = normalize_for_routing(query)
    if not normalized_query:
        return None

    ranked: List[Department] = sorted(
        DEPARTMENTS,
        key=lambda department: department_score(normalized_query, department),
        reverse=True,
    )
    if not ranked:
        return None

    top = ranked[0]
    if department_score(normalized_query, top) <= 0:
        return None
    return top


def department_score(normalized_query: str, department: Department) -> int:
    score = 0
    department_name = normalize_for_routing(department.name)
    if department_name in normalized_query:
        score += 3

    for term in department.handles:
        normalized_term = normalize_for_routing(term)
        if normalized_term and normalized_term in normalized_query:
            score += 1
    return score
