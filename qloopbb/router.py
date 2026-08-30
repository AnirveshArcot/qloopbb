import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class RouteKind(str, Enum):
    EMPTY = "empty"
    DIRECT = "direct"
    EMERGENCY = "emergency"
    MEDICAL_ADVICE = "medical_advice"
    APPOINTMENT = "appointment"
    BILLING = "billing"
    DEPARTMENT_LOOKUP = "department_lookup"
    POLICY_LOOKUP = "policy_lookup"


@dataclass(frozen=True)
class RouteDecision:
    kind: RouteKind
    confidence: float
    reason: str
    tool_name: Optional[str] = None
    needs_retrieval: bool = False
    safe_handoff: bool = False


KeywordPattern = Tuple[str, Sequence[str]]


EMERGENCY_PATTERNS: Sequence[KeywordPattern] = (
    ("breathing emergency", ("can't breathe", "cannot breathe", "not breathing")),
    ("cardiac emergency", ("heart attack", "severe chest pain")),
    ("stroke emergency", ("stroke", "face drooping", "slurred speech")),
    ("critical injury", ("unconscious", "bleeding heavily", "heavy bleeding")),
    ("self harm risk", ("suicidal", "kill myself", "hurt myself")),
)

MEDICAL_ADVICE_PATTERNS: Sequence[KeywordPattern] = (
    ("diagnosis request", ("diagnose", "what do i have", "what is wrong with me")),
    ("treatment request", ("what medicine", "which medicine", "dosage", "dose should i")),
    ("clinical advice request", ("should i take", "is it safe to take", "can i take")),
)

APPOINTMENT_TERMS = (
    "appointment",
    "book",
    "schedule",
    "reschedule",
    "cancel",
    "doctor",
    "visit",
    "mri",
    "ct scan",
    "ultrasound",
    "x ray",
    "x-ray",
    "colonoscopy",
)

BILLING_TERMS = (
    "bill",
    "billing",
    "invoice",
    "payment",
    "insurance",
    "claim",
    "receipt",
)

POLICY_TERMS = (
    "policy",
    "visiting hours",
    "visitor",
    "parking",
    "directions",
    "records",
    "medical record",
)

DEPARTMENT_TERMS = (
    "department",
    "orthopedics",
    "orthopedic",
    "radiology",
    "gastroenterology",
    "cardiology",
    "knee",
    "hip",
    "shoulder",
    "fracture",
    "bone",
    "joint",
    "stomach",
    "digestion",
    "liver",
    "chest pain follow up",
    "blood pressure",
    "ecg",
)

DIRECT_TERMS = (
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "okay",
    "cool",
)


def route_utterance(text: str) -> RouteDecision:
    normalized = normalize_for_routing(text)
    if not normalized:
        return RouteDecision(
            kind=RouteKind.EMPTY,
            confidence=1.0,
            reason="No usable transcript text.",
        )

    matched_reason = first_keyword_reason(normalized, EMERGENCY_PATTERNS)
    if matched_reason:
        return RouteDecision(
            kind=RouteKind.EMERGENCY,
            confidence=0.95,
            reason=matched_reason,
            tool_name="safe_transfer",
            safe_handoff=True,
        )

    matched_reason = first_keyword_reason(normalized, MEDICAL_ADVICE_PATTERNS)
    if matched_reason:
        return RouteDecision(
            kind=RouteKind.MEDICAL_ADVICE,
            confidence=0.9,
            reason=matched_reason,
            tool_name="safe_transfer",
            safe_handoff=True,
        )

    if any_term(normalized, BILLING_TERMS):
        return RouteDecision(
            kind=RouteKind.BILLING,
            confidence=0.85,
            reason="Billing or insurance keyword matched.",
            tool_name="billing_lookup",
        )

    if any_term(normalized, APPOINTMENT_TERMS):
        return RouteDecision(
            kind=RouteKind.APPOINTMENT,
            confidence=0.8,
            reason="Scheduling keyword matched.",
            tool_name="appointment_lookup",
        )

    if any_term(normalized, POLICY_TERMS):
        return RouteDecision(
            kind=RouteKind.POLICY_LOOKUP,
            confidence=0.75,
            reason="Policy or hospital information keyword matched.",
            needs_retrieval=True,
        )

    if any_term(normalized, DEPARTMENT_TERMS):
        return RouteDecision(
            kind=RouteKind.DEPARTMENT_LOOKUP,
            confidence=0.75,
            reason="Department or symptom routing keyword matched.",
            needs_retrieval=True,
            tool_name="department_lookup",
        )

    if any_term(normalized, DIRECT_TERMS):
        return RouteDecision(
            kind=RouteKind.DIRECT,
            confidence=0.65,
            reason="Conversational keyword matched.",
        )

    return RouteDecision(
        kind=RouteKind.DIRECT,
        confidence=0.35,
        reason="No specialized route matched.",
    )


def normalize_for_routing(text: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return re.sub(r"\s+", " ", collapsed)


def first_keyword_reason(text: str, patterns: Sequence[KeywordPattern]) -> Optional[str]:
    for reason, terms in patterns:
        if any_term(text, terms):
            return reason
    return None


def any_term(text: str, terms: Sequence[str]) -> bool:
    return any(keyword_in_text(text, term) for term in terms)


def keyword_in_text(text: str, term: str) -> bool:
    normalized_term = normalize_for_routing(term)
    if not normalized_term:
        return False
    pattern = r"(^|\s)" + re.escape(normalized_term) + r"($|\s)"
    return re.search(pattern, text) is not None
