import unittest

from qloopbb.router import RouteKind, route_utterance


class RouterTests(unittest.TestCase):
    def test_routes_appointment_to_tool_without_retrieval(self) -> None:
        route = route_utterance("I need to schedule an MRI")

        self.assertEqual(route.kind, RouteKind.APPOINTMENT)
        self.assertEqual(route.tool_name, "appointment_lookup")
        self.assertFalse(route.needs_retrieval)

    def test_routes_department_lookup_to_retrieval(self) -> None:
        route = route_utterance("My knee hurts")

        self.assertEqual(route.kind, RouteKind.DEPARTMENT_LOOKUP)
        self.assertTrue(route.needs_retrieval)
        self.assertEqual(route.tool_name, "department_lookup")

    def test_routes_billing_to_tool_without_retrieval(self) -> None:
        route = route_utterance("Can you help with my insurance claim?")

        self.assertEqual(route.kind, RouteKind.BILLING)
        self.assertEqual(route.tool_name, "billing_lookup")
        self.assertFalse(route.needs_retrieval)

    def test_routes_medical_advice_to_safe_handoff(self) -> None:
        route = route_utterance("What medicine should I take for this?")

        self.assertEqual(route.kind, RouteKind.MEDICAL_ADVICE)
        self.assertTrue(route.safe_handoff)
        self.assertEqual(route.tool_name, "safe_transfer")

    def test_routes_policy_lookup_to_retrieval(self) -> None:
        route = route_utterance("What are the visiting hours?")

        self.assertEqual(route.kind, RouteKind.POLICY_LOOKUP)
        self.assertTrue(route.needs_retrieval)


if __name__ == "__main__":
    unittest.main()
