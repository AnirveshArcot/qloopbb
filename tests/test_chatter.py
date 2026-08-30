import unittest

from qloopbb.chatter import build_chatter
from qloopbb.router import RouteKind, route_utterance


class ChatterTests(unittest.TestCase):
    def test_appointment_route_gets_chatter(self) -> None:
        route = route_utterance("I need to schedule an MRI")

        self.assertEqual(
            build_chatter(route),
            "Let me check the scheduling details.",
        )

    def test_direct_route_gets_no_chatter(self) -> None:
        route = route_utterance("hello")

        self.assertIsNone(build_chatter(route))

    def test_safe_handoff_gets_no_chatter(self) -> None:
        route = route_utterance("what medicine should I take")

        self.assertEqual(route.kind, RouteKind.MEDICAL_ADVICE)
        self.assertIsNone(build_chatter(route))


if __name__ == "__main__":
    unittest.main()
