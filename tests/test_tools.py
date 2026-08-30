import unittest

from qloopbb.router import route_utterance
from qloopbb.tools import LocalHospitalTools, best_department_match


class ToolTests(unittest.TestCase):
    def test_best_department_match_handles_mri(self) -> None:
        department = best_department_match("I need to schedule an MRI")

        self.assertIsNotNone(department)
        self.assertEqual(department.name, "Radiology")

    def test_best_department_match_handles_knee(self) -> None:
        department = best_department_match("My knee hurts")

        self.assertIsNotNone(department)
        self.assertEqual(department.name, "Orthopedics")

    def test_local_tools_runs_billing_lookup(self) -> None:
        route = route_utterance("I need help with an insurance claim")
        result = LocalHospitalTools().run(route, "I need help with an insurance claim")

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "billing_lookup")
        self.assertEqual(result.data["department"], "Billing")


if __name__ == "__main__":
    unittest.main()
