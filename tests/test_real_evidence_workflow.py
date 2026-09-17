"""Regression coverage for the opt-in bounded real-video evidence lane."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RealEvidenceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads((ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8"))
        self.steps = self.workflow["jobs"]["linux"]["steps"]
        matches = [step for step in self.steps if step.get("name") == "Bounded real-video CPU evidence"]
        self.assertEqual(len(matches), 1)
        self.step = matches[0]

    def test_evidence_lane_is_pull_request_and_branch_scoped(self):
        condition = self.step.get("if", "")
        self.assertIn("github.event_name == 'pull_request'", condition)
        self.assertIn("startsWith(github.head_ref, 'evidence/')", condition)
        self.assertIn("steps.scope.outputs.run_full_tests == 'true'", condition)

    def test_evidence_lane_is_bounded_and_non_retaining(self):
        command = self.step.get("run", "")
        self.assertIn("openvino==2026.3.1", command)
        self.assertIn("opencv-python-headless==4.12.0.88", command)
        self.assertIn("analytics_lab.validation_seed", command)
        self.assertIn("analytics_lab.validation_cli", command)
        self.assertIn("analytics_lab.detector_diagnostics", command)
        self.assertIn("$RUNNER_TEMP/analytics-validation", command)
        self.assertNotIn("upload-artifact", command)
        self.assertNotIn("self-hosted", json.dumps(self.workflow))

    def test_evidence_lane_does_not_change_permissions_or_job_budget(self):
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        linux = self.workflow["jobs"]["linux"]
        self.assertEqual(linux["runs-on"], "ubuntu-24.04")
        self.assertLessEqual(linux["timeout-minutes"], 5)


if __name__ == "__main__":
    unittest.main()
